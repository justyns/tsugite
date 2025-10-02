"""Agent execution engine using smolagents."""

from pathlib import Path
from typing import Any, Dict, List, Optional

from smolagents import CodeAgent

from tsugite.md_agents import AgentConfig, parse_agent, validate_agent_execution
from tsugite.models import get_model
from tsugite.renderer import AgentRenderer
from tsugite.tool_adapter import get_smolagents_tools
from tsugite.tools import call_tool
from tsugite.tools.tasks import get_task_manager, reset_task_manager

TSUGITE_DEFAULT_INSTRUCTIONS = (
    "You are operating inside the Tsugite micro-agent runtime. Follow the rendered task faithfully, use the available "
    "tools when they meaningfully advance the work, and maintain a living plan via the task_* tools. Create or update "
    "tasks whenever you define new sub-work, mark progress as you go, and rely on the task summary to decide the next "
    "action. Provide a clear, actionable final response without unnecessary filler."
)


def _combine_instructions(*segments: str) -> str:
    """Join instruction segments, skipping empties."""

    parts = [segment.strip() for segment in segments if segment and segment.strip()]
    return "\n\n".join(parts)


def execute_prefetch(prefetch_config: List[Dict[str, Any]]) -> Dict[str, Any]:
    context = {}
    for config in prefetch_config:
        tool_name = config.get("tool")
        args = config.get("args", {})
        assign_name = config.get("assign")

        if not tool_name or not assign_name:
            continue

        try:
            context[assign_name] = call_tool(tool_name, **args)
        except Exception as e:
            print(f"Warning: Prefetch tool '{tool_name}' failed: {e}")
            context[assign_name] = None

    return context


def _execute_agent_with_prompt(
    rendered_prompt: str,
    agent_config: AgentConfig,
    model_override: Optional[str] = None,
    custom_logger: Optional[Any] = None,
    trust_mcp_code: bool = False,
    delegation_agents: Optional[List[tuple[str, Path]]] = None,
    skip_task_reset: bool = False,
) -> str:
    """Execute agent with a pre-rendered prompt.

    Low-level execution function used by both run_agent and run_multistep_agent.

    Args:
        rendered_prompt: Pre-rendered prompt to execute
        agent_config: Agent configuration
        model_override: Override agent's model
        custom_logger: Custom logger
        trust_mcp_code: Trust MCP server code
        delegation_agents: Delegation agents list
        skip_task_reset: Skip resetting task manager (for multi-step agents)

    Returns:
        Agent execution result

    Raises:
        RuntimeError: If execution fails
    """
    # Initialize task manager for this execution (unless skipped for multi-step)
    if not skip_task_reset:
        reset_task_manager()

    combined_instructions = _combine_instructions(
        TSUGITE_DEFAULT_INSTRUCTIONS,
        getattr(agent_config, "instructions", ""),
    )

    # Create smolagents tools
    try:
        task_tools = ["task_add", "task_update", "task_complete", "task_list", "task_get"]
        all_tools = list(agent_config.tools) + task_tools

        if delegation_agents:
            all_tools.append("spawn_agent")

        tools = get_smolagents_tools(all_tools)
    except Exception as e:
        raise RuntimeError(f"Failed to create tools: {e}")

    # Add delegation tools if provided
    if delegation_agents:
        from .agent_composition import create_delegation_tools

        delegation_tools = create_delegation_tools(delegation_agents)
        tools.extend(delegation_tools)

    # Load MCP tools if configured
    if agent_config.mcp_servers:
        try:
            from tsugite.mcp_config import load_mcp_config
            from tsugite.mcp_integration import load_all_mcp_tools

            global_mcp_config = load_mcp_config()
            mcp_tools = load_all_mcp_tools(agent_config.mcp_servers, global_mcp_config, trust_mcp_code)
            tools.extend(mcp_tools)
        except Exception as e:
            print(f"Warning: Failed to load MCP tools: {e}")
            print("Continuing without MCP tools.")

    # Create model
    model_string = model_override or agent_config.model
    if not model_string:
        from tsugite.config import load_config

        config = load_config()
        model_string = config.default_model

    if not model_string:
        raise RuntimeError(
            "No model specified. Set a model in agent frontmatter, use --model flag, "
            "or set a default with 'tsugite config set-default <model>'"
        )

    try:
        model = get_model(model_string)
    except Exception as e:
        raise RuntimeError(f"Failed to create model '{model_string}': {e}")

    # Create and run agent
    try:
        agent_kwargs = {
            "tools": tools,
            "model": model,
            "max_steps": agent_config.max_steps,
            "instructions": combined_instructions or None,
        }

        if custom_logger is not None:
            agent_kwargs["logger"] = custom_logger
            agent_kwargs["verbosity_level"] = -1

        agent = CodeAgent(**agent_kwargs)
        result = agent.run(rendered_prompt)
        return str(result)

    except Exception as e:
        raise RuntimeError(f"Agent execution failed: {e}")


def run_agent(
    agent_path: Path,
    prompt: str,
    context: Optional[Dict[str, Any]] = None,
    model_override: Optional[str] = None,
    debug: bool = False,
    custom_logger: Optional[Any] = None,
    trust_mcp_code: bool = False,
    delegation_agents: Optional[List[tuple[str, Path]]] = None,
) -> str:
    """Run a Tsugite agent using smolagents.

    Args:
        agent_path: Path to agent markdown file
        prompt: User prompt/task for the agent
        context: Additional context variables
        model_override: Override agent's default model
        debug: Enable debug output (rendered prompt)
        custom_logger: Custom logger for agent output
        trust_mcp_code: Whether to trust remote code from MCP servers
        delegation_agents: List of (name, path) tuples for agents to make available for delegation

    Returns:
        Agent execution result as string

    Raises:
        ValueError: If agent file is invalid
        RuntimeError: If agent execution fails
    """
    if context is None:
        context = {}

    # Initialize task manager for this agent session
    reset_task_manager()
    task_manager = get_task_manager()

    # Parse agent configuration
    try:
        agent_text = agent_path.read_text()
        agent = parse_agent(agent_text, agent_path)
        agent_config = agent.config
    except Exception as e:
        raise ValueError(f"Failed to parse agent file: {e}")

    # Execute prefetch tools if any
    prefetch_context = {}
    if agent_config.prefetch:
        try:
            prefetch_context = execute_prefetch(agent_config.prefetch)
        except Exception as e:
            print(f"Warning: Prefetch execution failed: {e}")

    # Add task context (will be updated as agent creates tasks)
    task_context = task_manager.get_task_summary()

    # Prepare full context for template rendering
    full_context = {
        **context,
        **prefetch_context,
        "user_prompt": prompt,
        "task_summary": task_context,
    }

    # Render agent template
    renderer = AgentRenderer()
    try:
        rendered_prompt = renderer.render(agent.content, full_context)

        if debug:
            print("\n" + "=" * 60)
            print("DEBUG: Rendered Prompt")
            print("=" * 60)
            print(rendered_prompt)
            print("=" * 60 + "\n")

    except Exception as e:
        raise ValueError(f"Template rendering failed: {e}")

    # Execute with the low-level helper
    return _execute_agent_with_prompt(
        rendered_prompt=rendered_prompt,
        agent_config=agent_config,
        model_override=model_override,
        custom_logger=custom_logger,
        trust_mcp_code=trust_mcp_code,
        delegation_agents=delegation_agents,
    )


def validate_agent_file(agent_path: Path) -> tuple[bool, str]:
    """Validate that an agent file can be executed.

    Args:
        agent_path: Path to agent markdown file

    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        # Parse agent
        agent_text = agent_path.read_text()
        agent = parse_agent(agent_text, agent_path)

        # Use centralized validation
        return validate_agent_execution(agent)

    except Exception as e:
        return False, f"Agent file validation failed: {e}"


def get_agent_info(agent_path: Path) -> Dict[str, Any]:
    """Get information about an agent without executing it.

    Args:
        agent_path: Path to agent markdown file

    Returns:
        Dictionary with agent information
    """
    try:
        agent_text = agent_path.read_text()
        agent = parse_agent(agent_text, agent_path)
        agent_config = agent.config

        model_display = agent_config.model
        if not model_display:
            from tsugite.config import load_config

            config = load_config()
            if config.default_model:
                model_display = f"{config.default_model} (default)"
            else:
                model_display = "not set"

        return {
            "name": agent_config.name,
            "description": getattr(agent_config, "description", "No description"),
            "model": model_display,
            "max_steps": agent_config.max_steps,
            "tools": agent_config.tools,
            "prefetch_count": (len(agent_config.prefetch) if agent_config.prefetch else 0),
            "permissions_profile": getattr(agent_config, "permissions_profile", None),
            "valid": validate_agent_file(agent_path)[0],
            "instructions": getattr(agent_config, "instructions", ""),
        }
    except Exception as e:
        return {
            "error": str(e),
            "valid": False,
        }


def run_multistep_agent(
    agent_path: Path,
    prompt: str,
    context: Optional[Dict[str, Any]] = None,
    model_override: Optional[str] = None,
    debug: bool = False,
    custom_logger: Optional[Any] = None,
    trust_mcp_code: bool = False,
    delegation_agents: Optional[List[tuple[str, Path]]] = None,
) -> str:
    """Run a multi-step Tsugite agent.

    Multi-step agents use <!-- tsu:step --> directives to execute sequentially,
    with each step being a full agent run. Results from earlier steps can be
    assigned to variables and used in later steps.

    Args:
        agent_path: Path to agent markdown file
        prompt: User prompt/task for the agent
        context: Additional context variables
        model_override: Override agent's default model
        debug: Enable debug output (rendered prompts for each step)
        custom_logger: Custom logger for agent output
        trust_mcp_code: Whether to trust remote code from MCP servers
        delegation_agents: List of (name, path) tuples for agents to make available

    Returns:
        Result from the final step

    Raises:
        ValueError: If agent file is invalid or step parsing fails
        RuntimeError: If any step execution fails
    """
    from tsugite.md_agents import extract_step_directives, has_step_directives

    if context is None:
        context = {}

    # Read and parse agent
    try:
        agent_text = agent_path.read_text()
        agent = parse_agent(agent_text, agent_path)
    except Exception as e:
        raise ValueError(f"Failed to parse agent file: {e}")

    # Extract steps from raw markdown (before any rendering)
    if not has_step_directives(agent.content):
        raise ValueError(f"Agent {agent_path} does not contain step directives. Use run_agent() instead.")

    try:
        preamble, steps = extract_step_directives(agent.content)
    except Exception as e:
        raise ValueError(f"Failed to parse step directives: {e}")

    if not steps:
        raise ValueError("No valid step directives found in agent")

    # Validate unique step names
    step_names = [s.name for s in steps]
    if len(step_names) != len(set(step_names)):
        duplicates = [name for name in step_names if step_names.count(name) > 1]
        raise ValueError(f"Duplicate step names found: {', '.join(set(duplicates))}")

    # Initialize task manager ONCE for entire multi-step execution
    # This allows tasks to persist and accumulate across steps
    reset_task_manager()
    task_manager = get_task_manager()

    # Initialize context with user prompt
    step_context = {
        **context,
        "user_prompt": prompt,
        "task_summary": task_manager.get_task_summary(),
    }

    # Execute prefetch once (before any steps)
    if agent.config.prefetch:
        try:
            prefetch_context = execute_prefetch(agent.config.prefetch)
            step_context.update(prefetch_context)
        except Exception as e:
            print(f"Warning: Prefetch execution failed: {e}")

    # Execute each step sequentially
    final_result = None

    for i, step in enumerate(steps, 1):
        # Add step information to context for this step
        step_context["step_number"] = i
        step_context["step_name"] = step.name
        step_context["total_steps"] = len(steps)

        # Show step progress (unless in debug mode which has its own output)
        step_header = f"[Step {i}/{len(steps)}: {step.name}]"

        if custom_logger and not debug:
            # Set multi-step context for nested progress display
            if hasattr(custom_logger, "ui_handler"):
                custom_logger.ui_handler.set_multistep_context(i, step.name, len(steps))

            # Use custom logger's console with color
            custom_logger.console.print(f"[cyan]{step_header} Starting...[/cyan]")
        elif not debug:
            # Direct output for native-ui/silent modes
            print(f"{step_header} Starting...")

        if debug:
            print(f"\n{'=' * 60}")
            print(f"DEBUG: Executing Step {i}/{len(steps)}: {step.name}")
            print(f"{'=' * 60}")

        # Render this step's content with current context
        renderer = AgentRenderer()
        try:
            rendered_step_prompt = renderer.render(step.content, step_context)

            if debug:
                print("\nRendered Prompt:")
                print(rendered_step_prompt)
                print(f"{'=' * 60}\n")

        except Exception as e:
            raise RuntimeError(f"Step '{step.name}' template rendering failed: {e}")

        # Execute this step as a full agent run
        try:
            step_result = _execute_agent_with_prompt(
                rendered_prompt=rendered_step_prompt,
                agent_config=agent.config,
                model_override=model_override,
                custom_logger=custom_logger,
                trust_mcp_code=trust_mcp_code,
                delegation_agents=delegation_agents,
                skip_task_reset=True,  # Don't reset tasks between steps
            )

            final_result = step_result

            # Store result in context if assign variable specified
            if step.assign_var:
                step_context[step.assign_var] = step_result
                if debug:
                    print(f"Assigned result to variable: {step.assign_var}")

            # Update task summary for next step
            step_context["task_summary"] = task_manager.get_task_summary()

            # Show step completion
            if custom_logger and not debug:
                # Clear multi-step context after step completes
                if hasattr(custom_logger, "ui_handler"):
                    custom_logger.ui_handler.clear_multistep_context()

                custom_logger.console.print(f"[green]{step_header} Complete[/green]")
            elif not debug:
                print(f"{step_header} Complete")

        except Exception as e:
            # Clear multi-step context on error too
            if custom_logger and hasattr(custom_logger, "ui_handler"):
                custom_logger.ui_handler.clear_multistep_context()

            raise RuntimeError(f"Step '{step.name}' execution failed: {e}")

    return final_result or ""
