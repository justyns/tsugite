"""Agent execution engine using TsugiteAgent."""

import asyncio
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from rich.console import Console

from tsugite.core.agent import TsugiteAgent
from tsugite.core.executor import LocalExecutor
from tsugite.core.tools import create_tool_from_tsugite
from tsugite.md_agents import AgentConfig, parse_agent, validate_agent_execution
from tsugite.renderer import AgentRenderer
from tsugite.tools import call_tool
from tsugite.tools.tasks import get_task_manager, reset_task_manager
from tsugite.utils import is_interactive

# Console for warnings and debug output (stderr)
_stderr_console = Console(file=sys.stderr, no_color=False)

TSUGITE_DEFAULT_INSTRUCTIONS = (
    "You are operating inside the Tsugite micro-agent runtime. Follow the rendered task faithfully, use the available "
    "tools when they meaningfully advance the work, and maintain a living plan via the task_* tools. Create or update "
    "tasks whenever you define new sub-work, mark progress as you go, and rely on the task summary to decide the next "
    "action. Provide a clear, actionable final response without unnecessary filler.\n\n"
    "Interactive Mode: The `is_interactive` variable indicates whether you're running in an interactive terminal. "
    "Interactive-only tools (like ask_user) are automatically available only when is_interactive is True."
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
            _stderr_console.print(f"[yellow]Warning: Prefetch tool '{tool_name}' failed: {e}[/yellow]")
            context[assign_name] = None

    return context


def execute_tool_directives(
    content: str, existing_context: Optional[Dict[str, Any]] = None
) -> tuple[str, Dict[str, Any]]:
    """Execute tool directives in content and return updated context.

    Tool directives are inline <!-- tsu:tool --> comments that execute tools
    during the rendering phase, similar to prefetch but embedded in content.

    Args:
        content: Markdown content with tool directives
        existing_context: Current template context (for error messages, not used for execution)

    Returns:
        Tuple of (modified_content, updated_context)
        - modified_content: Directives replaced with execution notes
        - updated_context: Original context + tool results

    Example:
        >>> content = '<!-- tsu:tool name="read_file" args={"path": "test.txt"} assign="data" -->'
        >>> modified, context = execute_tool_directives(content)
        >>> 'data' in context
        True
    """
    from tsugite.md_agents import extract_tool_directives

    if existing_context is None:
        existing_context = {}

    # Extract tool directives
    try:
        directives = extract_tool_directives(content)
    except ValueError as e:
        # If parsing fails, return content unchanged with empty context
        _stderr_console.print(f"[yellow]Warning: Failed to parse tool directives: {e}[/yellow]")
        return content, {}

    if not directives:
        # No directives to execute
        return content, {}

    # Execute directives in order
    new_context = {}
    modified_content = content

    for directive in directives:
        try:
            # Execute the tool
            result = call_tool(directive.name, **directive.args)
            new_context[directive.assign_var] = result

            # Replace directive with execution note
            replacement = f"<!-- Tool '{directive.name}' executed, result in {directive.assign_var} -->"
            modified_content = modified_content.replace(directive.raw_match, replacement)

        except Exception as e:
            _stderr_console.print(f"[yellow]Warning: Tool directive '{directive.name}' failed: {e}[/yellow]")
            new_context[directive.assign_var] = None

            # Replace with failure note
            replacement = f"<!-- Tool '{directive.name}' failed: {e} -->"
            modified_content = modified_content.replace(directive.raw_match, replacement)

    return modified_content, new_context


def _extract_reasoning_content(agent: TsugiteAgent, custom_logger: Optional[Any] = None) -> None:
    """Extract and display reasoning content from TsugiteAgent memory.

    For models like Claude/Deepseek that expose reasoning_content, displays the actual reasoning.

    Args:
        agent: The TsugiteAgent instance that just completed execution
        custom_logger: Custom logger to display reasoning content
    """
    if not hasattr(agent, "memory") or not agent.memory.reasoning_history:
        return

    # Display each reasoning entry
    for reasoning_content in agent.memory.reasoning_history:
        if reasoning_content and custom_logger:
            # Check if custom_logger has ui_handler (custom UI mode)
            if hasattr(custom_logger, "ui_handler"):
                from tsugite.ui import UIEvent

                custom_logger.ui_handler.handle_event(
                    UIEvent.REASONING_CONTENT, {"content": reasoning_content, "step": None}
                )
            # Otherwise try to log directly (fallback)
            elif hasattr(custom_logger, "console"):
                from rich.panel import Panel

                custom_logger.console.print(
                    Panel(
                        reasoning_content,
                        title="[bold magenta]🧠 Reasoning[/bold magenta]",
                        border_style="magenta",
                    )
                )


async def _execute_agent_with_prompt(
    rendered_prompt: str,
    agent_config: AgentConfig,
    model_override: Optional[str] = None,
    custom_logger: Optional[Any] = None,
    trust_mcp_code: bool = False,
    delegation_agents: Optional[List[tuple[str, Path]]] = None,
    skip_task_reset: bool = False,
    model_kwargs: Optional[Dict[str, Any]] = None,
    injectable_vars: Optional[Dict[str, Any]] = None,
    return_token_usage: bool = False,
    stream: bool = False,
) -> str | tuple[str, Optional[int], Optional[float]]:
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
        model_kwargs: Additional model parameters (response_format, temperature, etc.)
        injectable_vars: Variables to inject into Python execution namespace
        return_token_usage: Whether to return token usage and cost from LiteLLM
        stream: Whether to stream responses in real-time

    Returns:
        Agent execution result as string, or tuple of (result, token_count, cost) if return_token_usage=True

    Raises:
        RuntimeError: If execution fails
    """
    # Initialize task manager for this execution (unless skipped for multi-step)
    if not skip_task_reset:
        reset_task_manager()

    # Build base instructions
    base_instructions = _combine_instructions(
        TSUGITE_DEFAULT_INSTRUCTIONS,
        getattr(agent_config, "instructions", ""),
    )

    # Add variable documentation if variables are available
    if injectable_vars:
        var_docs = "\n\nAVAILABLE PYTHON VARIABLES:\n"
        for var_name, var_value in injectable_vars.items():
            preview = str(var_value)[:100]
            if len(str(var_value)) > 100:
                preview += "..."
            var_docs += f"- {var_name}: {preview}\n"
        combined_instructions = base_instructions + var_docs
    else:
        combined_instructions = base_instructions

    # Create Tool objects from tsugite tools
    try:
        from tsugite.tools import expand_tool_specs

        # Expand tool specifications (categories, globs, regular names)
        expanded_tools = expand_tool_specs(agent_config.tools) if agent_config.tools else []

        # Add task management tools
        task_tools = ["task_add", "task_update", "task_complete", "task_list", "task_get"]
        all_tool_names = expanded_tools + task_tools

        if delegation_agents:
            all_tool_names.append("spawn_agent")

        # Filter out ask_user tool in non-interactive mode
        if not is_interactive() and "ask_user" in all_tool_names:
            all_tool_names.remove("ask_user")

        # Register per-agent custom shell tools (if any)
        if agent_config.custom_tools:
            from tsugite.shell_tool_config import parse_tool_definition_from_dict
            from tsugite.tools.shell_tools import register_shell_tools

            try:
                custom_tool_definitions = [
                    parse_tool_definition_from_dict(tool_dict) for tool_dict in agent_config.custom_tools
                ]
                register_shell_tools(custom_tool_definitions)

                # Add custom tool names to the tool list
                for tool_def in custom_tool_definitions:
                    all_tool_names.append(tool_def.name)
            except Exception as e:
                _stderr_console.print(f"[yellow]Warning: Failed to register custom tools: {e}[/yellow]")

        # Convert to Tool objects
        tools = [create_tool_from_tsugite(name) for name in all_tool_names]
    except Exception as e:
        raise RuntimeError(f"Failed to create tools: {e}")

    # Add delegation tools if provided
    if delegation_agents:
        from .agent_composition import create_delegation_tools

        delegation_tools = create_delegation_tools(delegation_agents)
        tools.extend(delegation_tools)

    # Load MCP tools if configured
    mcp_clients = []  # Track clients for cleanup
    if agent_config.mcp_servers:
        try:
            from tsugite.mcp_client import load_mcp_tools
            from tsugite.mcp_config import load_mcp_config

            global_mcp_config = load_mcp_config()

            # Load tools from each configured MCP server
            for server_name, allowed_tools in agent_config.mcp_servers.items():
                if server_name not in global_mcp_config:
                    _stderr_console.print(
                        f"[yellow]Warning: MCP server '{server_name}' not found in config. Skipping.[/yellow]"
                    )
                    continue

                server_config = global_mcp_config[server_name]
                try:
                    mcp_client, mcp_tools = await load_mcp_tools(server_config, allowed_tools)
                    mcp_clients.append(mcp_client)  # Keep client alive for tools to work
                    tools.extend(mcp_tools)
                    _stderr_console.print(
                        f"[green]Loaded {len(mcp_tools)} tools from MCP server '{server_name}'[/green]"
                    )
                except Exception as e:
                    _stderr_console.print(
                        f"[yellow]Warning: Failed to load MCP tools from '{server_name}': {e}[/yellow]"
                    )
        except Exception as e:
            _stderr_console.print(f"[yellow]Warning: Failed to load MCP tools: {e}[/yellow]")
            _stderr_console.print("[yellow]Continuing without MCP tools.[/yellow]")

    # Get model string
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

    # Merge reasoning_effort from agent config into model_kwargs
    final_model_kwargs = dict(model_kwargs or {})
    if hasattr(agent_config, "reasoning_effort") and agent_config.reasoning_effort:
        # Only add if not already specified in model_kwargs
        if "reasoning_effort" not in final_model_kwargs:
            final_model_kwargs["reasoning_effort"] = agent_config.reasoning_effort

    # Create executor
    executor = LocalExecutor()

    # Inject variables into executor (for multi-step agents)
    if injectable_vars:
        await executor.send_variables(injectable_vars)

    # Create and run agent
    try:
        # Extract ui_handler from custom_logger if available
        ui_handler = None
        if custom_logger and hasattr(custom_logger, "ui_handler"):
            ui_handler = custom_logger.ui_handler

        agent = TsugiteAgent(
            model_string=model_string,
            tools=tools,
            instructions=combined_instructions or "",
            max_steps=agent_config.max_steps,
            executor=executor,
            model_kwargs=final_model_kwargs,
            ui_handler=ui_handler,
            model_name=model_string,
            text_mode=agent_config.text_mode,
        )

        # Run agent
        result = await agent.run(rendered_prompt, return_full_result=return_token_usage, stream=stream)

        # Extract and display reasoning content if present
        _extract_reasoning_content(agent, custom_logger)

        # Return appropriate format
        if return_token_usage:
            from tsugite.core.agent import AgentResult

            if isinstance(result, AgentResult):
                return str(result.output), result.token_usage, result.cost
            else:
                return str(result), None, None
        else:
            from tsugite.core.agent import AgentResult

            if isinstance(result, AgentResult):
                return str(result.output)
            else:
                return str(result)

    except Exception as e:
        raise RuntimeError(f"Agent execution failed: {e}")
    finally:
        # Clean up MCP client connections
        for client in mcp_clients:
            try:
                await client.disconnect()
            except Exception:
                pass  # Best effort cleanup


def run_agent(
    agent_path: Path,
    prompt: str,
    context: Optional[Dict[str, Any]] = None,
    model_override: Optional[str] = None,
    debug: bool = False,
    custom_logger: Optional[Any] = None,
    trust_mcp_code: bool = False,
    delegation_agents: Optional[List[tuple[str, Path]]] = None,
    return_token_usage: bool = False,
    stream: bool = False,
    force_text_mode: bool = False,
) -> str | tuple[str, Optional[int], Optional[float]]:
    """Run a Tsugite agent.

    Args:
        agent_path: Path to agent markdown file
        prompt: User prompt/task for the agent
        context: Additional context variables
        model_override: Override agent's default model
        debug: Enable debug output (rendered prompt)
        custom_logger: Custom logger for agent output
        trust_mcp_code: Whether to trust remote code from MCP servers
        delegation_agents: List of (name, path) tuples for agents to make available for delegation
        return_token_usage: Whether to return token usage and cost from LiteLLM
        stream: Whether to stream responses in real-time
        force_text_mode: Force text_mode=True regardless of agent config (useful for chat UI)

    Returns:
        Agent execution result as string, or tuple of (result, token_count, cost) if return_token_usage=True

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

    # Override text_mode if force_text_mode is True (for chat UI)
    if force_text_mode:
        agent_config.text_mode = True

    # Execute prefetch tools if any
    prefetch_context = {}
    if agent_config.prefetch:
        try:
            prefetch_context = execute_prefetch(agent_config.prefetch)
        except Exception as e:
            _stderr_console.print(f"[yellow]Warning: Prefetch execution failed: {e}[/yellow]")

    # Execute tool directives in content
    modified_content, tool_context = execute_tool_directives(agent.content, prefetch_context)

    # Add task context (will be updated as agent creates tasks)
    task_context = task_manager.get_task_summary()

    # Check if running in interactive mode
    interactive_mode = is_interactive()

    # Prepare full context for template rendering
    full_context = {
        **context,
        **prefetch_context,
        **tool_context,  # Add tool directive results
        "user_prompt": prompt,
        "task_summary": task_context,
        "is_interactive": interactive_mode,
    }

    # Render agent template (use modified content with directives replaced)
    renderer = AgentRenderer()
    try:
        rendered_prompt = renderer.render(modified_content, full_context)

        if debug:
            _stderr_console.rule("[bold cyan]DEBUG: Rendered Prompt[/bold cyan]")
            _stderr_console.print(rendered_prompt)
            _stderr_console.rule("[bold cyan]End Rendered Prompt[/bold cyan]")

    except Exception as e:
        raise ValueError(f"Template rendering failed: {e}")

    # Execute with the low-level helper (wrapping async call)
    return asyncio.run(
        _execute_agent_with_prompt(
            rendered_prompt=rendered_prompt,
            agent_config=agent_config,
            model_override=model_override,
            custom_logger=custom_logger,
            trust_mcp_code=trust_mcp_code,
            delegation_agents=delegation_agents,
            return_token_usage=return_token_usage,
            stream=stream,
        )
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
            "attachments": agent_config.attachments,
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

    # Check if running in interactive mode
    interactive_mode = is_interactive()

    # Initialize context with user prompt
    step_context = {
        **context,
        "user_prompt": prompt,
        "task_summary": task_manager.get_task_summary(),
        "is_interactive": interactive_mode,
    }

    # Execute prefetch once (before any steps)
    if agent.config.prefetch:
        try:
            prefetch_context = execute_prefetch(agent.config.prefetch)
            step_context.update(prefetch_context)
        except Exception as e:
            _stderr_console.print(f"[yellow]Warning: Prefetch execution failed: {e}[/yellow]")

    # Execute each step sequentially
    final_result = None

    for i, step in enumerate(steps, 1):
        # Add step information to context for this step
        step_context["step_number"] = i
        step_context["step_name"] = step.name
        step_context["total_steps"] = len(steps)

        # Show step progress (unless in debug mode which has its own output)
        step_header = f"[Step {i}/{len(steps)}: {step.name}]"

        # Retry loop
        max_attempts = step.max_retries + 1
        errors = []

        for attempt in range(max_attempts):
            # Add retry context variables
            step_context["is_retry"] = attempt > 0
            step_context["retry_count"] = attempt
            step_context["max_retries"] = step.max_retries
            step_context["last_error"] = errors[-1] if errors else ""
            step_context["all_errors"] = errors

            if custom_logger and not debug:
                # Set multi-step context for nested progress display
                if hasattr(custom_logger, "ui_handler"):
                    custom_logger.ui_handler.set_multistep_context(i, step.name, len(steps))

                # Show retry attempt if applicable
                if attempt > 0:
                    custom_logger.console.print(f"[yellow]{step_header} Retry {attempt}/{step.max_retries}...[/yellow]")
                else:
                    custom_logger.console.print(f"[cyan]{step_header} Starting...[/cyan]")
            elif not debug:
                # Direct output for native-ui/silent modes
                if attempt > 0:
                    _stderr_console.print(f"{step_header} Retry {attempt}/{step.max_retries}...")
                else:
                    _stderr_console.print(f"{step_header} Starting...")

            if debug:
                if attempt > 0:
                    _stderr_console.rule(
                        f"[bold cyan]DEBUG: Retrying Step {i}/{len(steps)}: {step.name} (Attempt {attempt + 1}/{max_attempts})[/bold cyan]"
                    )
                else:
                    _stderr_console.rule(f"[bold cyan]DEBUG: Executing Step {i}/{len(steps)}: {step.name}[/bold cyan]")

            # Execute tool directives in this step's content
            step_modified_content, step_tool_context = execute_tool_directives(step.content, step_context)

            # Update step context with tool results
            step_context.update(step_tool_context)

            # Render this step's content with current context
            renderer = AgentRenderer()
            try:
                rendered_step_prompt = renderer.render(step_modified_content, step_context)

                if debug:
                    _stderr_console.print("\n[bold]Rendered Prompt:[/bold]")
                    _stderr_console.print(rendered_step_prompt)
                    _stderr_console.rule()

            except Exception as e:
                error_msg = f"Template rendering failed: {e}"
                errors.append(error_msg)

                if attempt == max_attempts - 1:
                    if custom_logger and hasattr(custom_logger, "ui_handler"):
                        custom_logger.ui_handler.clear_multistep_context()
                    raise RuntimeError(
                        f"Step '{step.name}' failed after {max_attempts} attempts. Errors: {'; '.join(errors)}"
                    )

                if step.retry_delay > 0:
                    time.sleep(step.retry_delay)
                continue

            # Prepare variables to inject into Python namespace
            # Filter out metadata variables, only inject step results
            metadata_vars = {
                "user_prompt",
                "task_summary",
                "step_number",
                "step_name",
                "total_steps",
                "is_retry",
                "retry_count",
                "max_retries",
                "last_error",
                "all_errors",
            }
            injectable_vars = {k: v for k, v in step_context.items() if k not in metadata_vars}

            # Execute this step as a full agent run (wrapping async call)
            try:
                step_result = asyncio.run(
                    _execute_agent_with_prompt(
                        rendered_prompt=rendered_step_prompt,
                        agent_config=agent.config,
                        model_override=model_override,
                        custom_logger=custom_logger,
                        trust_mcp_code=trust_mcp_code,
                        delegation_agents=delegation_agents,
                        skip_task_reset=True,  # Don't reset tasks between steps
                        model_kwargs=step.model_kwargs,
                        injectable_vars=injectable_vars,
                    )
                )

                final_result = step_result

                # Store result in context if assign variable specified
                if step.assign_var:
                    step_context[step.assign_var] = step_result
                    if debug:
                        _stderr_console.print(f"[dim]Assigned result to variable: {step.assign_var}[/dim]")

                # Update task summary for next step
                step_context["task_summary"] = task_manager.get_task_summary()

                # Show step completion
                if custom_logger and not debug:
                    # Clear multi-step context after step completes
                    if hasattr(custom_logger, "ui_handler"):
                        custom_logger.ui_handler.clear_multistep_context()

                    custom_logger.console.print(f"[green]{step_header} Complete[/green]")
                elif not debug:
                    _stderr_console.print(f"[green]{step_header} Complete[/green]")

                # Success - break retry loop
                break

            except Exception as e:
                error_msg = str(e)
                errors.append(error_msg)

                if attempt == max_attempts - 1:
                    if custom_logger and hasattr(custom_logger, "ui_handler"):
                        custom_logger.ui_handler.clear_multistep_context()
                    raise RuntimeError(
                        f"Step '{step.name}' failed after {max_attempts} attempts. Errors: {'; '.join(errors)}"
                    )

                if step.retry_delay > 0:
                    time.sleep(step.retry_delay)

                if custom_logger and not debug:
                    custom_logger.console.print(f"[yellow]Step '{step.name}' failed: {error_msg}[/yellow]")
                elif not debug:
                    _stderr_console.print(f"[yellow]Step '{step.name}' failed: {error_msg}[/yellow]")

    return final_result or ""
