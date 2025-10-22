"""Agent execution engine using TsugiteAgent."""

import asyncio
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from rich.console import Console

from tsugite.core.agent import TsugiteAgent
from tsugite.core.executor import LocalExecutor
from tsugite.core.tools import create_tool_from_tsugite
from tsugite.md_agents import AgentConfig, parse_agent_file, validate_agent_execution
from tsugite.renderer import AgentRenderer
from tsugite.tools import call_tool
from tsugite.tools.tasks import get_task_manager, reset_task_manager
from tsugite.utils import is_interactive

# Console for warnings and debug output (stderr)
_stderr_console = Console(file=sys.stderr, no_color=False)

# Thread-local storage for tracking currently executing agent
_current_agent_context = threading.local()


def _set_current_agent(name: str) -> None:
    """Set the name of the currently executing agent in thread-local storage."""
    _current_agent_context.name = name


def _get_current_agent() -> Optional[str]:
    """Get the name of the currently executing agent from thread-local storage."""
    return getattr(_current_agent_context, "name", None)


def _clear_current_agent() -> None:
    """Clear the currently executing agent from thread-local storage."""
    if hasattr(_current_agent_context, "name"):
        delattr(_current_agent_context, "name")


@dataclass
class StepMetrics:
    """Metrics for a single step execution."""

    step_name: str
    step_number: int
    duration: float  # seconds
    tokens_used: Optional[int] = None
    cost: Optional[float] = None
    status: str = "success"  # success, failed, skipped
    error: Optional[str] = None


def get_default_instructions(text_mode: bool = False) -> str:
    """Get default instructions based on agent mode.

    Args:
        text_mode: Whether agent is in text mode

    Returns:
        Mode-appropriate default instructions
    """
    base = (
        "You are operating inside the Tsugite micro-agent runtime. Follow the rendered task faithfully, use the available "
        "tools when they meaningfully advance the work, and maintain a living plan via the task_* tools. Create or update "
        "tasks whenever you define new sub-work, mark progress as you go, and rely on the task summary to decide the next "
        "action. Provide a clear, actionable final response without unnecessary filler.\n\n"
    )

    if text_mode:
        completion = (
            "Task Completion: For conversational responses, use the format 'Thought: [your response]'. "
            "When using tools or code, write Python code blocks and call final_answer(result) when complete.\n\n"
        )
    else:
        completion = (
            "Task Completion: Write Python code to accomplish your task. "
            "When you have completed your task, call final_answer(result) to signal completion and return the result.\n\n"
        )

    interactive = (
        "Interactive Mode: The `is_interactive` variable indicates whether you're running in an interactive terminal. "
        "Interactive-only tools (like ask_user) are automatically available only when is_interactive is True."
    )

    return base + completion + interactive


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
) -> str | tuple[str, Optional[int], Optional[float], int, list]:
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
        Agent execution result as string, or tuple of (result, token_count, cost, steps) if return_token_usage=True

    Raises:
        RuntimeError: If execution fails
    """
    # Initialize task manager for this execution (unless skipped for multi-step)
    if not skip_task_reset:
        reset_task_manager()

    # Build base instructions
    base_instructions = _combine_instructions(
        get_default_instructions(text_mode=agent_config.text_mode),
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
                step_count = len(result.steps) if result.steps else 0
                steps_list = result.steps if result.steps else []

                # If result has error, raise it AFTER we've already extracted the steps
                # The exception will be caught by the benchmark, but steps are already available
                if result.error:
                    # Create custom exception that includes execution details
                    error = RuntimeError(f"Agent execution failed: {result.error}")
                    # Attach execution details to exception for debugging
                    error.execution_steps = steps_list
                    error.token_usage = result.token_usage
                    error.cost = result.cost
                    error.step_count = step_count
                    raise error

                return str(result.output), result.token_usage, result.cost, step_count, steps_list
            else:
                return str(result), None, None, 0, []
        else:
            from tsugite.core.agent import AgentResult

            if isinstance(result, AgentResult):
                return str(result.output)
            else:
                return str(result)

    except Exception as e:
        # Preserve execution details if they're attached to the original exception
        # (This happens when agent hits max_steps and we want execution trace for debugging)
        if hasattr(e, "execution_steps"):
            new_error = RuntimeError(f"Agent execution failed: {e}")
            new_error.execution_steps = e.execution_steps
            new_error.token_usage = getattr(e, "token_usage", None)
            new_error.cost = getattr(e, "cost", None)
            new_error.step_count = getattr(e, "step_count", 0)
            raise new_error
        else:
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
) -> str | tuple[str, Optional[int], Optional[float], int, list]:
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
        Agent execution result as string, or tuple of (result, token_count, cost, step_count, execution_steps) if return_token_usage=True

    Raises:
        ValueError: If agent file is invalid
        RuntimeError: If agent execution fails
    """
    if context is None:
        context = {}

    # Initialize task manager for this agent session
    reset_task_manager()
    task_manager = get_task_manager()

    # Parse agent configuration (with inheritance resolution)
    try:
        agent = parse_agent_file(agent_path)
        agent_config = agent.config
    except Exception as e:
        raise ValueError(f"Failed to parse agent file: {e}")

    # Set current agent in thread-local storage for spawn_agent tracking
    _set_current_agent(agent_config.name)

    try:
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
            "text_mode": agent_config.text_mode,
            "tools": agent_config.tools,  # Make tools list available to templates
            # Subagent context (set by spawn_agent if this is a spawned agent)
            "is_subagent": context.get("is_subagent", False),
            "parent_agent": context.get("parent_agent", None),
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
    finally:
        # Always clear the current agent context when done
        _clear_current_agent()


def validate_agent_file(agent_path: Path) -> tuple[bool, str]:
    """Validate that an agent file can be executed.

    Args:
        agent_path: Path to agent markdown file (or builtin agent path like <builtin-default>)

    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        # Parse agent with inheritance resolution
        agent = parse_agent_file(agent_path)

        # Use centralized validation
        return validate_agent_execution(agent)

    except Exception as e:
        return False, f"Agent file validation failed: {e}"


def get_agent_info(agent_path: Path) -> Dict[str, Any]:
    """Get information about an agent without executing it.

    Args:
        agent_path: Path to agent markdown file (or builtin agent path like <builtin-default>)

    Returns:
        Dictionary with agent information
    """
    try:
        # Parse agent with inheritance resolution
        agent = parse_agent_file(agent_path)
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
    stream: bool = False,
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
        stream: Whether to stream responses in real-time (currently unused for multi-step agents)

    Returns:
        Result from the final step

    Raises:
        ValueError: If agent file is invalid or step parsing fails
        RuntimeError: If any step execution fails
    """
    from tsugite.md_agents import extract_step_directives, has_step_directives

    if context is None:
        context = {}

    # Parse agent (with inheritance resolution)
    try:
        agent = parse_agent_file(agent_path)
    except Exception as e:
        raise ValueError(f"Failed to parse agent file: {e}")

    # Set current agent in thread-local storage for spawn_agent tracking
    _set_current_agent(agent.config.name)

    try:
        # Extract steps from raw markdown (before any rendering)
        if not has_step_directives(agent.content):
            raise ValueError(f"Agent {agent_path} does not contain step directives. Use run_agent() instead.")

        preamble, steps = extract_step_directives(agent.content)
    except Exception as e:
        raise ValueError(f"Failed to parse step directives: {e}")

    try:
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
            "text_mode": agent.config.text_mode,
            "tools": agent.config.tools,  # Make tools list available to templates
            # Subagent context (set by spawn_agent if this is a spawned agent)
            "is_subagent": context.get("is_subagent", False),
            "parent_agent": context.get("parent_agent", None),
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
        step_metrics: List[StepMetrics] = []

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
            step_start_time = time.time()

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
                        custom_logger.console.print(
                            f"[yellow]{step_header} Retry {attempt}/{step.max_retries}...[/yellow]"
                        )
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
                        _stderr_console.rule(
                            f"[bold cyan]DEBUG: Executing Step {i}/{len(steps)}: {step.name}[/bold cyan]"
                        )

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

                        # Build detailed error message for rendering failure
                        available_vars = list(step_context.keys())
                        previous_step = steps[i - 2].name if i > 1 else "None"

                        error_lines = [
                            "",
                            "Step Template Rendering Failed",
                            "━" * 60,
                            f"Step: {step.name} ({i}/{len(steps)})",
                            f"Previous Step: {previous_step}",
                            f"Attempts: {max_attempts}",
                            "",
                            f"Context Variables: {', '.join(available_vars)}",
                            "",
                            "Errors:",
                        ]

                        for idx, err in enumerate(errors, 1):
                            error_lines.append(f"  Attempt {idx}: {err}")

                        error_lines.extend(
                            [
                                "━" * 60,
                                "",
                                "To debug:",
                                "  1. Check for undefined variables in step template",
                                "  2. Verify previous steps assigned expected variables",
                                "  3. Run with --debug to see full context",
                                "",
                            ]
                        )

                        raise RuntimeError("\n".join(error_lines))

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
                    # Wrap execution with timeout if specified
                    async def execute_step():
                        coro = _execute_agent_with_prompt(
                            rendered_prompt=rendered_step_prompt,
                            agent_config=agent.config,
                            model_override=model_override,
                            custom_logger=custom_logger,
                            trust_mcp_code=trust_mcp_code,
                            delegation_agents=delegation_agents,
                            skip_task_reset=True,  # Don't reset tasks between steps
                            model_kwargs=step.model_kwargs,
                            injectable_vars=injectable_vars,
                            stream=stream,
                        )

                        if step.timeout:
                            return await asyncio.wait_for(coro, timeout=step.timeout)
                        else:
                            return await coro

                    step_result = asyncio.run(execute_step())

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

                    # Record metrics for successful step
                    step_duration = time.time() - step_start_time
                    step_metrics.append(
                        StepMetrics(
                            step_name=step.name,
                            step_number=i,
                            duration=step_duration,
                            status="success",
                        )
                    )

                    # Success - break retry loop
                    break

                except asyncio.TimeoutError:
                    error_msg = f"Step timed out after {step.timeout} seconds"
                    errors.append(error_msg)

                except Exception as e:
                    error_msg = str(e)
                    errors.append(error_msg)

                    # Check if we should continue despite the error
                    if step.continue_on_error:
                        # Log warning but continue execution
                        if custom_logger and hasattr(custom_logger, "ui_handler"):
                            custom_logger.ui_handler.clear_multistep_context()

                        warning_msg = f"⚠ Step '{step.name}' failed but continuing (continue_on_error=true)"
                        if custom_logger:
                            custom_logger.console.print(f"[yellow]{warning_msg}[/yellow]")
                            custom_logger.console.print(f"[dim]Error: {error_msg}[/dim]")
                        else:
                            _stderr_console.print(f"[yellow]{warning_msg}[/yellow]")
                            _stderr_console.print(f"[dim]Error: {error_msg}[/dim]")

                        # Assign None to the variable if specified
                        if step.assign_var:
                            step_context[step.assign_var] = None
                            if debug:
                                _stderr_console.print(f"[dim]Assigned None to variable: {step.assign_var}[/dim]")

                        # Record metrics for skipped step
                        step_duration = time.time() - step_start_time
                        step_metrics.append(
                            StepMetrics(
                                step_name=step.name,
                                step_number=i,
                                duration=step_duration,
                                status="skipped",
                                error=error_msg,
                            )
                        )

                        # Break retry loop and move to next step
                        break

                    if attempt == max_attempts - 1:
                        if custom_logger and hasattr(custom_logger, "ui_handler"):
                            custom_logger.ui_handler.clear_multistep_context()

                        # Build detailed error message with context
                        available_vars = list(injectable_vars.keys())
                        previous_step = steps[i - 2].name if i > 1 else "None"

                        error_lines = [
                            "",
                            "Step Execution Failed",
                            "━" * 60,
                            f"Step: {step.name} ({i}/{len(steps)})",
                            f"Previous Step: {previous_step}",
                            f"Attempts: {max_attempts}",
                            "",
                            f"Available Variables: {', '.join(available_vars) if available_vars else 'None'}",
                            "",
                            "Errors:",
                        ]

                        for idx, err in enumerate(errors, 1):
                            error_lines.append(f"  Attempt {idx}: {err}")

                        error_lines.extend(
                            [
                                "━" * 60,
                                "",
                                "To debug:",
                                "  1. Run with --debug to see rendered prompts",
                                "  2. Check variable values in previous steps",
                                "  3. Verify step dependencies are correct",
                                "",
                            ]
                        )

                        raise RuntimeError("\n".join(error_lines))

                    if step.retry_delay > 0:
                        time.sleep(step.retry_delay)

                    if custom_logger and not debug:
                        custom_logger.console.print(f"[yellow]Step '{step.name}' failed: {error_msg}[/yellow]")
                    elif not debug:
                        _stderr_console.print(f"[yellow]Step '{step.name}' failed: {error_msg}[/yellow]")

            # Display metrics summary
            if step_metrics:
                _display_step_metrics(step_metrics, custom_logger if custom_logger else None)

            return final_result or ""
    finally:
        # Always clear the current agent context when done
        _clear_current_agent()


def _display_step_metrics(metrics: List[StepMetrics], custom_logger: Optional[Any] = None):
    """Display step execution metrics in a table."""
    from rich.table import Table

    console = custom_logger.console if custom_logger and hasattr(custom_logger, "console") else _stderr_console

    table = Table(title="Multi-Step Execution Metrics", show_header=True)
    table.add_column("Step", style="cyan")
    table.add_column("Duration", justify="right", style="yellow")
    table.add_column("Status", justify="center")

    total_duration = 0
    successful = 0
    failed = 0
    skipped = 0

    for m in metrics:
        status_color = {
            "success": "green",
            "failed": "red",
            "skipped": "yellow",
        }.get(m.status, "white")

        status_symbol = {
            "success": "✓",
            "failed": "✗",
            "skipped": "⚠",
        }.get(m.status, "?")

        table.add_row(
            f"{m.step_number}. {m.step_name}",
            f"{m.duration:.1f}s",
            f"[{status_color}]{status_symbol} {m.status}[/{status_color}]",
        )

        total_duration += m.duration
        if m.status == "success":
            successful += 1
        elif m.status == "failed":
            failed += 1
        elif m.status == "skipped":
            skipped += 1

    console.print()
    console.print(table)

    # Summary line
    summary_parts = []
    summary_parts.append(f"Total: {total_duration:.1f}s")
    if successful > 0:
        summary_parts.append(f"[green]Success: {successful}[/green]")
    if skipped > 0:
        summary_parts.append(f"[yellow]Skipped: {skipped}[/yellow]")
    if failed > 0:
        summary_parts.append(f"[red]Failed: {failed}[/red]")

    console.print(" | ".join(summary_parts))
    console.print()


def preview_multistep_agent(
    agent_path: Path,
    prompt: str,
    context: Optional[Dict[str, Any]] = None,
    console: Optional[Any] = None,
):
    """Preview multi-step agent execution without running it.

    Shows the execution plan including steps, dependencies, attributes,
    and estimated resource usage.

    Args:
        agent_path: Path to agent markdown file
        prompt: User prompt/task for the agent
        context: Additional context variables
        console: Rich Console instance (defaults to stderr console)
    """
    import re

    from rich.table import Table

    # Use provided console or default to stderr
    if console is None:
        console = _stderr_console

    # Parse agent (with inheritance resolution)
    try:
        agent = parse_agent_file(agent_path)
    except Exception as e:
        console.print(f"[red]Error parsing agent: {e}[/red]")
        return

    # Extract steps
    from tsugite.md_agents import extract_step_directives, has_step_directives

    if not has_step_directives(agent.content):
        console.print("[yellow]This is a single-step agent (no step directives).[/yellow]")
        console.print("[dim]Dry-run preview is for multi-step agents only.[/dim]")
        return

    try:
        preamble, steps = extract_step_directives(agent.content)
    except Exception as e:
        console.print(f"[red]Error extracting steps: {e}[/red]")
        return

    # Display header
    console.print()
    console.print("[bold]Dry-Run Preview: Multi-Step Agent[/bold]")
    console.print("═" * 60)
    console.print(f"Agent: {agent.config.name}")
    console.print(f"File: {agent_path.name}")
    console.print(f"Prompt: {prompt}")
    console.print(f"Steps: {len(steps)}")
    console.print(f"Model: {agent.config.model or 'default'}")
    console.print(f"Tools: {', '.join(agent.config.tools) if agent.config.tools else 'None'}")
    console.print()

    # Show steps in table format
    table = Table(title="Execution Plan", show_header=True)
    table.add_column("#", style="cyan", width=3)
    table.add_column("Step Name", style="green")
    table.add_column("Attributes", style="yellow")
    table.add_column("Dependencies", style="dim")

    for i, step in enumerate(steps, 1):
        # Collect attributes
        attrs = []
        if step.assign_var:
            attrs.append(f"→ {step.assign_var}")
        if step.max_retries > 0:
            attrs.append(f"retries:{step.max_retries}")
        if step.timeout:
            attrs.append(f"timeout:{step.timeout}s")
        if step.continue_on_error:
            attrs.append("continue_on_error")
        if step.retry_delay > 0:
            attrs.append(f"delay:{step.retry_delay}s")

        attr_str = ", ".join(attrs) if attrs else "—"

        # Find dependencies (variables referenced in step content)
        variables_used = set(re.findall(r"\{\{\s*(\w+)", step.content))
        # Filter out template helpers and metadata
        metadata_vars = {
            "user_prompt",
            "task_summary",
            "step_number",
            "step_name",
            "total_steps",
            "now",
            "today",
            "is_interactive",
            "is_retry",
            "retry_count",
        }
        real_deps = variables_used - metadata_vars

        deps_str = ", ".join(sorted(real_deps)) if real_deps else "—"

        table.add_row(str(i), step.name, attr_str, deps_str)

    console.print(table)
    console.print()

    # Warnings
    warnings = []
    for step in steps:
        if step.timeout and step.timeout < 30:
            warnings.append(f"⚠ Step '{step.name}' has short timeout ({step.timeout}s)")
        if step.continue_on_error and not step.assign_var:
            warnings.append(f"⚠ Step '{step.name}' has continue_on_error but no assign variable")

    if warnings:
        console.print("[bold]Warnings:[/bold]")
        console.print("─" * 60)
        for warning in warnings:
            console.print(f"  [yellow]{warning}[/yellow]")
        console.print()

    console.print("━" * 60)
    console.print("[dim]Note: This is a preview only. No tools will be executed.[/dim]")
    console.print("[dim]Remove --dry-run to execute the agent.[/dim]")
    console.print()
