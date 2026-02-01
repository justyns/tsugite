"""Agent execution engine using TsugiteAgent."""

import asyncio
import time
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from tsugite.core.agent import TsugiteAgent
from tsugite.core.executor import LocalExecutor
from tsugite.exceptions import AgentExecutionError
from tsugite.md_agents import AgentConfig, parse_agent_file
from tsugite.options import ExecutionOptions
from tsugite.renderer import AgentRenderer
from tsugite.utils import is_interactive

from .helpers import (
    _stderr_console,
    clear_allowed_agents,
    clear_current_agent,
    clear_multistep_ui_context,
    get_display_console,
    get_ui_handler,
    print_step_progress,
    set_current_agent,
    set_multistep_ui_context,
)
from .metrics import StepMetrics, display_step_metrics
from .models import AgentExecutionResult

# Display constants for truncating long output
MAX_VARIABLE_PREVIEW_LENGTH = 100  # Max characters to show in variable documentation
MAX_CONTENT_PREVIEW_LENGTH = 200  # Max characters to show in debug attachment previews


class ExecutionContext:
    """Namespace for tsugite-provided execution context.

    Provides access to runtime metadata via attribute access (ctx.user_prompt, ctx.tasks, etc.)
    while keeping user-assigned step variables as top-level names in the execution namespace.
    """

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __repr__(self):
        attrs = ", ".join(f"{k}={v!r}" for k, v in self.__dict__.items())
        return f"ExecutionContext({attrs})"


if TYPE_CHECKING:
    from tsugite.agent_preparation import PreparedAgent
    from tsugite.events import EventBus


def _format_debug_output(prepared: "PreparedAgent") -> str:
    """Format debug output showing system prompt, attachments, and user prompt.

    Args:
        prepared: Prepared agent with rendered prompts and attachments

    Returns:
        Formatted debug string for printing to stderr
    """
    from tsugite.attachments.base import AttachmentContentType

    parts = ["\nDEBUG: Complete Prompt Context", "=" * 80, ""]

    parts.append("SYSTEM PROMPT:")
    parts.append("-" * 80)
    parts.append(prepared.system_message)
    parts.append("")

    if prepared.attachments:
        parts.append(f"ATTACHMENTS ({len(prepared.attachments)}):")
        parts.append("-" * 80)
        for attachment in prepared.attachments:
            if attachment.content_type == AttachmentContentType.TEXT:
                preview = (
                    attachment.content[:MAX_CONTENT_PREVIEW_LENGTH] + "..."
                    if len(attachment.content) > MAX_CONTENT_PREVIEW_LENGTH
                    else attachment.content
                )
                parts.append(f"• {attachment.name}")
                parts.append(f"  {preview}")
            elif attachment.source_url:
                parts.append(f"• {attachment.name}")
                parts.append(f"  [{attachment.content_type.value}: {attachment.source_url}]")
            else:
                parts.append(f"• {attachment.name}")
                parts.append(f"  [{attachment.content_type.value} file: {attachment.mime_type}]")
            parts.append("")
    else:
        parts.append("NO ATTACHMENTS")
        parts.append("")

    parts.append("USER PROMPT:")
    parts.append("-" * 80)
    parts.append(prepared.rendered_prompt)
    parts.append("")
    parts.append("=" * 80)

    return "\n".join(parts)


def _get_model_string(model_override: Optional[str], agent_config: AgentConfig) -> str:
    """Get model string with fallback to config default.

    Args:
        model_override: Model override from CLI
        agent_config: Agent configuration

    Returns:
        Model string

    Raises:
        RuntimeError: If no model is specified anywhere
    """
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

    return model_string


def _build_step_error_message(
    error_type: str,
    step_name: str,
    step_number: int,
    total_steps: int,
    errors: List[str],
    available_vars: List[str],
    previous_step: str,
    max_attempts: int,
    debug_tips: List[str],
) -> str:
    """Build detailed error message for step failures.

    Args:
        error_type: Type of error (e.g., "Template Rendering Failed", "Step Execution Failed")
        step_name: Name of the failed step
        step_number: Current step number (1-indexed)
        total_steps: Total number of steps
        errors: List of error messages from all attempts
        available_vars: List of available variable names
        previous_step: Name of the previous step
        max_attempts: Maximum number of retry attempts
        debug_tips: List of debugging suggestions

    Returns:
        Formatted error message string
    """
    error_lines = [
        "",
        f"Step {error_type}",
        "━" * 60,
        f"Step: {step_name} ({step_number}/{total_steps})",
        f"Previous Step: {previous_step}",
        f"Attempts: {max_attempts}",
        "",
    ]

    # Add variables section (format depends on whether we have any)
    if available_vars:
        var_label = "Context Variables" if "Template" in error_type else "Available Variables"
        error_lines.append(f"{var_label}: {', '.join(available_vars)}")
    else:
        error_lines.append("Available Variables: None")

    error_lines.extend(["", "Errors:"])

    # Add all error attempts
    for idx, err in enumerate(errors, 1):
        error_lines.append(f"  Attempt {idx}: {err}")

    # Add debugging tips
    error_lines.extend(["━" * 60, "", "To debug:"])
    for tip in debug_tips:
        error_lines.append(f"  {tip}")
    error_lines.append("")

    return "\n".join(error_lines)


def _combine_instructions(*segments: str) -> str:
    """Join instruction segments, skipping empties.

    Args:
        *segments: Variable number of instruction strings

    Returns:
        Combined instructions with segments separated by double newlines
    """
    parts = [segment.strip() for segment in segments if segment and segment.strip()]
    return "\n\n".join(parts)


def _setup_event_context(event_bus: Optional["EventBus"]) -> None:
    """Set event bus in UI context for tool access.

    Args:
        event_bus: Event bus to set in context, or None to skip
    """
    if event_bus:
        from tsugite.ui_context import set_ui_context

        set_ui_context(event_bus=event_bus)


def get_default_instructions() -> str:
    """Get minimal default instructions. Detailed guidance comes from skills.

    Returns:
        Default instructions for code execution mode
    """
    base = "You accomplish tasks by writing Python code.\n\n"

    output = (
        "## Output\n\n"
        "- `print(x)` - See in next turn (internal)\n"
        "- `send_message(msg)` - Show user progress\n"
        "- `final_answer(msg)` - Final response (stops execution)\n\n"
    )

    rules = (
        "## Rules\n\n"
        "1. Always respond with Python code blocks\n"
        "2. Call `final_answer()` when done\n"
        "3. Variables persist between turns\n"
    )

    return base + output + rules


def execute_prefetch(prefetch_config: List[Dict[str, Any]], event_bus: Optional["EventBus"] = None) -> Dict[str, Any]:
    from tsugite.tools import call_tool

    # Set event_bus in context so tools can access it
    _setup_event_context(event_bus)

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
            if event_bus:
                from tsugite.events import WarningEvent

                event_bus.emit(WarningEvent(message=f"Prefetch tool '{tool_name}' failed: {e}"))
            context[assign_name] = None

    return context


def execute_tool_directives(
    content: str, existing_context: Optional[Dict[str, Any]] = None, event_bus: Optional["EventBus"] = None
) -> tuple[str, Dict[str, Any]]:
    """Execute tool directives in content and return updated context.

    Tool directives are inline <!-- tsu:tool --> comments that execute tools
    during the rendering phase, similar to prefetch but embedded in content.

    Args:
        content: Markdown content with tool directives
        existing_context: Current template context (for error messages, not used for execution)
        event_bus: Optional event bus for emitting warnings

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
    from tsugite.tools import call_tool

    if existing_context is None:
        existing_context = {}

    # Set event_bus in context so tools can access it
    _setup_event_context(event_bus)

    # Extract tool directives
    try:
        directives = extract_tool_directives(content)
    except ValueError as e:
        # If parsing fails, return content unchanged with empty context
        if event_bus:
            from tsugite.events import WarningEvent

            event_bus.emit(WarningEvent(message=f"Failed to parse tool directives: {e}"))
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
            if event_bus:
                from tsugite.events import WarningEvent

                event_bus.emit(WarningEvent(message=f"Tool directive '{directive.name}' failed: {e}"))
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
            ui_handler = get_ui_handler(custom_logger)
            if ui_handler:
                from tsugite.events import EventBus, ReasoningContentEvent

                event_bus = EventBus()
                event_bus.subscribe(ui_handler.handle_event)
                event_bus.emit(ReasoningContentEvent(content=reasoning_content, step=None))


async def _execute_agent_with_prompt(
    prepared: "PreparedAgent",
    exec_options: Optional[ExecutionOptions] = None,
    workspace: Optional[Any] = None,
    custom_logger: Optional[Any] = None,
    skip_task_reset: bool = False,
    model_kwargs: Optional[Dict[str, Any]] = None,
    injectable_vars: Optional[Dict[str, Any]] = None,
    previous_messages: Optional[List[Dict]] = None,
    path_context: Optional[Any] = None,
) -> str | AgentExecutionResult:
    """Execute agent with a prepared agent.

    Low-level execution function used by both run_agent and run_multistep_agent.
    """
    if exec_options is None:
        exec_options = ExecutionOptions()

    agent_config = prepared.agent_config

    # Add variable documentation to instructions if variables are available
    combined_instructions = prepared.combined_instructions
    if injectable_vars:
        var_docs = "\n\nAVAILABLE PYTHON VARIABLES:\n"
        for var_name, var_value in injectable_vars.items():
            preview = str(var_value)[:MAX_VARIABLE_PREVIEW_LENGTH]
            if len(str(var_value)) > MAX_VARIABLE_PREVIEW_LENGTH:
                preview += "..."
            var_docs += f"- {var_name}: {preview}\n"
        combined_instructions = prepared.combined_instructions + var_docs

    # Extract ui_handler and create EventBus early so warnings can use it
    ui_handler = get_ui_handler(custom_logger)

    # Create EventBus and subscribe ui_handler
    from tsugite.events import EventBus, InfoEvent, WarningEvent

    event_bus = EventBus()
    if ui_handler:
        event_bus.subscribe(ui_handler.handle_event)

    # Start with tools from prepared agent
    tools = list(prepared.tools)  # Make a copy

    # Filter out interactive tools in subagent mode
    import os

    if os.environ.get("TSUGITE_SUBAGENT_MODE") == "1":
        tools = [t for t in tools if t.name not in ["ask_user", "ask_user_batch"]]

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
                from tsugite.core.tools import create_tool_from_tsugite

                tools.append(create_tool_from_tsugite(tool_def.name))
        except Exception as e:
            event_bus.emit(WarningEvent(message=f"Failed to register custom tools: {e}"))

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
                    event_bus.emit(WarningEvent(message=f"MCP server '{server_name}' not found in config. Skipping."))
                    continue

                server_config = global_mcp_config[server_name]
                try:
                    mcp_client, mcp_tools = await load_mcp_tools(server_config, allowed_tools)
                    mcp_clients.append(mcp_client)  # Keep client alive for tools to work
                    tools.extend(mcp_tools)

                    event_bus.emit(InfoEvent(message=f"Loaded {len(mcp_tools)} tools from MCP server '{server_name}'"))
                except Exception as e:
                    event_bus.emit(WarningEvent(message=f"Failed to load MCP tools from '{server_name}': {e}"))
        except Exception as e:
            event_bus.emit(WarningEvent(message=f"Failed to load MCP tools: {e}"))
            event_bus.emit(WarningEvent(message="Continuing without MCP tools."))

    # Get model string
    model_string = _get_model_string(exec_options.model_override, agent_config)

    # Merge reasoning_effort from agent config into model_kwargs
    final_model_kwargs = dict(model_kwargs or {})
    if hasattr(agent_config, "reasoning_effort") and agent_config.reasoning_effort:
        # Only add if not already specified in model_kwargs
        if "reasoning_effort" not in final_model_kwargs:
            final_model_kwargs["reasoning_effort"] = agent_config.reasoning_effort

    # Create executor with workspace directory and event bus
    workspace_dir = workspace.path if workspace else None
    executor = LocalExecutor(workspace_dir=workspace_dir, event_bus=event_bus, path_context=path_context)

    # Inject variables into executor (for multi-step agents)
    if injectable_vars:
        await executor.send_variables(injectable_vars)

    # Create and run agent
    try:
        agent = TsugiteAgent(
            model_string=model_string,
            tools=tools,
            instructions=combined_instructions or "",
            max_turns=agent_config.max_turns,
            executor=executor,
            model_kwargs=final_model_kwargs,
            event_bus=event_bus,
            model_name=model_string,
            attachments=prepared.attachments,
            skills=prepared.skills,
            previous_messages=previous_messages,
        )

        # Set event_bus in context so tools can access it during execution
        _setup_event_context(event_bus)

        # Run agent
        result = await agent.run(
            prepared.rendered_prompt,
            return_full_result=exec_options.return_token_usage,
            stream=exec_options.stream,
        )

        # Extract and display reasoning content if present
        _extract_reasoning_content(agent, custom_logger)

        # Return appropriate format
        if exec_options.return_token_usage:
            from tsugite.core.agent import AgentResult

            if isinstance(result, AgentResult):
                step_count = len(result.steps) if result.steps else 0
                steps_list = result.steps if result.steps else []

                # If result has error, raise it AFTER we've already extracted the steps
                # The exception will be caught by the benchmark, but steps are already available
                if result.error:
                    # Create custom exception that includes execution details
                    raise AgentExecutionError(
                        f"Agent execution failed: {result.error}",
                        execution_steps=steps_list,
                        token_usage=result.token_usage,
                        cost=result.cost,
                        step_count=step_count,
                    )

                return AgentExecutionResult(
                    response=str(result.output),
                    token_count=result.token_usage,
                    cost=result.cost,
                    step_count=step_count,
                    execution_steps=steps_list,
                    system_message=prepared.system_message,
                    attachments=prepared.attachments,
                )
            else:
                return AgentExecutionResult(
                    response=str(result),
                    token_count=None,
                    cost=None,
                    step_count=0,
                    execution_steps=[],
                    system_message=None,
                    attachments=[],
                )
        else:
            from tsugite.core.agent import AgentResult

            if isinstance(result, AgentResult):
                return str(result.output)
            else:
                return str(result)

    except Exception as e:
        # Preserve execution details if they're attached to the original exception
        # (This happens when agent hits max_turns and we want execution trace for debugging)
        if isinstance(e, AgentExecutionError):
            # Already has execution details, just re-raise
            raise
        elif hasattr(e, "execution_steps"):
            # Some other exception with attached details, convert to AgentExecutionError
            raise AgentExecutionError(
                f"Agent execution failed: {e}",
                execution_steps=e.execution_steps,
                token_usage=getattr(e, "token_usage", None),
                cost=getattr(e, "cost", None),
                step_count=getattr(e, "step_count", 0),
            )
        else:
            raise RuntimeError(f"Agent execution failed: {e}")
    finally:
        # Clean up MCP client connections
        for client in mcp_clients:
            try:
                await client.disconnect()
            except Exception:
                pass  # Best effort cleanup

        # Clean up pending tasks (but not LiteLLM clients - that's handled by run_async_with_cleanup wrapper)
        # ONLY run cleanup for top-level agents, not spawned agents
        import asyncio
        import threading

        if threading.current_thread() == threading.main_thread():
            # Get all tasks except the current one
            current_task = asyncio.current_task()
            all_tasks = asyncio.all_tasks()
            pending_tasks = [task for task in all_tasks if task is not current_task and not task.done()]

            # Cancel all pending tasks
            for task in pending_tasks:
                task.cancel()

            # Wait for all tasks to be cancelled
            if pending_tasks:
                await asyncio.gather(*pending_tasks, return_exceptions=True)


def run_agent(
    agent_path: Path,
    prompt: str,
    exec_options: Optional[ExecutionOptions] = None,
    context: Optional[Dict[str, Any]] = None,
    custom_logger: Optional[Any] = None,
    continue_conversation_id: Optional[str] = None,
    attachments: Optional[List[Any]] = None,
    path_context: Optional[Any] = None,
) -> str | AgentExecutionResult:
    """Run a Tsugite agent (sync wrapper around run_agent_async).

    Args:
        agent_path: Path to agent markdown file
        prompt: User prompt/task for the agent
        exec_options: Execution options (model, debug, stream, etc.)
        context: Additional context variables
        custom_logger: Custom logger for agent output
        continue_conversation_id: Optional conversation ID to continue
        attachments: Optional list of Attachment objects
        path_context: Optional PathContext with invoked_from, workspace_dir, effective_cwd

    Returns:
        Agent execution result as string or AgentExecutionResult with metrics
    """
    import json
    import os
    import sys

    if exec_options is None:
        exec_options = ExecutionOptions()

    # Handle subagent mode (subprocess-based execution)
    subagent_mode = os.environ.get("TSUGITE_SUBAGENT_MODE") == "1"
    if subagent_mode:
        try:
            stdin_data = json.loads(sys.stdin.read())
            prompt = stdin_data["prompt"]
            context = stdin_data.get("context", {})
        except Exception as e:
            error_event = {"type": "error", "error": f"Failed to parse stdin JSON: {e}"}
            print(json.dumps(error_event), flush=True)
            sys.exit(1)

        from tsugite.ui.jsonl import JSONLUIHandler

        custom_logger = SimpleNamespace(ui_handler=JSONLUIHandler())

    return asyncio.run(
        run_agent_async(
            agent_path=agent_path,
            prompt=prompt,
            exec_options=exec_options,
            context=context,
            custom_logger=custom_logger,
            continue_conversation_id=continue_conversation_id,
            attachments=attachments,
            path_context=path_context,
        )
    )


async def run_agent_async(
    agent_path: Path,
    prompt: str,
    exec_options: Optional[ExecutionOptions] = None,
    context: Optional[Dict[str, Any]] = None,
    workspace: Optional[Any] = None,
    custom_logger: Optional[Any] = None,
    continue_conversation_id: Optional[str] = None,
    attachments: Optional[List[Any]] = None,
    channel_metadata: Optional[Dict[str, Any]] = None,
    path_context: Optional[Any] = None,
) -> str | AgentExecutionResult:
    """Run a Tsugite agent (async version for tests and async contexts).

    Args:
        agent_path: Path to agent markdown file
        prompt: User prompt/task for the agent
        exec_options: Execution options (model, debug, stream, etc.)
        context: Additional context variables
        workspace: Optional Workspace for persistent context and working directory
        custom_logger: Custom logger for agent output
        continue_conversation_id: Optional conversation ID to continue
        attachments: Optional list of Attachment objects
        channel_metadata: Optional channel routing metadata (source, channel_id, user_id, reply_to)
        path_context: Optional PathContext with invoked_from, workspace_dir, effective_cwd

    Returns:
        Agent execution result as string or AgentExecutionResult with metrics
    """
    if exec_options is None:
        exec_options = ExecutionOptions()

    if context is None:
        context = {}

    # Load conversation history if continuing
    previous_messages = []
    if continue_conversation_id:
        from tsugite.agent_runner.history_integration import load_and_apply_history

        try:
            previous_messages = load_and_apply_history(continue_conversation_id)
        except ValueError:
            # New conversation (e.g., fresh workspace session) - start with empty history
            pass

    # Parse agent configuration (with inheritance resolution)
    try:
        agent = parse_agent_file(agent_path)
        agent_config = agent.config
    except Exception as e:
        raise ValueError(f"Failed to parse agent file: {e}")

    # Set current agent in thread-local storage for spawn_agent tracking
    set_current_agent(agent_config.name)

    try:
        # Prepare agent using unified preparation pipeline
        from tsugite.agent_preparation import AgentPreparer

        preparer = AgentPreparer()
        prepared = preparer.prepare(
            agent=agent,
            prompt=prompt,
            context=context,
            workspace=workspace,
            attachments=attachments,
            path_context=path_context,
        )

        # Debug output if requested
        if exec_options.debug:
            import sys

            print(_format_debug_output(prepared), file=sys.stderr)

        # Execute with the low-level helper (async - no asyncio.run wrapper)
        return await _execute_agent_with_prompt(
            prepared=prepared,
            exec_options=exec_options,
            workspace=workspace,
            custom_logger=custom_logger,
            previous_messages=previous_messages,
            path_context=path_context,
        )
    finally:
        # Always clear the current agent context when done
        clear_current_agent()
        clear_allowed_agents()


# Predefined loop condition helpers
# These are plain Jinja2 expressions (no {{ }}) that can be used in {% if %} blocks
LOOP_HELPERS = {}


def _build_injectable_vars(step_context: Dict[str, Any], assigned_vars: Optional[set] = None) -> Dict[str, Any]:
    """Build variables for injection into Python execution namespace.

    Creates a `ctx` object containing everything. Only user-assigned step variables
    (from `assign="varname"`) are exposed at top-level to avoid namespace pollution.

    Args:
        step_context: Full step context dictionary
        assigned_vars: Set of variable names assigned by user via step.assign_var.
                       If None, no variables are exposed at top-level.

    Returns:
        Dictionary with 'ctx' ExecutionContext and user-assigned variables at top-level
    """
    ctx = ExecutionContext(**step_context)

    # Only user-assigned step variables at top-level (not tsugite metadata)
    if assigned_vars:
        top_level = {k: v for k, v in step_context.items() if k in assigned_vars}
    else:
        top_level = {}

    return {"ctx": ctx, **top_level}


def _build_prepared_agent_for_step(
    agent: Any,
    rendered_step_prompt: str,
    step_context: Dict[str, Any],
    attachments: Optional[List[Any]] = None,
) -> "PreparedAgent":
    """Build a PreparedAgent for step execution.

    This creates the PreparedAgent manually since multistep agents handle
    their own rendering with accumulated context.

    Args:
        agent: Parsed agent with config
        rendered_step_prompt: Already-rendered step prompt
        step_context: Step execution context
        attachments: List of Attachment objects for prompt caching

    Returns:
        PreparedAgent ready for execution
    """
    from tsugite.agent_preparation import PreparedAgent
    from tsugite.core.agent import build_system_prompt
    from tsugite.core.tools import create_tool_from_tsugite
    from tsugite.tools import expand_tool_specs

    # Build instructions
    base_instructions = get_default_instructions()
    agent_instructions = getattr(agent.config, "instructions", "")
    combined_instructions = _combine_instructions(base_instructions, agent_instructions)

    # Expand and create tools
    expanded_tools = expand_tool_specs(agent.config.tools) if agent.config.tools else []
    # Always include spawn_agent if not already present
    if "spawn_agent" not in expanded_tools:
        expanded_tools.append("spawn_agent")
    tools = [create_tool_from_tsugite(name) for name in expanded_tools]

    # Build system message
    system_message = build_system_prompt(tools, combined_instructions)

    # Create PreparedAgent
    return PreparedAgent(
        agent=agent,
        agent_config=agent.config,
        system_message=system_message,
        user_message=rendered_step_prompt,
        rendered_prompt=rendered_step_prompt,
        tools=tools,
        context=step_context,
        combined_instructions=combined_instructions,
        prefetch_results={},  # Already executed in preamble
        attachments=attachments or [],
    )


def evaluate_loop_condition(expression: str, context: Dict[str, Any]) -> bool:
    """Evaluate a Jinja2 expression or helper as a boolean condition.

    Args:
        expression: Jinja2 template expression or predefined helper name
        context: Template context with tasks, variables, etc.

    Returns:
        Boolean result of condition evaluation

    Raises:
        ValueError: If expression is invalid or evaluation fails
    """
    from jinja2 import Template, TemplateSyntaxError

    # Check if it's a predefined helper
    if expression in LOOP_HELPERS:
        expression = LOOP_HELPERS[expression]

    try:
        # Wrap expression in {% if %} to get boolean result
        template_str = f"{{% if {expression} %}}true{{% endif %}}"
        template = Template(template_str)
        result = template.render(**context)
        return result.strip() == "true"
    except TemplateSyntaxError as e:
        raise ValueError(f"Invalid loop condition expression '{expression}': {e}") from e
    except Exception as e:
        raise ValueError(f"Error evaluating loop condition '{expression}': {e}") from e


def _prepare_retry_context(step_context: Dict[str, Any], step: Any, attempt: int, errors: List[str]) -> None:
    """Add retry-specific variables to step context.

    Args:
        step_context: Step context to update
        step: Step configuration
        attempt: Current attempt number (0-indexed)
        errors: List of previous errors
    """
    step_context["is_retry"] = attempt > 0
    step_context["retry_count"] = attempt
    step_context["max_retries"] = step.max_retries
    step_context["last_error"] = errors[-1] if errors else ""
    step_context["all_errors"] = errors


def _show_step_progress_message(
    custom_logger: Any,
    step_header: str,
    attempt: int,
    max_retries: int,
    i: int,
    step_name: str,
    total_steps: int,
    max_attempts: int,
    debug: bool,
    event_bus: Optional["EventBus"],
) -> None:
    """Display step progress message in UI.

    Args:
        custom_logger: Logger for UI updates
        step_header: Formatted step header
        attempt: Current attempt number (0-indexed)
        max_retries: Maximum retries allowed
        i: Step number (1-indexed)
        step_name: Name of the step
        total_steps: Total number of steps
        max_attempts: Total attempts (retries + 1)
        debug: Debug mode flag
        event_bus: Event bus for debug messages
    """
    if not debug:
        set_multistep_ui_context(custom_logger, i, step_name, total_steps)
        if attempt > 0:
            print_step_progress(custom_logger, step_header, f"Retry {attempt}/{max_retries}...", debug, "yellow")
        else:
            print_step_progress(custom_logger, step_header, "Starting...", debug, "cyan")

    if debug and event_bus:
        from tsugite.events import DebugMessageEvent

        if attempt > 0:
            event_bus.emit(
                DebugMessageEvent(
                    message=f"DEBUG: Retrying Step {i}/{total_steps}: {step_name} "
                    f"(Attempt {attempt + 1}/{max_attempts})"
                )
            )
        else:
            event_bus.emit(DebugMessageEvent(message=f"DEBUG: Executing Step {i}/{total_steps}: {step_name}"))


def _render_step_template(
    step: Any,
    step_context: Dict[str, Any],
    debug: bool,
    event_bus: Optional["EventBus"],
) -> str:
    """Render step template with current context.

    Args:
        step: Step configuration
        step_context: Current step context
        debug: Debug mode flag
        event_bus: Event bus for debug output

    Returns:
        Rendered step prompt

    Raises:
        Exception: If template rendering fails
    """
    # Execute tool directives in this step's content
    step_modified_content, step_tool_context = execute_tool_directives(step.content, step_context, event_bus)
    step_context.update(step_tool_context)

    # Render this step's content with current context
    renderer = AgentRenderer()
    rendered_step_prompt = renderer.render(step_modified_content, step_context)

    if debug and event_bus:
        from tsugite.events import DebugMessageEvent

        event_bus.emit(DebugMessageEvent(message="\n[bold]Rendered Prompt:[/bold]\n" + rendered_step_prompt))

    return rendered_step_prompt


async def _execute_step_with_retries(
    step: Any,
    step_context: Dict[str, Any],
    agent: Any,
    i: int,
    total_steps: int,
    steps: List[Any],
    step_header: str,
    exec_options: ExecutionOptions,
    custom_logger: Optional[Any],
    event_bus: Optional["EventBus"] = None,
    assigned_vars: Optional[set] = None,
) -> tuple[str, float]:
    """Execute a step with automatic retries on failure.

    Handles template rendering, step execution, error handling, and metrics recording.
    Retries up to max_retries times before failing.
    """
    debug = exec_options.debug
    max_attempts = step.max_retries + 1
    errors = []
    step_start_time = time.time()

    for attempt in range(max_attempts):
        # Add retry context variables
        _prepare_retry_context(step_context, step, attempt, errors)

        # Show progress message
        _show_step_progress_message(
            custom_logger,
            step_header,
            attempt,
            step.max_retries,
            i,
            step.name,
            total_steps,
            max_attempts,
            debug,
            event_bus,
        )

        # Render step template
        try:
            rendered_step_prompt = _render_step_template(step, step_context, debug, event_bus)
        except Exception as e:
            error_msg = f"Template rendering failed: {e}"
            errors.append(error_msg)

            if attempt == max_attempts - 1:
                clear_multistep_ui_context(custom_logger)
                error_msg = _build_step_error_message(
                    error_type="Template Rendering Failed",
                    step_name=step.name,
                    step_number=i,
                    total_steps=total_steps,
                    errors=errors,
                    available_vars=list(step_context.keys()),
                    previous_step=steps[i - 2].name if i > 1 else "None",
                    max_attempts=max_attempts,
                    debug_tips=[
                        "1. Check for undefined variables in step template",
                        "2. Verify previous steps assigned expected variables",
                        "3. Run with --debug to see full context",
                    ],
                )
                raise RuntimeError(error_msg)

            if step.retry_delay > 0:
                time.sleep(step.retry_delay)
            continue

        # Prepare variables and build PreparedAgent
        injectable_vars = _build_injectable_vars(step_context, assigned_vars)
        prepared = _build_prepared_agent_for_step(
            agent=agent,
            rendered_step_prompt=rendered_step_prompt,
            step_context=step_context,
        )

        # Execute this step as a full agent run
        try:

            async def execute_step():
                coro = _execute_agent_with_prompt(
                    prepared=prepared,
                    exec_options=exec_options,
                    custom_logger=custom_logger,
                    skip_task_reset=True,
                    model_kwargs=step.model_kwargs,
                    injectable_vars=injectable_vars,
                )
                if step.timeout:
                    return await asyncio.wait_for(coro, timeout=step.timeout)
                else:
                    return await coro

            step_result = await execute_step()

            # Store result in context if assign variable specified
            if step.assign_var:
                step_context[step.assign_var] = step_result
                if assigned_vars is not None:
                    assigned_vars.add(step.assign_var)
                if debug and event_bus:
                    from tsugite.events import DebugMessageEvent

                    event_bus.emit(DebugMessageEvent(message=f"Assigned result to variable: {step.assign_var}"))

            # Show step completion
            if not debug:
                clear_multistep_ui_context(custom_logger)
                print_step_progress(custom_logger, step_header, "Complete", debug, "green")

            # Calculate duration and return
            step_duration = time.time() - step_start_time
            return step_result, step_duration

        except asyncio.TimeoutError:
            error_msg = f"Step timed out after {step.timeout} seconds"
            errors.append(error_msg)
        except Exception as e:
            error_msg = str(e)
            errors.append(error_msg)

        # If not last attempt, handle retry delay and continue
        if attempt < max_attempts - 1:
            if step.retry_delay > 0:
                time.sleep(step.retry_delay)
            if not debug and event_bus:
                from tsugite.events import WarningEvent

                event_bus.emit(WarningEvent(message=f"Step '{step.name}' failed: {error_msg}"))
        else:
            # Last attempt failed
            clear_multistep_ui_context(custom_logger)
            error_msg = _build_step_error_message(
                error_type="Execution Failed",
                step_name=step.name,
                step_number=i,
                total_steps=total_steps,
                errors=errors,
                available_vars=list(_build_injectable_vars(step_context, assigned_vars).keys()),
                previous_step=steps[i - 2].name if i > 1 else "None",
                max_attempts=max_attempts,
                debug_tips=[
                    "1. Run with --debug to see rendered prompts",
                    "2. Check variable values in previous steps",
                    "3. Verify step dependencies are correct",
                ],
            )
            raise RuntimeError(error_msg)

    # Should never reach here, but for type safety
    raise RuntimeError("Unexpected: Retry loop completed without success or raising")


def _should_repeat_step(
    step: Any, step_context: Dict[str, Any], iteration: int, debug: bool, event_bus: Optional["EventBus"] = None
) -> bool:
    """Determine if a step should repeat based on loop conditions.

    Evaluates repeat_while, repeat_until, and max_iterations to decide
    whether the step should execute again.

    Args:
        step: Step configuration with repeat conditions
        step_context: Current step context for condition evaluation
        iteration: Current iteration count (1-indexed)
        debug: Whether debug mode is active
        event_bus: Optional event bus for emitting debug/warning messages

    Returns:
        True if step should repeat, False otherwise
    """
    should_repeat = False

    # Evaluate repeat conditions
    if step.repeat_while:
        try:
            should_repeat = evaluate_loop_condition(step.repeat_while, step_context)
            if debug and event_bus:
                from tsugite.events import DebugMessageEvent

                event_bus.emit(DebugMessageEvent(message=f"Loop condition (while): {should_repeat}"))
        except Exception as e:
            if event_bus:
                from tsugite.events import WarningEvent

                event_bus.emit(WarningEvent(message=f"Loop condition evaluation failed: {e}"))
            should_repeat = False

    elif step.repeat_until:
        try:
            condition_met = evaluate_loop_condition(step.repeat_until, step_context)
            should_repeat = not condition_met  # Repeat UNTIL condition is true
            if debug and event_bus:
                from tsugite.events import DebugMessageEvent

                event_bus.emit(
                    DebugMessageEvent(
                        message=f"Loop condition (until): condition_met={condition_met}, repeat={should_repeat}"
                    )
                )
        except Exception as e:
            if event_bus:
                from tsugite.events import WarningEvent

                event_bus.emit(WarningEvent(message=f"Loop condition evaluation failed: {e}"))
            should_repeat = False

    # Safety check: max iterations
    if should_repeat and iteration >= step.max_iterations:
        if event_bus:
            from tsugite.events import WarningEvent

            event_bus.emit(
                WarningEvent(
                    message=f"⚠️  Step '{step.name}' reached max_iterations ({step.max_iterations}). "
                    f'Use max_iterations="N" to increase limit.'
                )
            )
        should_repeat = False

    return should_repeat


async def _run_multistep_agent_impl(
    agent_path: Path,
    prompt: str,
    exec_options: Optional[ExecutionOptions] = None,
    context: Optional[Dict[str, Any]] = None,
    custom_logger: Optional[Any] = None,
) -> str:
    """Async implementation of multi-step agent execution."""
    if exec_options is None:
        exec_options = ExecutionOptions()
    from tsugite.md_agents import extract_step_directives, has_step_directives

    if context is None:
        context = {}

    # Parse agent (with inheritance resolution)
    try:
        agent = parse_agent_file(agent_path)
    except Exception as e:
        raise ValueError(f"Failed to parse agent file: {e}")

    # Set current agent in thread-local storage for spawn_agent tracking
    set_current_agent(agent.config.name)

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

        # Create event_bus for emitting events throughout multi-step execution
        from tsugite.events import DebugMessageEvent, EventBus, InfoEvent, WarningEvent

        event_bus = EventBus()
        ui_handler = get_ui_handler(custom_logger)
        if ui_handler:
            event_bus.subscribe(ui_handler.handle_event)

        # Check if running in interactive mode
        interactive_mode = is_interactive()

        # Initialize context with user prompt
        step_context = {
            **context,
            "user_prompt": prompt,
            "is_interactive": interactive_mode,
            "tools": agent.config.tools,
            "is_subagent": context.get("is_subagent", False),
            "parent_agent": context.get("parent_agent", None),
        }

        # Execute prefetch once (before any steps)
        if agent.config.prefetch:
            try:
                prefetch_context = execute_prefetch(agent.config.prefetch, event_bus)
                step_context.update(prefetch_context)
            except Exception as e:
                event_bus.emit(WarningEvent(message=f"Prefetch execution failed: {e}"))

        # Execute each step sequentially
        final_result = None
        step_metrics: List[StepMetrics] = []
        assigned_vars: set = set()  # Track user-assigned variables for namespace isolation

        for i, step in enumerate(steps, 1):
            # Add step information to context for this step
            step_context["step_number"] = i
            step_context["step_name"] = step.name
            step_context["total_steps"] = len(steps)

            # Loop control: iterate if step has repeat_while or repeat_until
            iteration = 0
            step_is_looping = bool(step.repeat_while or step.repeat_until)

            while True:
                iteration += 1

                # Add iteration context
                step_context["iteration"] = iteration
                step_context["max_iterations"] = step.max_iterations
                step_context["is_looping_step"] = step_is_looping

                # Show step progress (unless in debug mode which has its own output)
                if step_is_looping:
                    step_header = f"[Step {i}/{len(steps)}: {step.name} (Iteration {iteration})]"
                else:
                    step_header = f"[Step {i}/{len(steps)}: {step.name}]"

                # Execute step with automatic retries
                step_start_time = time.time()
                try:
                    step_result, step_duration = await _execute_step_with_retries(
                        step=step,
                        step_context=step_context,
                        agent=agent,
                        i=i,
                        total_steps=len(steps),
                        steps=steps,
                        step_header=step_header,
                        exec_options=exec_options,
                        custom_logger=custom_logger,
                        event_bus=event_bus,
                        assigned_vars=assigned_vars,
                    )

                    # Success - store result and record metrics
                    final_result = step_result
                    step_metrics.append(
                        StepMetrics(
                            step_name=step.name,
                            step_number=i,
                            duration=step_duration,
                            status="success",
                        )
                    )

                except RuntimeError as e:
                    # Step execution failed after all retries
                    if step.continue_on_error:
                        # Log warning but continue execution
                        clear_multistep_ui_context(custom_logger)

                        warning_msg = f"⚠ Step '{step.name}' failed but continuing (continue_on_error=true)"
                        event_bus.emit(WarningEvent(message=warning_msg))
                        event_bus.emit(InfoEvent(message=f"Error: {str(e)}"))

                        # Assign None to the variable if specified
                        if step.assign_var:
                            step_context[step.assign_var] = None
                            assigned_vars.add(step.assign_var)
                            if exec_options.debug:
                                event_bus.emit(
                                    DebugMessageEvent(message=f"Assigned None to variable: {step.assign_var}")
                                )

                        # Record metrics for skipped step
                        step_duration = time.time() - step_start_time
                        step_metrics.append(
                            StepMetrics(
                                step_name=step.name,
                                step_number=i,
                                duration=step_duration,
                                status="skipped",
                                error=str(e),
                            )
                        )
                    else:
                        # Re-raise if not continuing on error
                        raise

                # End of step execution - now check if we should repeat the step

                # Check if we should repeat this step (loop control)
                should_repeat = _should_repeat_step(step, step_context, iteration, exec_options.debug, event_bus)

                # Exit while loop if we shouldn't repeat
                if not should_repeat:
                    if step_is_looping and iteration > 1 and not exec_options.debug:
                        event_bus.emit(InfoEvent(message=f"Step '{step.name}' completed after {iteration} iterations"))
                    break

                if not exec_options.debug:
                    event_bus.emit(InfoEvent(message=f"🔁 Repeating step '{step.name}' (iteration {iteration + 1})"))

            # End of while True loop for step iteration

        # Display metrics summary
        if step_metrics:
            display_step_metrics(step_metrics, custom_logger if custom_logger else None)

        return final_result or ""
    finally:
        # Always clear the current agent context when done
        clear_current_agent()
        clear_allowed_agents()

        # Clean up pending tasks (but not LiteLLM clients - that's handled by run_async_with_cleanup wrapper)
        # ONLY run cleanup for top-level agents, not spawned agents
        import asyncio
        import threading

        if threading.current_thread() == threading.main_thread():
            # Get all tasks except the current one
            current_task = asyncio.current_task()
            all_tasks = asyncio.all_tasks()
            pending_tasks = [task for task in all_tasks if task is not current_task and not task.done()]

            # Cancel all pending tasks
            for task in pending_tasks:
                task.cancel()

            # Wait for all tasks to be cancelled
            if pending_tasks:
                await asyncio.gather(*pending_tasks, return_exceptions=True)


def run_multistep_agent(
    agent_path: Path,
    prompt: str,
    exec_options: Optional[ExecutionOptions] = None,
    context: Optional[Dict[str, Any]] = None,
    custom_logger: Optional[Any] = None,
) -> str:
    """Synchronous wrapper for multi-step agent execution.

    Args:
        agent_path: Path to agent markdown file
        prompt: User prompt/task for the agent
        exec_options: Execution options (model, debug, stream, etc.)
        context: Additional context variables
        custom_logger: Custom logger for agent output

    Returns:
        Result from the final step
    """
    import asyncio

    if exec_options is None:
        exec_options = ExecutionOptions()

    return asyncio.run(
        _run_multistep_agent_impl(
            agent_path=agent_path,
            prompt=prompt,
            exec_options=exec_options,
            context=context,
            custom_logger=custom_logger,
        )
    )


async def run_multistep_agent_async(
    agent_path: Path,
    prompt: str,
    exec_options: Optional[ExecutionOptions] = None,
    context: Optional[Dict[str, Any]] = None,
    custom_logger: Optional[Any] = None,
) -> str:
    """Asynchronous wrapper for multi-step agent execution."""
    if exec_options is None:
        exec_options = ExecutionOptions()

    return await _run_multistep_agent_impl(
        agent_path=agent_path,
        prompt=prompt,
        exec_options=exec_options,
        context=context,
        custom_logger=custom_logger,
    )


def preview_multistep_agent(
    agent_path: Path,
    prompt: str,
    context: Optional[Dict[str, Any]] = None,
    console: Optional[Any] = None,
    custom_logger: Optional[Any] = None,
):
    """Preview multi-step agent execution without running it.

    Shows the execution plan including steps, dependencies, attributes,
    and estimated resource usage.

    Args:
        agent_path: Path to agent markdown file
        prompt: User prompt/task for the agent
        context: Additional context variables
        console: Rich Console instance (defaults to stderr console)
        custom_logger: Custom logger with ui_handler for event emission
    """
    import re

    from rich.table import Table

    from tsugite.events import EventBus, InfoEvent, WarningEvent

    # Check if we should use event system
    ui_handler = get_ui_handler(custom_logger)
    event_bus = None
    if ui_handler:
        event_bus = EventBus()
        event_bus.subscribe(ui_handler.handle_event)

    # Use provided console or default to stderr (for non-event output)
    if console is None and not event_bus:
        console = _stderr_console

    # Helper to output messages (via events or console)
    def output(msg: str, is_warning: bool = False):
        if event_bus:
            if is_warning:
                event_bus.emit(WarningEvent(message=msg))
            else:
                event_bus.emit(InfoEvent(message=msg))
        elif console:
            console.print(msg)  # noqa: T201 - Intentional fallback when no event system available

    # Parse agent (with inheritance resolution)
    try:
        agent = parse_agent_file(agent_path)
    except Exception as e:
        output(f"[red]Error parsing agent: {e}[/red]")
        return

    # Extract steps
    from tsugite.md_agents import extract_step_directives, has_step_directives

    if not has_step_directives(agent.content):
        output("[yellow]This is a single-step agent (no step directives).[/yellow]", is_warning=True)
        output("[dim]Dry-run preview is for multi-step agents only.[/dim]")
        return

    try:
        preamble, steps = extract_step_directives(agent.content)
    except Exception as e:
        output(f"[red]Error extracting steps: {e}[/red]")
        return

    # Display header
    output("")
    output("[bold]Dry-Run Preview: Multi-Step Agent[/bold]")
    output("═" * 60)
    output(f"Agent: {agent.config.name}")
    output(f"File: {agent_path.name}")
    output(f"Prompt: {prompt}")
    output(f"Steps: {len(steps)}")
    output(f"Model: {agent.config.model or 'default'}")
    output(f"Tools: {', '.join(agent.config.tools) if agent.config.tools else 'None'}")
    output("")

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
        # Filter out template helpers and metadata (these are always available, not real deps)
        builtin_vars = {
            "user_prompt",
            "step_number",
            "step_name",
            "total_steps",
            "is_retry",
            "retry_count",
            "max_retries",
            "last_error",
            "all_errors",
            "is_interactive",
            "tools",
            "is_subagent",
            "parent_agent",
            "iteration",
            "max_iterations",
            "is_looping_step",
            "now",
            "today",
        }
        real_deps = variables_used - builtin_vars

        deps_str = ", ".join(sorted(real_deps)) if real_deps else "—"

        table.add_row(str(i), step.name, attr_str, deps_str)

    # Output table (via console fallback since tables need special rendering)
    if event_bus:
        # For events, render table to string
        from io import StringIO

        buffer = StringIO()
        temp_console = get_display_console(custom_logger)
        temp_console.file = buffer
        temp_console.print(table)  # noqa: T201 - Rendering to buffer, not user console
        event_bus.emit(InfoEvent(message=buffer.getvalue()))
    elif console:
        console.print(table)  # noqa: T201 - Intentional fallback when no event system available

    output("")

    # Warnings
    warning_messages = []
    for step in steps:
        if step.timeout and step.timeout < 30:
            warning_messages.append(f"⚠ Step '{step.name}' has short timeout ({step.timeout}s)")
        if step.continue_on_error and not step.assign_var:
            warning_messages.append(f"⚠ Step '{step.name}' has continue_on_error but no assign variable")

    if warning_messages:
        output("[bold]Warnings:[/bold]")
        output("─" * 60)
        for warning in warning_messages:
            output(f"  [yellow]{warning}[/yellow]", is_warning=True)
        output("")

    output("━" * 60)
    output("[dim]Note: This is a preview only. No tools will be executed.[/dim]")
    output("[dim]Remove --dry-run to execute the agent.[/dim]")
    output("")
