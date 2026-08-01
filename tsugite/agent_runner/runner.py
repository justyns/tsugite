"""Agent execution engine using TsugiteAgent."""

import asyncio
import fnmatch
import logging

logger = logging.getLogger(__name__)
from dataclasses import dataclass, field  # noqa: E402
from pathlib import Path  # noqa: E402
from types import SimpleNamespace  # noqa: E402
from typing import TYPE_CHECKING, Any, Dict, List, Optional  # noqa: E402

from tsugite.config import get_xdg_data_path  # noqa: E402
from tsugite.core.agent import TsugiteAgent  # noqa: E402
from tsugite.core.executor_registry import get_executor_class  # noqa: E402
from tsugite.core.proxy import _parse_pattern  # noqa: E402
from tsugite.exceptions import (  # noqa: E402
    AgentExecutionError,
    is_prompt_too_long_error,
    is_unresumable_history_error,
)
from tsugite.md_agents import AgentConfig, parse_agent_file  # noqa: E402
from tsugite.models import resolve_effective_model, strip_reserved_model_kwargs  # noqa: E402
from tsugite.options import ExecutionOptions  # noqa: E402
from tsugite.renderer import AgentRenderer  # noqa: E402
from tsugite.utils import is_interactive  # noqa: E402

from .helpers import (  # noqa: E402
    _stderr_console,
    build_sandbox_policy,
    clear_allowed_agents,
    clear_current_agent,
    clear_sandbox_context,
    get_display_console,
    get_ui_handler,
    set_allowed_secrets,
    set_current_agent,
    set_sandbox_context,
)
from .models import AgentExecutionResult, AgentSkippedError  # noqa: E402

# Display constants for truncating long output
MAX_VARIABLE_PREVIEW_LENGTH = 100  # Max characters to show in variable documentation
MAX_CONTENT_PREVIEW_LENGTH = 200  # Max characters to show in debug attachment previews


def _resolve_state_path(session_id: Optional[str]) -> Optional[Path]:
    """Return the per-session JSON state path, or None for an ephemeral run."""
    if not session_id:
        return None
    return get_xdg_data_path("state") / session_id / "state.json"


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
    model_string = resolve_effective_model(model_override, agent_config.model)

    if not model_string:
        raise RuntimeError(
            "No model specified. Set a model in agent frontmatter, use --model flag, "
            "or set a default with 'tsugite config set-default <model>'"
        )

    return model_string


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


def _render_args(value: Any, renderer: AgentRenderer) -> Any:
    """Recursively render Jinja in string leaves of a tool args structure.

    Non-string values pass through unchanged. Strings without Jinja markers
    skip the render call.
    """
    if isinstance(value, str):
        if "{{" not in value and "{%" not in value:
            return value
        return renderer.render_string(value, {})
    if isinstance(value, dict):
        return {k: _render_args(v, renderer) for k, v in value.items()}
    if isinstance(value, list):
        return [_render_args(v, renderer) for v in value]
    if isinstance(value, tuple):
        return tuple(_render_args(v, renderer) for v in value)
    return value


def execute_prefetch(prefetch_config: List[Dict[str, Any]], event_bus: Optional["EventBus"] = None) -> Dict[str, Any]:
    if not prefetch_config:
        return {}

    from tsugite.tools import call_tool

    # Set event_bus in context so tools can access it
    _setup_event_context(event_bus)

    renderer = AgentRenderer()
    context = {}
    for config in prefetch_config:
        tool_name = config.get("tool")
        args = config.get("args", {})
        assign_name = config.get("assign")

        if not tool_name or not assign_name:
            continue

        try:
            rendered_args = _render_args(args, renderer)
        except Exception as e:
            raise RuntimeError(f"Prefetch render failed for tool '{tool_name}': {e}") from e

        try:
            context[assign_name] = call_tool(tool_name, **rendered_args)
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
    renderer = AgentRenderer()

    for directive in directives:
        try:
            rendered_args = _render_args(directive.args, renderer)
        except Exception as e:
            raise RuntimeError(f"Tool directive render failed for '{directive.name}': {e}") from e

        try:
            # Execute the tool
            result = call_tool(directive.name, **rendered_args)
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


def _make_pre_llm_call_callback(hooks_dir: Path, agent_name: str):
    """Build an async callback that fires pre_llm_call hooks to mutate outgoing
    messages in place. Returns None when no such hooks exist (keeps the hot path clean)."""
    from tsugite.hooks import fire_hooks, load_hooks_config

    cfg = load_hooks_config(hooks_dir)
    if not cfg or not cfg.pre_llm_call:
        return None

    async def _callback(messages, model):
        await fire_hooks(
            hooks_dir,
            "pre_llm_call",
            {"messages": messages, "model": model, "agent": agent_name},
            interactive=is_interactive(),
        )

    return _callback


async def _cancel_sibling_tasks() -> None:
    """At the end of a top-level run, cancel the event loop's other pending tasks.

    Only top-level agents (on the main thread) clean up; a spawned agent leaves
    the shared loop alone. Provider-client cleanup is handled separately by the
    run_async_with_cleanup wrapper.
    """
    import threading

    if threading.current_thread() != threading.main_thread():
        return
    current_task = asyncio.current_task()
    pending = [t for t in asyncio.all_tasks() if t is not current_task and not t.done()]
    for task in pending:
        task.cancel()
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)


async def _execute_agent_with_prompt(
    prepared: "PreparedAgent",
    exec_options: Optional[ExecutionOptions] = None,
    workspace: Optional[Any] = None,
    custom_logger: Optional[Any] = None,
    model_kwargs: Optional[Dict[str, Any]] = None,
    injectable_vars: Optional[Dict[str, Any]] = None,
    previous_messages: Optional[List[Dict]] = None,
    path_context: Optional[Any] = None,
    resume_session: Optional[str] = None,
    resume_after_compaction: bool = False,
    hook_vars: Optional[Dict[str, str]] = None,
    continue_conversation_id: Optional[str] = None,
    user_input_for_history: Optional[str] = None,
    channel_metadata: Optional[Dict[str, Any]] = None,
) -> str | AgentExecutionResult:
    """Execute agent with a prepared agent.

    Shared by single-shot runs and by each step of a multi-step run.
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
    from tsugite.events import EventBus, WarningEvent

    event_bus = EventBus()
    if ui_handler:
        event_bus.subscribe(ui_handler.handle_event)

    # Subscribe workspace hooks (if configured)
    from tsugite.hooks import setup_hook_handler

    if workspace:
        hooks_dir = workspace.path
    elif path_context:
        hooks_dir = path_context.effective_cwd
    else:
        hooks_dir = Path.cwd()
    setup_hook_handler(hooks_dir, event_bus, interactive=is_interactive())
    pre_llm_call_cb = _make_pre_llm_call_callback(hooks_dir, agent_config.name)

    # Start with tools from prepared agent
    tools = list(prepared.tools)  # Make a copy

    # Filter out interactive tools in subagent mode
    import os

    if os.environ.get("TSUGITE_SUBAGENT_MODE") == "1":
        tools = [t for t in tools if t.name not in ["ask_user", "ask_user_batch"]]

    # Filter out interactive_only tools when no UI handler (e.g. scheduled tasks)
    if not ui_handler:
        from tsugite.tools import get_interactive_only_names

        interactive_names = get_interactive_only_names()
        tools = [t for t in tools if t.name not in interactive_names]

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

    # Get model string
    model_string = _get_model_string(exec_options.model_override, agent_config)

    # Merge model_kwargs from the agent frontmatter first (lowest precedence), then
    # explicit caller kwargs override. This lets agents declare e.g.
    # `model_kwargs: {response_format: {type: json_object}}` once and have every
    # invocation get structured output without each caller threading it through.
    final_model_kwargs = {}
    if hasattr(agent_config, "model_kwargs") and agent_config.model_kwargs:
        final_model_kwargs.update(agent_config.model_kwargs)
    final_model_kwargs.update(model_kwargs or {})
    # Reject provider-call args (messages/model/stream) that would collide with the
    # explicit keywords splatted alongside **model_kwargs in core/agent.py.
    final_model_kwargs = strip_reserved_model_kwargs(final_model_kwargs)

    # Resolve reasoning_effort. Precedence (highest wins): exec_options override >
    # caller-supplied model_kwargs > agent.model_kwargs > agent.reasoning_effort field.
    # The override is a user-facing CLI/daemon knob - it MUST beat any agent default,
    # including one baked into agent.model_kwargs (which would otherwise shadow it).
    # Caller kwargs and agent.model_kwargs are already in final_model_kwargs from the merge
    # above, so only the override and the agent.reasoning_effort fallback resolve here.
    effort_override = getattr(exec_options, "reasoning_effort_override", None)
    if effort_override:
        final_model_kwargs["reasoning_effort"] = effort_override
    elif "reasoning_effort" not in final_model_kwargs:
        if hasattr(agent_config, "reasoning_effort") and agent_config.reasoning_effort:
            final_model_kwargs["reasoning_effort"] = agent_config.reasoning_effort

    from tsugite.models import resolve_reasoning_effort

    if "reasoning_effort" in final_model_kwargs:
        resolved_effort = resolve_reasoning_effort(model_string, final_model_kwargs["reasoning_effort"])
        if resolved_effort is None:
            final_model_kwargs.pop("reasoning_effort", None)
        else:
            final_model_kwargs["reasoning_effort"] = resolved_effort

    # Create executor with workspace directory and event bus
    workspace_dir = _resolve_workspace_dir(workspace, path_context)

    state_path = _resolve_state_path(continue_conversation_id)

    # The daemon config (exec_options) is the ceiling; the agent's frontmatter may
    # only tighten it (opt in, force no_network, narrow domains), never loosen.
    # build_sandbox_policy returns (None, None) when the sandbox is off; the same
    # helper backs `tsu exec` so the two paths never drift.
    sandbox_config, sandbox_ctx = build_sandbox_policy(
        exec_options, workspace_dir=workspace_dir, agent_config=agent_config
    )

    if sandbox_config is not None:
        # bwrap sandboxing is a subprocess feature, so a sandboxed run always uses
        # the subprocess executor regardless of the configured default backend.
        from tsugite.core.subprocess_executor import SubprocessExecutor

        executor = SubprocessExecutor(
            workspace_dir=workspace_dir,
            event_bus=event_bus,
            path_context=path_context,
            sandbox_config=sandbox_config,
            state_path=state_path,
            session_id=continue_conversation_id,
        )
        # Presence of the context == "this agent is sandboxed".
        set_sandbox_context(sandbox_ctx)
    else:
        executor_cls = get_executor_class()
        executor = executor_cls(
            workspace_dir=workspace_dir,
            event_bus=event_bus,
            path_context=path_context,
            state_path=state_path,
            session_id=continue_conversation_id,
        )
        set_sandbox_context(None)

    # Inject variables into executor (for multi-step agents)
    if injectable_vars:
        await executor.send_variables(injectable_vars)

    # Open or create the session storage so the agent can record events live
    # (model_request/response, code_execution, tool_invocation). Without this,
    # the only events stored would be the final aggregate written at end-of-run
    # by save_run_to_history — losing the per-turn trace the UI shows.
    from .history_integration import open_or_create_session, record_user_input

    session_storage = None
    try:
        session_storage = open_or_create_session(
            agent_path=prepared.agent.file_path,
            agent_name=prepared.agent_config.name
            or (prepared.agent.file_path.stem if prepared.agent.file_path else "agent"),
            model=model_string,
            continue_conversation_id=continue_conversation_id,
            workspace=workspace.name if workspace else None,
        )
        if session_storage is not None:
            # Record only what the user actually typed (or said). The full
            # rendered template — tools, skills, instructions, daemon context —
            # is reproducible from the agent file and recorded as a separate
            # model_request event; replaying the whole thing here would clog the
            # chat bubble with noise the user never wrote.
            display_prompt = user_input_for_history or prepared.original_prompt or prepared.rendered_prompt
            record_user_input(
                session_storage,
                display_prompt,
                attachments=prepared.attachments,
                channel_metadata=channel_metadata,
            )
    except Exception as e:
        logger.debug("Could not open session storage for live event recording: %s", e)
        session_storage = None

    # Create and run agent
    try:
        agent = TsugiteAgent(
            model_string=model_string,
            tools=tools,
            instructions=combined_instructions or "",
            max_turns=exec_options.max_turns_override or agent_config.max_turns,
            executor=executor,
            model_kwargs=final_model_kwargs,
            event_bus=event_bus,
            model_name=model_string,
            attachments=prepared.attachments,
            skills=prepared.skills,
            expiring_skills=prepared.expiring_skills,
            previous_messages=previous_messages,
            resume_session=resume_session,
            resume_after_compaction=resume_after_compaction,
            hook_vars=hook_vars,
            storage=session_storage,
            pre_llm_call=pre_llm_call_cb,
        )
        # Set event_bus in context so tools can access it during execution
        _setup_event_context(event_bus)

        # Set default interaction backend for CLI if none is set
        from tsugite.interaction import TerminalInteractionBackend, get_interaction_backend, set_interaction_backend

        if get_interaction_backend() is None and is_interactive():
            set_interaction_backend(TerminalInteractionBackend())

        # Run agent
        result = await agent.run(
            prepared.rendered_prompt,
            return_full_result=exec_options.return_token_usage,
            stream=exec_options.stream,
        )

        # Extract and display reasoning content if present
        _extract_reasoning_content(agent, custom_logger)

        try:
            from tsugite.usage import get_usage_store

            get_usage_store().record(
                agent=prepared.agent_config.name,
                model=model_string,
                source="cli",
                total_tokens=agent.total_tokens,
                cost_usd=agent.reported_cost,
                cache_creation_tokens=agent.cache_creation_tokens,
                cache_read_tokens=agent.cache_read_tokens,
            )
        except Exception as e:
            logger.debug("Failed to record usage: %s", e)

        # Return appropriate format
        if exec_options.return_token_usage:
            from tsugite.core.agent import AgentResult

            if isinstance(result, AgentResult):
                step_count = len(result.steps) if result.steps else 0
                steps_list = result.steps if result.steps else []

                # If result has error, raise it AFTER we've already extracted the steps
                # The exception will be caught by the caller, but steps are already available
                if result.error:
                    raise AgentExecutionError(
                        f"Agent execution failed: {result.error}",
                        execution_steps=steps_list,
                        token_usage=result.token_usage,
                        cost=result.cost,
                        step_count=step_count,
                        partial_output=str(result.output) if result.output else None,
                    )

                return AgentExecutionResult(
                    response=str(result.output),
                    token_count=result.token_usage,
                    cost=result.cost,
                    step_count=step_count,
                    execution_steps=steps_list,
                    system_message=prepared.system_message,
                    attachments=prepared.attachments,
                    provider_state=result.provider_state,
                    last_input_tokens=result.last_input_tokens,
                    cache_creation_tokens=agent.cache_creation_tokens,
                    cache_read_tokens=agent.cache_read_tokens,
                    session_id=session_storage.session_id if session_storage else None,
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
                    session_id=session_storage.session_id if session_storage else None,
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
        # Drop the sandbox policy from this thread so a later run on the same
        # pooled thread (daemon asyncio.to_thread) can't read a stale context
        # before it sets its own.
        clear_sandbox_context()

        # Clean up subprocess executor temp files
        if hasattr(executor, "cleanup"):
            try:
                executor.cleanup()
            except Exception:
                pass

        await _cancel_sibling_tasks()


def resolve_effective_sandbox(
    *,
    daemon_enabled: bool,
    daemon_domains: list,
    daemon_no_network: bool,
    fm_network: Optional[dict],
    fm_sandbox: Optional[dict],
) -> tuple[bool, list, bool]:
    """Combine the daemon sandbox policy with tighten-only frontmatter overrides.

    The daemon config (or CLI flags) is the ceiling. An agent's frontmatter may make
    itself MORE restricted - opt into the sandbox, force `no_network`, or narrow the
    domain allowlist - but never less: it cannot disable the sandbox or reach a
    domain the daemon didn't allow.

    Returns (enabled, allow_domains, no_network).
    """
    fm_network = fm_network or {}
    fm_sandbox = fm_sandbox or {}

    enabled = bool(daemon_enabled) or bool(fm_sandbox.get("enabled"))
    no_network = bool(daemon_no_network) or bool(fm_sandbox.get("no_network"))

    # Domains the agent declares (network hints + an explicit cap list), capped to
    # the daemon ceiling. An empty ceiling means "all", so the agent's declared set
    # becomes the allowlist (narrowing from all to that set).
    desired = set(fm_network.get("domains") or []) | set(fm_sandbox.get("allow_domains") or [])
    base = list(daemon_domains or [])
    if desired:
        # Keep each desired pattern only if the daemon ceiling permits it, using the
        # proxy's glob semantics (so agent "api.github.com" is kept under daemon
        # "*.github.com"). An empty ceiling permits everything.
        effective = sorted(d for d in desired if _domain_within_ceiling(d, base))
        if not effective:
            # The agent asked only for domains outside the ceiling -> grant none.
            no_network = True
        allow_domains = effective
    else:
        allow_domains = base

    return enabled, allow_domains, no_network


def _domain_within_ceiling(desired_pattern: str, ceiling: list) -> bool:
    """True if a desired domain:port pattern is permitted by the daemon ceiling.

    Uses the proxy's glob + port semantics so e.g. 'api.github.com' is within
    '*.github.com', but 'github.com:22' is NOT within 'github.com' (which only allows
    the default 80/443). The ceiling is a union: the desired ports must be covered by
    the union of port sets from ALL ceiling patterns whose domain glob matches (so
    ['github.com:80','github.com:443'] together cover the default 80/443). An empty
    port set means "all ports" (from '*:*'); an empty ceiling list means "all allowed".
    """
    if not ceiling:
        return True
    d_domain, d_ports = _parse_pattern(desired_pattern.lower())
    # The proxy treats the allowlist as a union, so collect the ports from EVERY
    # ceiling pattern whose domain glob matches and check the desired ports against
    # their union (e.g. ["github.com:80","github.com:443"] together cover 80/443).
    matched = False
    allowed_ports: set = set()
    for c in ceiling:
        c_domain, c_ports = _parse_pattern(c.lower())
        if not fnmatch.fnmatch(d_domain, c_domain):
            continue
        matched = True
        if not c_ports:  # this ceiling pattern allows all ports
            return True
        allowed_ports |= c_ports
    if not matched:
        return False
    # Desired wanting all ports (empty set) needs an all-ports ceiling (handled above).
    return bool(d_ports) and d_ports <= allowed_ports


def _resolve_workspace_dir(workspace: Optional[Any], path_context: Optional[Any]) -> Optional[Path]:
    """Resolve the executor's workspace directory.

    Prefer an explicit Workspace object (CLI workspace runs); otherwise fall back
    to the PathContext's workspace_dir. The daemon passes only a path_context (no
    Workspace), so without this fallback the sandbox would bind/chdir the daemon's
    CWD instead of the agent's workspace (or job worktree).
    """
    if workspace:
        return workspace.path
    if path_context and getattr(path_context, "workspace_dir", None):
        return path_context.workspace_dir
    return None


@dataclass
class _RunSetup:
    """State established once per run and reused by every prompt within it.

    A single-shot agent sends one prompt; a multi-step agent sends one per step.
    Both go through `_run_unit`, so everything resolved before the first prompt
    (hooks, history, resume state, UI wiring) is gathered here rather than
    threaded through as a dozen parameters.
    """

    exec_options: ExecutionOptions
    hooks_dir: Path
    hook_message: str
    hook_vars: Dict[str, str]
    agent_stem: str
    ui_handler: Optional[Any] = None
    on_status: Optional[Any] = None
    on_hook_result: Optional[Any] = None
    workspace: Optional[Any] = None
    custom_logger: Optional[Any] = None
    path_context: Optional[Any] = None
    attachments: Optional[List[Any]] = None
    channel_metadata: Optional[Dict[str, Any]] = None
    user_input_for_history: Optional[str] = None
    previous_messages: List[Dict] = field(default_factory=list)
    resume_session: Optional[str] = None
    resume_after_compaction: bool = False
    conversation_id: Optional[str] = None


def _prepare_step(agent: Any, prompt: str, context: Dict[str, Any], setup: "_RunSetup") -> "PreparedAgent":
    """Run one prompt through the preparation pipeline.

    Shared by normal steps and by steps that delegate via `agent=`, so both see
    the same tool directives and template variables.
    """
    from tsugite.agent_preparation import AgentPreparer

    return AgentPreparer().prepare(
        agent=agent,
        prompt=prompt,
        context=context,
        attachments=setup.attachments,
        path_context=setup.path_context,
    )


async def _run_unit(
    agent: Any,
    prompt: str,
    context: Dict[str, Any],
    setup: _RunSetup,
    *,
    model_kwargs: Optional[Dict[str, Any]] = None,
    injectable_vars: Optional[Dict[str, Any]] = None,
) -> str | AgentExecutionResult:
    """Prepare and execute one prompt: the whole agent, or one step of it.

    Multi-step agents reach here once per step with `agent.content` narrowed to
    that step's content, which is what keeps a step from seeing its siblings'
    instructions.
    """
    from tsugite.hooks import fire_hooks

    exec_options = setup.exec_options

    prepared = _prepare_step(agent, prompt, context, setup)

    post_ctx_results = await fire_hooks(
        setup.hooks_dir,
        "post_context_build",
        {
            "message": setup.hook_message,
            "agent_name": setup.agent_stem,
            "system_message": prepared.system_message[:500] if prepared.system_message else "",
            "rendered_prompt": prepared.rendered_prompt[:500] if prepared.rendered_prompt else "",
            "tools": [t.name for t in prepared.tools] if prepared.tools else [],
        },
        interactive=is_interactive(),
        on_status=setup.on_status,
        on_result=setup.on_hook_result,
    )
    if post_ctx_results.captured:
        if "system_message" in post_ctx_results.captured:
            prepared.system_message = post_ctx_results.captured["system_message"]
        if "rendered_prompt" in post_ctx_results.captured:
            prepared.rendered_prompt = post_ctx_results.captured["rendered_prompt"]

    # Short-circuit if run_if guard evaluated to false
    if prepared.skipped:
        raise AgentSkippedError(prepared.skip_reason or "run_if guard")

    if exec_options.debug:
        import sys

        print(_format_debug_output(prepared), file=sys.stderr)

    execute_kwargs = dict(
        prepared=prepared,
        exec_options=exec_options,
        workspace=setup.workspace,
        custom_logger=setup.custom_logger,
        path_context=setup.path_context,
        hook_vars=setup.hook_vars,
        continue_conversation_id=setup.conversation_id,
        user_input_for_history=setup.user_input_for_history,
        channel_metadata=setup.channel_metadata,
        model_kwargs=model_kwargs,
        injectable_vars=injectable_vars,
    )

    try:
        return await _execute_agent_with_prompt(
            previous_messages=setup.previous_messages,
            resume_session=setup.resume_session,
            resume_after_compaction=setup.resume_after_compaction,
            **execute_kwargs,
        )
    except (RuntimeError, AgentExecutionError) as e:
        err_str = str(e).lower()
        poisoned = is_unresumable_history_error(err_str)
        if setup.resume_session and (
            "process ended" in err_str
            or "no conversation found" in err_str
            or is_prompt_too_long_error(err_str)
            or "format_error_loop" in err_str
            or poisoned
        ):
            logger.warning("Provider session resume failed (%s), retrying with fresh session", e)
            if poisoned:
                # Durably sever the unresumable session so later messages stop
                # re-resolving it from history; the retry below runs fresh.
                from tsugite.agent_runner.history_integration import record_resume_reset

                reset = record_resume_reset(setup.conversation_id)
                # Surface it live on this turn too: the durable record alone only
                # shows on the next reload, so emit an SSE frame the open
                # conversation renders before the fresh retry streams in.
                if reset and setup.ui_handler is not None and hasattr(setup.ui_handler, "_emit"):
                    setup.ui_handler._emit("resume_reset", reset)
            try:
                from tsugite.agent_runner.history_integration import load_and_apply_history

                fallback_messages = load_and_apply_history(setup.conversation_id)
            except Exception:
                logger.warning("Failed to load history for fallback, starting fresh")
                fallback_messages = []
            return await _execute_agent_with_prompt(previous_messages=fallback_messages, **execute_kwargs)
        raise


def run_agent(
    agent_path: Path,
    prompt: str,
    exec_options: Optional[ExecutionOptions] = None,
    context: Optional[Dict[str, Any]] = None,
    custom_logger: Optional[Any] = None,
    continue_conversation_id: Optional[str] = None,
    attachments: Optional[List[Any]] = None,
    path_context: Optional[Any] = None,
    user_input_for_history: Optional[str] = None,
    channel_metadata: Optional[Dict[str, Any]] = None,
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

        # Delegated files handed down by spawn_agent: the parent already gated them
        # for this model, so materialize them into attachments the same way agent
        # frontmatter attachments are (FileHandler.fetch, size gates included).
        delegated_files = stdin_data.get("files") or []
        if delegated_files:
            from tsugite.attachments.delegation import materialize_delegation_attachments

            delegated = materialize_delegation_attachments([Path(f) for f in delegated_files])
            attachments = (attachments or []) + delegated

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
            user_input_for_history=user_input_for_history,
            channel_metadata=channel_metadata,
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
    user_input_for_history: Optional[str] = None,
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

    from tsugite.hooks import fire_pre_message_hooks

    hooks_dir = workspace.path if workspace else (path_context.effective_cwd if path_context else Path.cwd())
    hook_message = context.pop("raw_message", prompt)

    ui_handler = get_ui_handler(custom_logger)
    if ui_handler and hasattr(ui_handler, "_emit"):

        def on_status(msg):
            return ui_handler._emit("hook_status", {"message": msg})

        def on_hook_result(ex):
            return ui_handler._emit("hook_execution", ex.model_dump(exclude={"type", "timestamp"}))

    else:
        on_status = None
        on_hook_result = None

    hook_vars = await fire_pre_message_hooks(
        hooks_dir,
        {"message": hook_message, "agent_name": agent_path.stem},
        interactive=is_interactive(),
        on_status=on_status,
        on_result=on_hook_result,
    )
    context.update(hook_vars)

    # Fire pre_context_build hooks (plugin hooks can inject extra context + blocks)
    from tsugite.hooks import collect_context_blocks, fire_hooks, render_blocks

    context_blocks: list = []
    pre_ctx_results = await fire_hooks(
        hooks_dir,
        "pre_context_build",
        {"message": hook_message, "agent_name": agent_path.stem, **context, "blocks": context_blocks},
        interactive=is_interactive(),
        on_status=on_status,
        on_result=on_hook_result,
    )
    context.update(pre_ctx_results.captured)
    context["context_blocks"] = render_blocks(collect_context_blocks(context_blocks, context.get("rag_context")))

    # Load conversation history if continuing
    previous_messages = []
    resume_session = None
    resume_after_compaction = False
    if continue_conversation_id:
        from tsugite.agent_runner.history_integration import (
            get_resumable_session_state,
            load_and_apply_history,
        )

        # Resume an existing provider session if one was recorded for this conversation.
        session_info = get_resumable_session_state(continue_conversation_id)
        if session_info:
            resume_session = session_info.session_id
            resume_after_compaction = session_info.compacted
            logger.info(
                "Resuming provider session %s (compacted=%s)",
                resume_session,
                resume_after_compaction,
            )
        else:
            logger.debug("No resumable provider session for %s", continue_conversation_id)

        # Load serialized history even when a provider session is resumable:
        # session-owning providers need it as fallback material if the resume
        # replay is rejected (e.g. a poisoned Claude Code sidecar transcript
        # that 400s on every send). Providers ignore it while a resume is live.
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
    set_allowed_secrets(agent_config.allowed_secrets)

    setup = _RunSetup(
        exec_options=exec_options,
        hooks_dir=hooks_dir,
        hook_message=hook_message,
        hook_vars=hook_vars,
        agent_stem=agent_path.stem,
        ui_handler=ui_handler,
        on_status=on_status,
        on_hook_result=on_hook_result,
        workspace=workspace,
        custom_logger=custom_logger,
        path_context=path_context,
        attachments=attachments,
        channel_metadata=channel_metadata,
        user_input_for_history=user_input_for_history,
        previous_messages=previous_messages,
        resume_session=resume_session,
        resume_after_compaction=resume_after_compaction,
        conversation_id=continue_conversation_id,
    )

    try:
        from tsugite.md_agents import has_step_directives

        if has_step_directives(agent.content):
            from .steps import run_steps

            return await run_steps(agent, prompt, context, setup)
        return await _run_unit(agent, prompt, context, setup)
    finally:
        # Always clear the current agent context when done
        clear_current_agent()
        clear_allowed_agents()


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
    output(f"Model: {resolve_effective_model(agent_model=agent.config.model) or 'unknown'}")
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
