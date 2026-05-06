"""Orchestrator for `<!-- tsu:exec -->` directives in agent content.

Mirrors the shape of `execute_tool_directives` (see `runner.py`): parses directives
from content, runs each one in document order, replaces the directive text with an
execution note, and returns the modified content plus a context dict of assigned vars.
"""

from typing import Any, Dict, List, Optional, Tuple

from tsugite.agent_runner.exec_runner import run_python_block
from tsugite.md_agents import ExecDirective, extract_exec_directives


def _emit_warning(event_bus: Optional[Any], message: str) -> None:
    if not event_bus:
        return
    from tsugite.events import WarningEvent

    event_bus.emit(WarningEvent(message=message))


def _splice_replacements(content: str, replacements: List[Tuple[int, int, str]]) -> str:
    """Apply (start, end, replacement) edits in reverse order so earlier offsets stay valid.

    Avoids the O(N^2) cost - and the duplicate-string clobber risk - of running
    `str.replace(raw_match, ...)` once per directive when two blocks happen to share
    identical args+body.
    """
    out = content
    for start, end, replacement in sorted(replacements, key=lambda r: r[0], reverse=True):
        out = out[:start] + replacement + out[end:]
    return out


def _success_note(directive: ExecDirective) -> str:
    if directive.assign_var:
        return f"<!-- Exec '{directive.name}' executed, result in {directive.assign_var} -->"
    return f"<!-- Exec '{directive.name}' executed -->"


def execute_exec_directives(
    content: str,
    existing_context: Optional[Dict[str, Any]] = None,
    sandbox_config: Optional[Any] = None,
    workspace_dir: Optional[str] = None,
    event_bus: Optional[Any] = None,
) -> Tuple[str, Dict[str, Any]]:
    """Execute exec directives in `content` and return (modified_content, exec_context).

    Args:
        content: Markdown content possibly containing `<!-- tsu:exec ... -->` blocks.
        existing_context: Vars from prior pipeline stages (prefetch, tool directives,
            caller). Made available to each exec block as Python locals. Vars assigned
            by an earlier exec block in the same content are also visible to later ones.
        sandbox_config: Optional SandboxConfig; when set, blocks run inside bubblewrap.
        workspace_dir: Optional workspace path for sandbox bind mounts.
        event_bus: Optional EventBus for emitting warnings on continue_on_error paths.

    Returns:
        (modified_content, exec_context) where modified_content has each directive replaced
        with a brief execution note, and exec_context maps assigned-var names to their values.
    """
    # Cheap guard: skip the regex compile-and-walk for the common case of agents
    # with no exec blocks. prepare_agent runs this on every render/run.
    if "tsu:exec" not in content:
        return content, {}

    base_context: Dict[str, Any] = dict(existing_context or {})

    try:
        directives = extract_exec_directives(content)
    except ValueError as e:
        _emit_warning(event_bus, f"Failed to parse exec directives: {e}")
        return content, {}

    if not directives:
        return content, {}

    exec_context: Dict[str, Any] = {}
    replacements: List[Tuple[int, int, str]] = []

    for directive in directives:
        block_locals = {**base_context, **exec_context}
        failure: Optional[str] = None
        result: Optional[ExecBlockResult] = None
        try:
            result = run_python_block(
                code=directive.code,
                locals_dict=block_locals,
                timeout=directive.timeout,
                continue_on_error=directive.continue_on_error,
                sandbox_config=sandbox_config,
                workspace_dir=workspace_dir,
            )
        except Exception as e:
            if not directive.continue_on_error:
                raise
            failure = str(e)

        # run_python_block returns a result with `.error` set (instead of raising)
        # when continue_on_error is true and the user code raised. Treat that the
        # same as a host-level exception so the warning + replacement note fire once.
        if failure is None and result is not None and result.error:
            failure = result.error

        if failure is not None:
            _emit_warning(event_bus, f"Exec directive '{directive.name}' failed: {failure}")
            if directive.assign_var:
                exec_context[directive.assign_var] = None
            if directive.stdout_assign:
                exec_context[directive.stdout_assign] = result.stdout if result is not None else ""
            replacements.append(
                (directive.start_pos, directive.end_pos, f"<!-- Exec '{directive.name}' failed: {failure} -->")
            )
            continue

        assert result is not None
        if directive.assign_var:
            exec_context[directive.assign_var] = result.return_value
        if directive.stdout_assign:
            exec_context[directive.stdout_assign] = result.stdout
        replacements.append((directive.start_pos, directive.end_pos, _success_note(directive)))

    return _splice_replacements(content, replacements), exec_context
