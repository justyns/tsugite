"""Session-scoped scratchpad tools for maintaining working state across compaction."""

from . import tool
from .sessions import get_current_session_id

SCRATCHPAD_SOFT_LIMIT = 4000
SCRATCHPAD_HARD_LIMIT = 8000

# Access session runner via module attribute (set by daemon at startup)
from . import sessions as _sessions_mod


def _get_session(session_id: str):
    return _sessions_mod._session_runner.store.get_session(session_id)


def _update_session(session_id: str, **fields):
    return _sessions_mod._session_runner.store.update_session(session_id, **fields)


def _emit_event(session_id: str):
    bus = getattr(_sessions_mod._session_runner, "_event_bus", None) or getattr(
        _sessions_mod._session_runner, "event_bus", None
    )
    if bus:
        bus.emit("session_update", {"action": "scratchpad_updated", "id": session_id})


@tool(require_daemon=True)
def scratchpad_read() -> str:
    """Read the session scratchpad buffer.

    Returns the current scratchpad content, or empty string if nothing has been written.

    Returns:
        The scratchpad content.
    """
    session_id = get_current_session_id()
    if not session_id:
        return ""
    try:
        return _get_session(session_id).scratchpad
    except ValueError:
        return ""


@tool(require_daemon=True)
def scratchpad_write(content: str) -> str:
    """Overwrite the session scratchpad buffer.

    Use this to maintain working state that survives context compaction:
    plans, progress, key findings, blockers, next steps.
    Keep it concise. Overwrite with trimmed content when items are resolved.

    Args:
        content: New scratchpad content (replaces existing).

    Returns:
        Confirmation message.
    """
    session_id = get_current_session_id()
    if not session_id:
        return "Error: no active session"
    if len(content) > SCRATCHPAD_HARD_LIMIT:
        return f"Error: content is {len(content)} chars, exceeds hard limit of {SCRATCHPAD_HARD_LIMIT}. Trim before writing."
    _update_session(session_id, scratchpad=content)
    _emit_event(session_id)
    msg = f"Scratchpad updated ({len(content)} chars)"
    if len(content) > SCRATCHPAD_SOFT_LIMIT:
        msg += f". Warning: buffer is {len(content)} chars (limit: {SCRATCHPAD_HARD_LIMIT}). Consider trimming stale content."
    return msg


@tool(require_daemon=True)
def scratchpad_append(content: str) -> str:
    """Append to the session scratchpad buffer.

    Adds content to the end of the scratchpad with a newline separator.

    Args:
        content: Content to append.

    Returns:
        Confirmation message.
    """
    session_id = get_current_session_id()
    if not session_id:
        return "Error: no active session"
    try:
        current = _get_session(session_id).scratchpad
    except ValueError:
        return "Error: session not found"
    new_content = f"{current}\n{content}" if current else content
    if len(new_content) > SCRATCHPAD_HARD_LIMIT:
        return f"Error: resulting content would be {len(new_content)} chars, exceeds hard limit of {SCRATCHPAD_HARD_LIMIT}. Use scratchpad_write() to replace with trimmed content."
    _update_session(session_id, scratchpad=new_content)
    _emit_event(session_id)
    msg = f"Appended to scratchpad ({len(new_content)} chars total)"
    if len(new_content) > SCRATCHPAD_SOFT_LIMIT:
        msg += f". Warning: buffer is {len(new_content)} chars (limit: {SCRATCHPAD_HARD_LIMIT}). Consider trimming stale content."
    return msg
