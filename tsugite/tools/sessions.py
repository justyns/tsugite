"""Session tools for agents to manage async agent sessions."""

from dataclasses import asdict
from typing import Optional

from . import call_on_loop, tool

_session_runner = None
_loop = None

CURRENT_SESSION = "current"


def set_session_runner(runner, loop=None):
    """Called by the daemon to set/clear the session runner reference."""
    global _session_runner, _loop
    _session_runner = runner
    _loop = loop


def _call(fn, *args, timeout=30, **kwargs):
    """Call a session runner method on the event loop thread (thread-safe)."""
    return call_on_loop(_loop, fn, *args, timeout=timeout, **kwargs)


def get_current_session_id():
    from tsugite_daemon.session_runner import get_current_session_id as _get

    return _get()


def _resolve_session_arg(session_id: Optional[str]) -> str:
    sid = session_id if session_id and session_id != CURRENT_SESSION else get_current_session_id()
    if not sid:
        raise ValueError("No current session - pass session_id explicitly")
    live = _call(_session_runner.store.resolve_live, sid)
    return live.id if live else sid


@tool(require_daemon=True)
def session_reply(message: str, session_id: Optional[str] = None) -> dict:
    """Send a follow-up message to an existing session, continuing its conversation.

    Args:
        message: Message to send to the session.
        session_id: Session to reply to. Defaults to the current session; "current" means the same.

    Returns:
        Dict with session_id and the agent's response.
    """
    session_id = _resolve_session_arg(session_id)
    result = _call(_session_runner.reply_to_session, session_id, message, timeout=120)
    return {"session_id": session_id, "response": str(result)[:2000]}


@tool(require_daemon=True)
def start_session(
    prompt: str,
    agent: Optional[str] = None,
    model: Optional[str] = None,
    agent_file: Optional[str] = None,
    session_id: Optional[str] = None,
    notify: Optional[list[str]] = None,
    notify_sessions: Optional[list[str]] = None,
) -> dict:
    """Start a new async agent session that runs in the background.

    IMPORTANT: Always confirm with the user before starting sessions.

    Args:
        prompt: Task instruction for the agent session.
        agent: Agent name configured in daemon. Defaults to the current agent.
        model: Optional model override.
        agent_file: Agent file name or path.
        session_id: Custom session ID. Auto-generated if not provided.
        notify: Notification channels for result delivery.
        notify_sessions: Session IDs to message when this session finishes.

    Returns:
        Session details including ID and status
    """
    from tsugite_daemon.session_store import Session, SessionSource

    if agent is None:
        from tsugite.agent_runner.helpers import resolve_run_agent

        agent = resolve_run_agent()

    # Inherit the sandbox: a sandboxed agent's spawned session must stay
    # sandboxed regardless of the target agent's own config.
    from tsugite.agent_runner.helpers import sandbox_context_to_override

    metadata = {}
    sandbox_override = sandbox_context_to_override()
    if sandbox_override is not None:
        metadata["sandbox_override"] = sandbox_override

    session = Session(
        id=session_id or "",
        agent=agent,
        source=SessionSource.BACKGROUND.value,
        prompt=prompt,
        model=model,
        agent_file=agent_file,
        notify=notify or [],
        notify_sessions=notify_sessions or [],
        metadata=metadata,
    )
    result = _call(_session_runner.start_session, session)
    return asdict(result)


@tool(require_daemon=True)
def session_notify(notify_session: str, session_id: Optional[str] = None) -> dict:
    """Ask a session to message another session when it finishes.

    Args:
        notify_session: Session that receives the completion message.
        session_id: Session whose completion triggers the notification. Defaults to the
            current session; "current" means the same.

    Returns:
        Dict with session_id and its current notify targets.
    """
    session_id = _resolve_session_arg(session_id)
    session = _call(_session_runner.add_notify_session, session_id, notify_session)
    return {"session_id": session.id, "notify_sessions": list(session.notify_sessions)}


@tool(require_daemon=True)
def session_acknowledge(delivery_id: Optional[str] = None, session_id: Optional[str] = None) -> dict:
    """Acknowledge a delivery the user has now dealt with, clearing its needs-you flag.

    Args:
        delivery_id: Delivery to acknowledge, from `<pending_deliveries>`. Omit to
            acknowledge every outstanding delivery on the session.
        session_id: Session holding the delivery. Defaults to the current session;
            "current" means the same.

    Returns:
        Dict with session_id, whether the session still needs the user, and the
        delivery ids still outstanding.
    """
    session_id = _resolve_session_arg(session_id)
    session = _call(_session_runner.clear_attention, session_id, delivery_id)
    return {
        "session_id": session.id,
        "needs_attention": session.needs_attention,
        "pending_deliveries": session.pending_delivery_ids,
    }


@tool(require_daemon=True)
def list_sessions(
    source: Optional[str] = None,
    status: Optional[str] = None,
    agent: Optional[str] = None,
    parent_id: Optional[str] = None,
) -> list:
    """List agent sessions with optional filters.

    Args:
        source: Filter by source type (interactive, schedule, webhook, background, spawned)
        status: Filter by status (active, running, completed, failed, etc.)
        agent: Filter by agent name
        parent_id: Filter by parent session/schedule ID

    Returns:
        List of sessions with id, agent, source, status, created_at
    """
    sessions = _call(
        _session_runner.store.list_sessions,
        agent=agent,
        source=source,
        status=status,
        parent_id=parent_id,
    )
    return [
        {
            "id": s.id,
            "agent": s.agent,
            "source": s.source,
            "status": s.status,
            "title": s.title,
            "prompt": (s.prompt or "")[:200],
            "created_at": s.created_at,
            "parent_id": s.parent_id,
        }
        for s in sessions
    ]


@tool(require_daemon=True)
def session_status(session_id: Optional[str] = None) -> dict:
    """Get detailed status of an agent session.

    Args:
        session_id: Session to check. Defaults to the current session; "current" means the same.

    Returns:
        Full session details
    """
    return _call(_session_runner.store.session_detail, _resolve_session_arg(session_id))


@tool(require_daemon=True)
def cancel_session(session_id: Optional[str] = None) -> dict:
    """Cancel a running agent session.

    Args:
        session_id: Session to cancel. Defaults to the current session; "current" means the same.

    Returns:
        Updated session details
    """
    session_id = _resolve_session_arg(session_id)
    _call(_session_runner.cancel_session, session_id)
    session = _call(_session_runner.store.get_session, session_id)
    return asdict(session)


@tool(require_daemon=True)
def rename_session(title: str, session_id: Optional[str] = None) -> dict:
    """Rename a session by setting its title.

    Args:
        title: New title for the session.
        session_id: Session to rename. Defaults to the current session; "current" means the same.

    Returns:
        Updated session details.
    """
    session = _call(_session_runner.rename_session, _resolve_session_arg(session_id), title)
    return {"session_id": session.id, "title": session.title}


@tool(require_daemon=True)
def session_metadata(key: str, value: Optional[str] = None, session_id: Optional[str] = None) -> dict:
    """Set, update, or delete a metadata key on a session.

    Args:
        key: Metadata key to set or delete.
        value: Value to set. Pass None to delete the key.
        session_id: Target session. Defaults to the current session; "current" means the same.

    Returns:
        Dict with session_id and updated metadata.
    """
    try:
        session_id = _resolve_session_arg(session_id)
        if value is None:
            session = _call(_session_runner.delete_session_metadata, session_id, key)
        else:
            session = _call(_session_runner.update_session_metadata, session_id, {key: value})
        return {"session_id": session_id, "metadata": session.metadata}
    except ValueError as e:
        return {"error": str(e)}
