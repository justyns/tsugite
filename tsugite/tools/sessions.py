"""Session tools for agents to manage async agent sessions.

Tools use @tool(require_daemon=True) so they only appear in daemon mode.
"""

import asyncio
from dataclasses import asdict
from typing import Optional

from . import tool

_session_runner = None
_loop = None


def set_session_runner(runner, loop=None):
    """Called by the daemon to set/clear the session runner reference."""
    global _session_runner, _loop
    _session_runner = runner
    _loop = loop


def _call(fn, *args, timeout=30, **kwargs):
    """Call a session runner method on the event loop thread (thread-safe)."""

    async def _wrapper():
        result = fn(*args, **kwargs)
        if asyncio.iscoroutine(result):
            return await result
        return result

    future = asyncio.run_coroutine_threadsafe(_wrapper(), _loop)
    return future.result(timeout=timeout)


@tool(require_daemon=True)
def session_reply(session_id: str, message: str) -> dict:
    """Send a follow-up message to an existing session, continuing its conversation.

    Args:
        session_id: ID of the session to reply to.
        message: Message to send to the session.

    Returns:
        Dict with session_id and the agent's response.
    """
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

    Returns:
        Session details including ID and status
    """
    from tsugite.daemon.session_store import Session, SessionSource

    if agent is None:
        from tsugite.agent_runner.helpers import get_current_agent

        agent = get_current_agent() or "default"

    session = Session(
        id=session_id or "",
        agent=agent,
        source=SessionSource.BACKGROUND.value,
        prompt=prompt,
        model=model,
        agent_file=agent_file,
        notify=notify or [],
    )

    result = _call(_session_runner.start_session, session)
    return asdict(result)


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
def session_status(session_id: str) -> dict:
    """Get detailed status of an agent session.

    Args:
        session_id: Session ID to check

    Returns:
        Full session details
    """
    return _call(_session_runner.store.session_detail, session_id)


@tool(require_daemon=True)
def cancel_session(session_id: str) -> dict:
    """Cancel a running agent session.

    Args:
        session_id: Session ID to cancel

    Returns:
        Updated session details
    """
    _call(_session_runner.cancel_session, session_id)
    session = _call(_session_runner.store.get_session, session_id)
    return asdict(session)


@tool(require_daemon=True)
def rename_session(session_id: str, title: str) -> dict:
    """Rename a session by setting its title.

    Args:
        session_id: Session ID to rename.
        title: New title for the session.

    Returns:
        Updated session details.
    """
    session = _call(_session_runner.rename_session, session_id, title)
    return {"session_id": session.id, "title": session.title}


@tool(require_daemon=True)
def session_spawn(
    prompt: str,
    agent: Optional[str] = None,
    name: Optional[str] = None,
    parent_session_id: Optional[str] = None,
    notify: Optional[list[str]] = None,
) -> dict:
    """Spawn a new long-running session with parent linkage.

    Unlike start_session, spawned sessions are meant for long-running agent-managed
    workstreams that may outlive the current conversation.

    Args:
        prompt: Task instruction for the spawned session.
        agent: Agent name. Defaults to the current agent.
        name: Human-readable name for the session (used as session ID prefix).
        parent_session_id: Parent session ID for tracking lineage.
        notify: Notification channels for result delivery.

    Returns:
        Session details including ID and status
    """
    from tsugite.daemon.session_store import Session, SessionSource

    if agent is None:
        from tsugite.agent_runner.helpers import get_current_agent

        agent = get_current_agent() or "default"

    session_id = f"spawn-{name}" if name else ""

    session = Session(
        id=session_id,
        agent=agent,
        source=SessionSource.SPAWNED.value,
        prompt=prompt,
        parent_id=parent_session_id,
        notify=notify or [],
    )

    result = _call(_session_runner.start_session, session)
    return asdict(result)
