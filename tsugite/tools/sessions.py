"""Session tools for agents to manage async agent sessions.

Tools use @tool(require_daemon=True) so they only appear in daemon mode.
"""

import asyncio
from dataclasses import asdict
from typing import Optional
from uuid import uuid4

from . import tool

_session_runner = None
_loop = None


def set_session_runner(runner, loop=None):
    """Called by the daemon to set/clear the session runner reference."""
    global _session_runner, _loop
    _session_runner = runner
    _loop = loop


def _call(fn, *args, **kwargs):
    """Call a session runner method on the event loop thread (thread-safe)."""

    async def _wrapper():
        return fn(*args, **kwargs)

    future = asyncio.run_coroutine_threadsafe(_wrapper(), _loop)
    return future.result(timeout=30)


@tool(require_daemon=True)
def start_session(
    prompt: str,
    agent: Optional[str] = None,
    model: Optional[str] = None,
    agent_file: Optional[str] = None,
    session_id: Optional[str] = None,
    notify: Optional[list[str]] = None,
    sandbox: bool = False,
    allow_domains: Optional[list[str]] = None,
    no_network: bool = False,
) -> dict:
    """Start a new async agent session that runs in the background.

    Sessions support review gates — if the agent calls create_review during execution,
    the session pauses until a human approves or declines via the web UI or API.

    IMPORTANT: Always confirm with the user before starting sessions.

    Args:
        prompt: Task instruction for the agent session.
        agent: Agent name configured in daemon. Defaults to the current agent.
        model: Optional model override.
        agent_file: Agent file name or path.
        session_id: Custom session ID. Auto-generated if not provided.
        notify: Notification channels for result delivery.
        sandbox: Run in subprocess sandbox.
        allow_domains: Allowed network domains (sandbox only).
        no_network: Disable network entirely (sandbox only).

    Returns:
        Session details including ID and state
    """
    from tsugite.daemon.agent_session import AgentSession

    if agent is None:
        from tsugite.agent_runner.helpers import get_current_agent

        agent = get_current_agent() or "default"

    sid = session_id or f"session-{uuid4().hex[:8]}"
    session = AgentSession(
        id=sid,
        agent=agent,
        prompt=prompt,
        model=model,
        agent_file=agent_file,
        notify=notify or [],
        sandbox=sandbox,
        allow_domains=allow_domains or [],
        no_network=no_network,
    )

    result = _call(_session_runner.start_session, session)
    return asdict(result)


@tool(require_daemon=True)
def list_sessions() -> list:
    """List all agent sessions with their current status.

    Returns:
        List of sessions with id, agent, state, created_at
    """
    sessions = _call(_session_runner.store.list_sessions)
    return [
        {
            "id": s.id,
            "agent": s.agent,
            "state": s.state,
            "prompt": s.prompt[:200],
            "created_at": s.created_at,
        }
        for s in sessions
    ]


@tool(require_daemon=True)
def session_status(session_id: str) -> dict:
    """Get detailed status of an agent session.

    Args:
        session_id: Session ID to check

    Returns:
        Full session details including pending review if any
    """
    session = _call(_session_runner.store.get_session, session_id)
    result = asdict(session)
    result["event_count"] = _call(_session_runner.store.event_count, session_id)

    if session.current_review_id:
        try:
            review = _call(_session_runner.store.get_review, session.current_review_id)
            result["pending_review"] = asdict(review)
        except ValueError:
            pass

    return result


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
def create_review(title: str, description: str = "", context: Optional[dict] = None) -> dict:
    """Create a review gate that pauses execution until a human approves or declines.

    This tool blocks until the review is resolved via the web UI or API.
    Use this for operations that need human approval (destructive actions,
    deployments, infrastructure changes, etc.).

    Args:
        title: Short description of what needs approval
        description: Detailed context for the reviewer
        context: Additional structured data for the reviewer

    Returns:
        Dict with decision ('approved' or 'declined') and reviewer comment
    """
    import threading

    from tsugite.daemon.agent_session import ReviewDecision, ReviewGate
    from tsugite.daemon.session_runner import SessionReviewBackend, get_current_session_id

    session_id = get_current_session_id()
    if not session_id:
        raise RuntimeError("create_review can only be used within an agent session")

    backend = _session_runner._review_backends.get(session_id)
    if not backend:
        raise RuntimeError("No review backend for this session")

    review = ReviewGate(id="", session_id=session_id, title=title, description=description, context=context or {})
    review = _call(_session_runner.store.create_review, review)

    event = threading.Event()
    backend._review_events[review.id] = event

    if not event.wait(timeout=SessionReviewBackend.TIMEOUT):
        _call(_session_runner.store.resolve_review, review.id, ReviewDecision.DECLINED.value, "Timed out")
        return {"decision": ReviewDecision.DECLINED.value, "comment": "Timed out waiting for review"}

    resolved = _call(_session_runner.store.get_review, review.id)
    backend._review_events.pop(review.id, None)
    return {"decision": resolved.decision, "comment": resolved.reviewer_comment}
