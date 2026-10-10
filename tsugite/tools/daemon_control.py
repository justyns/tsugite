"""Agent-requested daemon restart."""

import contextvars
import logging
import threading

from . import call_on_loop, deny_when_sandboxed, tool
from .sessions import get_current_session_id, session_reply

logger = logging.getLogger(__name__)

_controller = None
_loop = None
_approval_pending = threading.Lock()


def set_restart_controller(controller, loop):
    """Called by the daemon gateway to set/clear the restart controller."""
    global _controller, _loop
    _controller = controller
    _loop = loop


def _approval_detail(reason: str) -> str:
    """The reason plus the plugin files the restarted daemon would import."""
    from tsugite.plugins import local_plugin_files

    files = [f"- {ep.name}: {ep.value}" for ep in local_plugin_files()]
    if not files:
        return reason
    return reason + "\n\nSingle-file plugins that will load:\n" + "\n".join(files)


def _approve_then_restart(reason: str, origin_session: str | None) -> None:
    from tsugite.approval import request_approval

    outcome = None
    approved = False
    try:
        if request_approval("Restart the tsugite daemon?", detail=_approval_detail(reason)) == "approve":
            call_on_loop(_loop, _controller.request_restart)
            approved = True
        else:
            outcome = "The user denied the daemon restart."
            logger.warning("restart_daemon: %s", outcome)
    except Exception as e:
        logger.exception("restart_daemon approval flow failed")
        outcome = f"The daemon restart failed: {e}"
    finally:
        # After an approval the lock stays held until the daemon re-execs.
        if not approved:
            _approval_pending.release()
    if outcome and origin_session:
        try:
            session_reply(outcome, session_id=origin_session)
        except Exception:
            logger.exception("restart_daemon could not notify session %s", origin_session)


@tool(require_daemon=True)
@deny_when_sandboxed
def restart_daemon(reason: str) -> str:
    """Restart the daemon to pick up a new or edited single-file plugin, with the user's approval.

    A single-file plugin is imported once at startup and has no reload path, so a
    restart is the only way to pick up an edit.

    Refuses without asking when the restarted daemon would not come back (an
    unloadable daemon config, a plugin file that is missing or does not compile).
    Otherwise returns at once and asks the user in the background; a denial or a
    failed approval is sent back to this session as a message. Once approved the
    daemon stops accepting new HTTP chats, gives the in-flight turns up to two
    minutes to finish, then re-execs; running background jobs and terminal
    sessions do not survive it.

    Args:
        reason: Why the daemon should restart, shown to the user in the prompt.

    Returns:
        A message saying the approval was requested, is already pending, or why no restart was requested.
    """
    from tsugite.interaction import NonInteractiveBackend, get_interaction_backend

    if not _controller or not _loop:
        return "Restart is not available: no daemon is running this agent."

    # Reads config files off the loop thread; only request_restart is loop-affine.
    problems = _controller.preflight_restart()
    if problems:
        return "Restart refused. The daemon would not come back:\n" + "\n".join(f"- {p}" for p in problems)

    backend = get_interaction_backend()
    if backend is None or isinstance(backend, NonInteractiveBackend):
        return "Restart not approved."

    if not _approval_pending.acquire(blocking=False):
        return "A restart approval is already pending."

    ctx = contextvars.copy_context()
    try:
        threading.Thread(
            target=ctx.run, args=(_approve_then_restart, reason, get_current_session_id()), daemon=True
        ).start()
    except BaseException:
        _approval_pending.release()
        raise
    return "Asked the user to approve a restart."
