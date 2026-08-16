"""Agent-requested daemon restart."""

from . import call_on_loop, deny_when_sandboxed, tool

_controller = None
_loop = None


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


@tool(require_daemon=True)
@deny_when_sandboxed
def restart_daemon(reason: str) -> str:
    """Restart the daemon to pick up a new or edited single-file plugin, with the user's approval.

    A single-file plugin is imported once at startup and has no reload path, so a
    restart is the only way to pick up an edit.

    Asks the user to approve first, and refuses without asking when the restarted
    daemon would not come back (an unloadable daemon config, a plugin file that is
    missing or does not compile). Once approved the daemon stops accepting new HTTP
    chats, gives the in-flight turns up to two minutes to finish, then re-execs;
    running background jobs and terminal sessions do not survive it.

    Args:
        reason: Why the daemon should restart, shown to the user in the prompt.

    Returns:
        Approved, not approved, or the reason no restart was requested.
    """
    from tsugite.approval import request_approval

    if not _controller or not _loop:
        return "Restart is not available: no daemon is running this agent."

    # Reads config files off the loop thread; only request_restart is loop-affine.
    problems = _controller.preflight_restart()
    if problems:
        return "Restart refused. The daemon would not come back:\n" + "\n".join(f"- {p}" for p in problems)

    if request_approval("Restart the tsugite daemon?", detail=_approval_detail(reason)) != "approve":
        return "Restart not approved."

    call_on_loop(_loop, _controller.request_restart)
    return "Approved. The daemon restarts once this turn finishes."
