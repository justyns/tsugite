"""Agent-facing terminal tools — stubs for the v2 `pty_run` / `pty_send_keys` /
`pty_inspect` surface described in the terminal-viewer design brief.

These shells exist so the design is grounded in real signatures, but the tools
are intentionally NOT registered in `tools/__init__.py` yet. The agent-facing
PTY tool surface is a v2 stretch goal; v1 only exposes the slash command +
HTTP routes that humans use directly from the web UI.

When you do register these, mirror `tools/jobs.py`'s pattern: a module-level
setter for the daemon-owned PtyManager + TerminalSessionStore, plus
`asyncio.run_coroutine_threadsafe` if the implementation needs to touch the
event loop. For now everything raises NotImplementedError.
"""

from __future__ import annotations

from typing import Optional

from . import tool

_pty_manager = None
_terminal_store = None


def set_terminal_runtime(pty_manager, terminal_store) -> None:
    """Wire the daemon-owned PtyManager + TerminalSessionStore into this module.

    Called from the gateway alongside the HTTPServer wiring. No-op when called
    with None (used in shutdown to drop the references)."""
    global _pty_manager, _terminal_store
    _pty_manager = pty_manager
    _terminal_store = terminal_store


@tool(require_daemon=True)
def pty_run(cmd: str, cwd: Optional[str] = None) -> dict:
    """Spawn a daemon-managed PTY running `cmd`. v2 stub.

    The web UI sees the terminal appear in the sidebar alongside chat sessions
    and can stream its output. The agent gets a handle (`id`) for follow-up
    `pty_send_keys` / `pty_inspect` calls.

    Args:
        cmd: Shell command line (passed through `sh -c`).
        cwd: Working directory for the spawned process. Defaults to daemon cwd.

    Returns:
        Dict with `id` (terminal session id) and `cmd` (the literal command).
    """
    raise NotImplementedError("pty_run is a v2 stretch goal — use the /run slash command or POST /api/terminals for v1")


@tool(require_daemon=True)
def pty_send_keys(id: str, keys: str) -> dict:
    """Write `keys` to the stdin of an existing PTY session. v2 stub.

    Args:
        id: TerminalSession id returned by `pty_run`.
        keys: Raw bytes (as a string) to write. Pass control chars literally
            (e.g. `"\\x03"` for Ctrl-C, `"q\\n"` to dismiss `less`).

    Returns:
        Dict with `bytes_written`.
    """
    raise NotImplementedError("pty_send_keys is a v2 stretch goal")


@tool(require_daemon=True)
def pty_inspect(id: str, tail: int = 200) -> dict:
    """Return the most recent `tail` lines of a PTY's output. v2 stub.

    Args:
        id: TerminalSession id.
        tail: Number of trailing lines of the ring buffer to return.

    Returns:
        Dict with `state`, `exit_code`, `bytes_out`, `truncated`, `output`
        (the trailing window, ANSI included).
    """
    raise NotImplementedError("pty_inspect is a v2 stretch goal")
