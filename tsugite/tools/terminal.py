"""Agent-facing PTY tools that wrap the daemon's PtyManager + TerminalSessionStore.

These tools let an agent spawn interactive CLIs (Claude Code, ssh, psql, REPLs)
and drive them via stdin keystrokes + ring-buffer captures. PTYs appear in the
web UI's terminal sidebar and stream live via SSE.

Wiring: the daemon gateway calls `set_terminal_runtime(pty_manager, terminal_store)`
once the runtime is up. Tools resolve the manager + store via module-level refs;
they raise a friendly error dict when called outside daemon mode.

Lifetime: every PTY is session-scoped (parent = current daemon session).
Long-lived daemon-scope PTYs are deferred for v1.
"""

from __future__ import annotations

import logging
import signal as _signal
from typing import Optional

from . import tool

logger = logging.getLogger(__name__)

_pty_manager = None
_terminal_store = None
_state_change_callback = None

_SIGNAL_MAP = {
    "TERM": _signal.SIGTERM,
    "KILL": _signal.SIGKILL,
    "INT": _signal.SIGINT,
}


def set_terminal_runtime(pty_manager, terminal_store, state_change_callback=None) -> None:
    """Wire the daemon-owned PtyManager + TerminalSessionStore into this module.

    Called from the gateway alongside the HTTPServer wiring. No-op when called
    with None (used in shutdown to drop the references).

    Args:
        pty_manager: The daemon's PtyManager instance.
        terminal_store: The daemon's TerminalSessionStore instance.
        state_change_callback: Optional callable(terminal_id, new_state) used by
            spawn_terminal to broadcast PTY lifecycle transitions to the SSE feed.
    """
    global _pty_manager, _terminal_store, _state_change_callback
    _pty_manager = pty_manager
    _terminal_store = terminal_store
    _state_change_callback = state_change_callback


def runtime_available() -> bool:
    """True when the daemon wired a PtyManager + TerminalSessionStore in here.
    Also consulted by the adapters to decide whether to render PTY guidance."""
    return _pty_manager is not None and _terminal_store is not None


def _missing_runtime() -> dict:
    return {"error": "PTY runtime not available (not running in daemon mode)"}


def _get_terminal(terminal_id: str):
    """Return the TerminalSession or an error dict (caller forwards it)."""
    terminal = _terminal_store.get(terminal_id)
    if terminal is None:
        return None, {"error": f"Unknown terminal: {terminal_id}"}
    return terminal, None


@tool(require_daemon=True, category="terminal")
def pty_create(
    cmd: str,
    cwd: Optional[str] = None,
    env: Optional[dict] = None,
    name: Optional[str] = None,
    cols: int = 120,
    rows: int = 40,
) -> dict:
    """Spawn a PTY-backed process. Returns {terminal_id, pid, started_at, cmd}.

    The PTY appears in the web UI's terminal sidebar and streams live via SSE.
    Output is captured into a ring buffer for later `pty_capture` reads.

    Use this for *interactive* programs (ssh, psql, claude, python REPL, vim).
    For one-shot commands that exit on their own, prefer `run()`.

    Args:
        cmd: Shell command line (passed through `sh -c`).
        cwd: Working directory. Defaults to the daemon's cwd.
        env: Extra environment variables merged on top of the daemon env.
        name: Optional human-readable label (currently unused; reserved for v2).
        cols: PTY column width (informational; xterm.js does not reflow today).
        rows: PTY row height (informational).

    Returns:
        Dict with terminal_id, pid, started_at, cmd.
    """
    if not runtime_available():
        return _missing_runtime()

    from tsugite.daemon.session_runner import get_current_session_id
    from tsugite.daemon.terminal_runtime import spawn_terminal

    parent_session_id = get_current_session_id()

    try:
        session = spawn_terminal(
            store=_terminal_store,
            manager=_pty_manager,
            cmd=cmd,
            cwd=cwd,
            env=env,
            parent_session_id=parent_session_id,
            on_state_change=_state_change_callback,
        )
    except Exception as e:
        logger.exception("pty_create failed for cmd=%r", cmd)
        return {"error": f"Failed to spawn PTY: {e}"}

    return {
        "terminal_id": session.id,
        "pid": session.pid,
        "started_at": session.created_at,
        "cmd": session.cmd,
    }


@tool(require_daemon=True, category="terminal")
def pty_send_keys(terminal_id: str, keys: str, enter: bool = True) -> dict:
    """Write keystrokes to the PTY's stdin. `enter=True` appends \\n.

    For escape sequences (Ctrl+C, arrow keys), pass raw escape bytes:
    Ctrl+C = "\\x03", up arrow = "\\x1b[A", Esc = "\\x1b". Pair these with
    `enter=False` so we don't tack a newline on the end.

    Args:
        terminal_id: TerminalSession id returned by `pty_create`.
        keys: Literal characters / control bytes to send.
        enter: When True (default), append a newline.

    Returns:
        Dict with terminal_id and bytes_written, or {error: ...}.
    """
    if not runtime_available():
        return _missing_runtime()

    terminal, err = _get_terminal(terminal_id)
    if err:
        return err

    payload = keys
    if enter:
        payload = payload + "\n"

    try:
        data = payload.encode("utf-8")
    except UnicodeEncodeError as e:
        return {"error": f"Failed to encode keys: {e}"}

    written = _pty_manager.write_stdin(terminal_id, data)
    return {"terminal_id": terminal_id, "bytes_written": written}


@tool(require_daemon=True, category="terminal")
def pty_capture(terminal_id: str, lines: int = 50, tail: bool = True) -> dict:
    """Read the current PTY output buffer.

    Use this to confirm a command finished or to read a prompt before responding.
    `tail=True` returns the last N lines; `tail=False` returns the first N.

    Args:
        terminal_id: TerminalSession id.
        lines: Number of lines to return from the buffer.
        tail: When True (default), return the last `lines`. Else return the first.

    Returns:
        Dict with text, bytes_out, lines_out, truncated, state, exit_code.
    """
    if not runtime_available():
        return _missing_runtime()

    terminal, err = _get_terminal(terminal_id)
    if err:
        return err

    proc = _pty_manager.get(terminal_id)
    if proc is not None:
        raw = proc.buffer
        bytes_out = proc.bytes_out
        lines_out = proc.lines_out
        truncated = proc.truncated
    else:
        # PTY has exited and been evicted from the manager. The exit hook
        # persisted the full ring buffer to disk - replay that. Fall back to
        # the one-line last_line snapshot only if the log never got written.
        bytes_out = terminal.bytes_out
        lines_out = terminal.lines_out
        truncated = False
        log_path = _terminal_store.log_path(terminal_id)
        try:
            raw = log_path.read_bytes()
        except OSError:
            raw = (terminal.last_line or "").encode("utf-8", errors="replace")

    text = raw.decode("utf-8", errors="replace")
    split = text.splitlines()
    if tail:
        selected = split[-lines:] if lines > 0 else []
    else:
        selected = split[:lines] if lines > 0 else []
    rendered = "\n".join(selected)

    return {
        "terminal_id": terminal_id,
        "text": rendered,
        "bytes_out": bytes_out,
        "lines_out": lines_out,
        "truncated": truncated,
        "state": terminal.state,
        "exit_code": terminal.exit_code,
    }


@tool(require_daemon=True, category="terminal")
def pty_kill(terminal_id: str, signal: str = "TERM") -> dict:
    """Send a signal to the PTY. Returns {state, exit_code}.

    - TERM (default): SIGTERM to the PTY's process group. Transitions the
      TerminalSession to CANCELLED once the process actually exits.
    - KILL: SIGKILL, the escalation if TERM doesn't take.
    - INT: SIGINT (Ctrl+C), for graceful cancel of a running command without
      killing the shell itself.

    Args:
        terminal_id: TerminalSession id.
        signal: One of "TERM", "KILL", "INT" (case-insensitive).

    Returns:
        Dict with terminal_id, state, exit_code, signal.
    """
    if not runtime_available():
        return _missing_runtime()

    terminal, err = _get_terminal(terminal_id)
    if err:
        return err

    sig_name = (signal or "TERM").upper()
    sig = _SIGNAL_MAP.get(sig_name)
    if sig is None:
        return {"error": f"Unsupported signal '{signal}'. Use TERM, KILL, or INT."}

    proc = _pty_manager.get(terminal_id)
    if proc is None:
        # Already exited and reaped.
        return {
            "terminal_id": terminal_id,
            "state": terminal.state,
            "exit_code": terminal.exit_code,
            "signal": sig_name,
        }

    if sig_name == "INT":
        # SIGINT goes through stdin as the conventional Ctrl+C byte so the
        # *foreground* program in the PTY catches it, not the shell parent.
        _pty_manager.write_stdin(terminal_id, b"\x03")
    else:
        # TERM / KILL hit the process group via PtyManager.kill; double-calls
        # auto-escalate per PtyProcess.kill's grace logic.
        _pty_manager.kill(terminal_id)
        if sig_name == "KILL":
            # Force the escalation by calling again immediately.
            _pty_manager.kill(terminal_id)

    # State may not have updated yet (kill is async - the reader thread sees
    # EIO before exit_code populates). Re-read for the snapshot we return.
    terminal = _terminal_store.get(terminal_id)
    return {
        "terminal_id": terminal_id,
        "state": terminal.state if terminal else "unknown",
        "exit_code": terminal.exit_code if terminal else None,
        "signal": sig_name,
    }


@tool(require_daemon=True, category="terminal")
def pty_list(state: Optional[str] = None) -> list[dict]:
    """List all terminals the daemon owns, optionally filtered by state.

    Args:
        state: Filter by TerminalState value (running, succeeded, failed,
            cancelled, stream_lost, starting).

    Returns:
        List of dicts with terminal_id, cmd, state, pid, created_at,
        resolved_at, exit_code, parent_session_id.
    """
    if not runtime_available():
        return [_missing_runtime()]

    terminals = _terminal_store.list_all()
    if state:
        terminals = [t for t in terminals if t.state == state]

    return [
        {
            "terminal_id": t.id,
            "cmd": t.cmd,
            "state": t.state,
            "pid": t.pid,
            "created_at": t.created_at,
            "resolved_at": t.resolved_at,
            "exit_code": t.exit_code,
            "parent_session_id": t.parent_session_id,
        }
        for t in terminals
    ]
