"""Glue between `pty_manager` (runtime) and `terminal_store` (persistence).

Owns the lifecycle hook that translates PTY exit codes into TerminalState
transitions and persists final byte counts. Kept out of both pty_manager and
terminal_store so neither has to know about the other.
"""

from __future__ import annotations

import logging
from typing import Optional

from tsugite.daemon.pty_manager import DEFAULT_BUFFER_CAP, PtyManager, PtyProcess
from tsugite.daemon.terminal_store import (
    TerminalSession,
    TerminalSessionStore,
    TerminalState,
    TerminalStateTransitionError,
)

logger = logging.getLogger(__name__)


def parse_command(cmd: str) -> list[str]:
    """Split a user-typed `/run` command into argv. Wrapped in a sh -c so users
    can use pipes/redirection/`&&` without us re-implementing shell semantics."""
    cmd = cmd.strip()
    if not cmd:
        raise ValueError("Command cannot be empty")
    # Always shell out so things like `ls | grep foo` work. The PTY hands the
    # shell stdin/stdout; we don't need to be the parser.
    return ["/bin/sh", "-c", cmd]


def spawn_terminal(
    *,
    store: TerminalSessionStore,
    manager: PtyManager,
    cmd: str,
    cwd: Optional[str] = None,
    env: Optional[dict] = None,
    parent_session_id: Optional[str] = None,
    buffer_cap: int = DEFAULT_BUFFER_CAP,
    on_state_change=None,
) -> TerminalSession:
    """Create a TerminalSession record + spawn its PTY in one step.

    Wires the PTY's on_exit hook to drive state transitions so callers don't
    have to. Returns the persisted TerminalSession with state=RUNNING (the
    first chunk of output flips STARTING -> RUNNING immediately on spawn).

    on_state_change: optional callback(terminal_id, new_state) for broadcasting
    state changes to SSE subscribers. Caller is responsible for thread-safe
    dispatch (we may call it from the reader thread).
    """
    session = store.add(
        TerminalSession(
            id="",
            cmd=cmd,
            cwd=cwd,
            parent_session_id=parent_session_id,
        )
    )

    try:
        argv = parse_command(cmd)
        proc = manager.spawn(session.id, argv, cwd=cwd, env=env, buffer_cap=buffer_cap)
    except Exception as e:
        logger.exception("Failed to spawn PTY for terminal '%s': %s", session.id, e)
        try:
            store.update_state(session.id, TerminalState.FAILED.value)
            store.update(session.id, last_line=f"spawn failed: {e}")
        except TerminalStateTransitionError:
            pass
        if on_state_change:
            try:
                on_state_change(session.id, TerminalState.FAILED.value)
            except Exception:
                logger.exception("on_state_change failed during spawn-failure path")
        raise

    store.update(session.id, pid=proc.pid)
    try:
        store.update_state(session.id, TerminalState.RUNNING.value)
    except TerminalStateTransitionError:
        pass
    if on_state_change:
        try:
            on_state_change(session.id, TerminalState.RUNNING.value)
        except Exception:
            logger.exception("on_state_change failed during RUNNING transition")

    proc.on_exit(lambda p: _on_pty_exit(p, store, session.id, on_state_change))
    return store.get(session.id)


def _on_pty_exit(
    proc: PtyProcess,
    store: TerminalSessionStore,
    terminal_id: str,
    on_state_change=None,
) -> None:
    """Translate the PTY's exit code into a TerminalState transition.

    - kill() was called → CANCELLED
    - exit_code == 0 → SUCCEEDED
    - exit_code != 0 → FAILED
    - exit_code is None (shouldn't happen, but defensive) → STREAM_LOST
    """
    terminal = store.get(terminal_id)
    if terminal is None:
        logger.warning("PTY exit hook fired for unknown terminal '%s'", terminal_id)
        return

    try:
        store.update(
            terminal_id,
            exit_code=proc.exit_code,
            bytes_out=proc.bytes_out,
            lines_out=proc.lines_out,
            last_line=proc.last_line,
        )
    except KeyError:
        return

    if proc.killed:
        target = TerminalState.CANCELLED.value
    elif proc.exit_code is None:
        target = TerminalState.STREAM_LOST.value
    elif proc.exit_code == 0:
        target = TerminalState.SUCCEEDED.value
    else:
        target = TerminalState.FAILED.value

    try:
        store.update_state(terminal_id, target)
    except TerminalStateTransitionError:
        # Already terminal — e.g. the caller manually transitioned us first.
        pass

    if on_state_change:
        try:
            on_state_change(terminal_id, target)
        except Exception:
            logger.exception("on_state_change failed in PTY exit handler")


def safe_cmd_for_label(cmd: str, max_len: int = 80) -> str:
    """Trim a command for display in lightweight contexts (sidebar row, etc.)."""
    cmd = cmd.strip()
    if len(cmd) <= max_len:
        return cmd
    return cmd[: max_len - 1] + "…"


__all__ = ["parse_command", "spawn_terminal", "safe_cmd_for_label"]
