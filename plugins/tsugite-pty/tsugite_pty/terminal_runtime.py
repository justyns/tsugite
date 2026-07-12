"""Glue between `pty_manager` (runtime) and `terminal_store` (persistence).

Owns the lifecycle hook that translates PTY exit codes into TerminalState
transitions and persists final byte counts. Kept out of both pty_manager and
terminal_store so neither has to know about the other.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Optional

from tsugite_pty.pty_manager import DEFAULT_BUFFER_CAP, PtyManager, PtyProcess
from tsugite_pty.terminal_store import (
    TerminalSession,
    TerminalSessionStore,
    TerminalState,
    TerminalStateTransitionError,
)

logger = logging.getLogger(__name__)

# Grace before an exited PTY is evicted from the manager, so a late SSE
# reconnect can still replay the final buffer before the 1 MB record is freed.
EVICT_GRACE_SECONDS = 30.0

# Sentinel distinguishing "sandbox_ctx not passed" (resolve the agent-inherited
# policy, forcing no-network) from an explicit None (run unsandboxed).
_UNSET = object()


def parse_command(cmd: str) -> list[str]:
    """Split a user-typed `/run` command into argv. Wrapped in a sh -c so users
    can use pipes/redirection/`&&` without us re-implementing shell semantics."""
    cmd = cmd.strip()
    if not cmd:
        raise ValueError("Command cannot be empty")
    # Always shell out so things like `ls | grep foo` work. The PTY hands the
    # shell stdin/stdout; we don't need to be the parser.
    return ["/bin/sh", "-c", cmd]


# Resolver wired by the gateway: session_id -> Optional[SandboxContext], so a
# terminal opened outside an agent turn (the /run command, the HTTP API) still
# inherits its parent session's agent sandbox config.
_session_sandbox_resolver = None


def set_session_sandbox_resolver(fn) -> None:
    """Wire the session -> sandbox-policy resolver (called from the gateway)."""
    global _session_sandbox_resolver
    _session_sandbox_resolver = fn


def resolve_terminal_sandbox(parent_session_id: Optional[str]):
    """Sandbox policy for a terminal: the running agent's thread-local context if
    present (agent-turn pty_create), else the parent session's agent config
    (terminals opened via /run or the API). None when nothing is sandboxed."""
    from tsugite.agent_runner import get_sandbox_context

    ctx = get_sandbox_context()
    if ctx is not None:
        return ctx
    if _session_sandbox_resolver is not None and parent_session_id:
        return _session_sandbox_resolver(parent_session_id)
    return None


def maybe_sandbox_argv(
    argv: list[str], cwd: Optional[str], sandbox_ctx=None, force_no_network: bool = True
) -> list[str]:
    """Wrap a PTY command in bwrap when its agent runs sandboxed.

    PTYs run in the daemon (parent) process, so without this a sandboxed agent
    could use pty_create to execute outside the sandbox. Agent-inherited sandboxed
    PTYs are filesystem-isolated to the workspace and get no network (no filtering
    proxy is wired for the long-lived PTY path) - the agent's own code/shell still
    reach the network through the executor's filtered proxy.

    force_no_network=True (the agent-inherited default) forces no-network - a
    sandboxed agent's PTY must never gain network. A caller passing an explicit
    sandbox_ctx sets force_no_network=False so the context's own no_network is
    honored (e.g. filesystem isolation with network on).

    Returns argv unchanged when sandbox_ctx is None (not sandboxed). Fails closed
    if a policy is active but no workspace dir is known.
    """
    if sandbox_ctx is None:
        return argv

    from pathlib import Path

    from tsugite.core.sandbox import SandboxConfig, get_sandbox_class

    workspace_dir = sandbox_ctx.workspace_dir or (Path(cwd) if cwd else None)
    if workspace_dir is None:
        raise RuntimeError("Cannot sandbox PTY: no workspace directory in the active sandbox policy")

    sandbox_cls = get_sandbox_class()
    if sandbox_cls is None:
        raise RuntimeError("Cannot sandbox PTY: no sandbox backend installed (pip install tsugite-sandbox)")

    sandbox = sandbox_cls(
        config=SandboxConfig(
            no_network=force_no_network or sandbox_ctx.no_network,
            extra_ro_binds=list(sandbox_ctx.extra_ro_binds),
            extra_rw_binds=list(sandbox_ctx.extra_rw_binds),
        ),
        workspace_dir=Path(workspace_dir),
        state_dir=None,
    )
    return sandbox.build_command(argv)


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
    sandbox_ctx=_UNSET,
) -> TerminalSession:
    """Create a TerminalSession record + spawn its PTY in one step.

    Wires the PTY's on_exit hook to drive state transitions so callers don't
    have to. Returns the persisted TerminalSession with state=RUNNING (the
    first chunk of output flips STARTING -> RUNNING immediately on spawn).

    on_state_change: optional callback(terminal_id, new_state) for broadcasting
    state changes to SSE subscribers. Caller is responsible for thread-safe
    dispatch (we may call it from the reader thread).

    sandbox_ctx: leave unset to resolve the agent-inherited policy (which forces
    no-network). Pass explicitly to override - a SandboxContext honors its own
    no_network (filesystem isolation with network on), or None runs unsandboxed.
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
        if sandbox_ctx is _UNSET:
            argv = maybe_sandbox_argv(argv, cwd, resolve_terminal_sandbox(parent_session_id))
        else:
            argv = maybe_sandbox_argv(argv, cwd, sandbox_ctx, force_no_network=False)
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

    proc.on_exit(lambda p: _on_pty_exit(p, store, manager, session.id, on_state_change))
    return store.get(session.id)


def _on_pty_exit(
    proc: PtyProcess,
    store: TerminalSessionStore,
    manager: PtyManager,
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
        # Already terminal - e.g. the caller manually transitioned us first.
        pass

    if on_state_change:
        try:
            on_state_change(terminal_id, target)
        except Exception:
            logger.exception("on_state_change failed in PTY exit handler")

    # Persist the captured output to disk so the SSE stream handler can replay
    # it for clients that re-open the terminal after the in-memory PtyProcess
    # gets evicted. Skip empty buffers (no point writing 0 bytes).
    try:
        buf = proc.buffer
        if buf:
            log_path = store.log_path(terminal_id)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = log_path.with_suffix(".tmp")
            tmp.write_bytes(buf)
            os.replace(str(tmp), str(log_path))
    except Exception:
        logger.exception("Failed to persist terminal log for '%s'", terminal_id)

    # The final record is already persisted above; free the in-memory PtyProcess
    # (which pins a ~1 MB ring buffer) after a grace window so late SSE reconnects
    # can still hit the in-memory replay path before we drop the buffer.
    timer = threading.Timer(EVICT_GRACE_SECONDS, manager.remove, args=(terminal_id,))
    timer.daemon = True
    timer.start()


__all__ = ["parse_command", "spawn_terminal"]
