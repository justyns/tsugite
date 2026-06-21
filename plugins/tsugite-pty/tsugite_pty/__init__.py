"""Tsugite plugin: PTY terminal runtime + tools (daemon-only).

`tsugite-daemon` depends on this package and wires the runtime via
`tools.set_terminal_runtime`. The `pty_*` tools register through the
`tsugite.plugins` entry point (`tsugite_pty.tools`) and degrade gracefully when no
runtime is wired (i.e. outside the daemon).
"""

from tsugite_pty.pty_manager import DEFAULT_BUFFER_CAP, PtyManager, PtyProcess
from tsugite_pty.terminal_runtime import set_session_sandbox_resolver, spawn_terminal
from tsugite_pty.terminal_store import (
    TerminalSession,
    TerminalSessionStore,
    TerminalState,
    TerminalStateTransitionError,
)

__all__ = [
    "DEFAULT_BUFFER_CAP",
    "PtyManager",
    "PtyProcess",
    "TerminalSession",
    "TerminalSessionStore",
    "TerminalState",
    "TerminalStateTransitionError",
    "set_session_sandbox_resolver",
    "spawn_terminal",
]
