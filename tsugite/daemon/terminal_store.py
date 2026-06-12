"""Persistent TerminalSession store backed by a JSON file.

Mirrors `WebhookStore` / `JobStore`: in-memory dict + atomic tmpfile-swap saves.
State transitions are guarded by `_VALID_TRANSITIONS` to prevent invalid moves.

PAUSED-FOLLOW from the design brief is intentionally NOT modeled here - it's a
frontend-only concern (the user scrolled up; output keeps streaming). The backend
state machine tracks only what the daemon needs to know to drive the PTY.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional
from uuid import uuid4

from tsugite.daemon.record_store import JsonRecordStore, now_iso

logger = logging.getLogger(__name__)


class TerminalState(str, Enum):
    STARTING = "starting"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    STREAM_LOST = "stream_lost"


_TERMINAL_STATES = frozenset(
    {
        TerminalState.SUCCEEDED.value,
        TerminalState.FAILED.value,
        TerminalState.CANCELLED.value,
        TerminalState.STREAM_LOST.value,
    }
)

# Unidirectional state machine. STARTING can transition to RUNNING (PTY produced
# its first byte) or directly to a failure mode (PTY spawn failed → FAILED;
# user killed before output → CANCELLED). RUNNING is the only state with all
# terminal exits available. Terminal states are sinks (no outgoing edges).
_VALID_TRANSITIONS: dict[str, frozenset[str]] = {
    TerminalState.STARTING.value: frozenset(
        {TerminalState.RUNNING.value, TerminalState.FAILED.value, TerminalState.CANCELLED.value}
    ),
    TerminalState.RUNNING.value: frozenset(
        {
            TerminalState.SUCCEEDED.value,
            TerminalState.FAILED.value,
            TerminalState.CANCELLED.value,
            TerminalState.STREAM_LOST.value,
        }
    ),
    TerminalState.SUCCEEDED.value: frozenset(),
    TerminalState.FAILED.value: frozenset(),
    TerminalState.CANCELLED.value: frozenset(),
    TerminalState.STREAM_LOST.value: frozenset(),
}


class TerminalStateTransitionError(ValueError):
    """Raised when a TerminalSession state change violates the state machine."""


@dataclass
class TerminalSession:
    id: str
    cmd: str
    cwd: Optional[str] = None
    state: str = TerminalState.STARTING.value
    pid: Optional[int] = None
    exit_code: Optional[int] = None
    created_at: str = ""
    updated_at: str = ""
    resolved_at: Optional[str] = None
    bytes_out: int = 0
    lines_out: int = 0
    last_line: str = ""
    # The chat session that spawned this terminal via /run, if any. Lets the UI
    # render the terminal's sidebar row underneath / alongside its parent chat.
    parent_session_id: Optional[str] = None

    def __post_init__(self):
        if not self.id:
            self.id = f"term-{uuid4().hex[:8]}"
        now = now_iso()
        if not self.created_at:
            self.created_at = now
        if not self.updated_at:
            self.updated_at = now


class TerminalSessionStore(JsonRecordStore):
    """JSON-backed persistent store for TerminalSession records."""

    record_cls = TerminalSession
    collection_key = "terminals"
    record_label = "terminal"
    valid_transitions = _VALID_TRANSITIONS
    terminal_states = _TERMINAL_STATES
    transition_error_cls = TerminalStateTransitionError

    def log_path(self, terminal_id: str) -> Path:
        """Path to the on-disk output log for a terminal.

        The log is written once when the PTY exits (by `terminal_runtime`) so
        the SSE stream can replay output for the client even after the
        in-memory PtyProcess has been evicted. The file may not exist if the
        PTY hasn't exited yet or produced no output; callers check `.exists()`.
        """
        return self._path.parent / "terminal_logs" / f"{terminal_id}.log"

    def _after_load(self) -> bool:
        """Records persisted as starting/running belong to a dead daemon process -
        the fresh PtyManager has no proc for them, so kill would no-op and restart
        would 409 forever. Resolve them as stream_lost."""
        reconciled = 0
        for terminal in self._records.values():
            if terminal.state not in _TERMINAL_STATES:
                terminal.state = TerminalState.STREAM_LOST.value
                terminal.updated_at = now_iso()
                if not terminal.resolved_at:
                    terminal.resolved_at = terminal.updated_at
                reconciled += 1
        if reconciled:
            logger.info("Marked %d stale terminal(s) from previous daemon run as stream_lost", reconciled)
        return reconciled > 0
