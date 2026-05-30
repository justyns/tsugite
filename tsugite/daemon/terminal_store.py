"""Persistent TerminalSession store backed by a JSON file.

Mirrors `WebhookStore` / `JobStore`: in-memory dict + atomic tmpfile-swap saves.
State transitions are guarded by `_VALID_TRANSITIONS` to prevent invalid moves.

PAUSED-FOLLOW from the design brief is intentionally NOT modeled here — it's a
frontend-only concern (the user scrolled up; output keeps streaming). The backend
state machine tracks only what the daemon needs to know to drive the PTY.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional
from uuid import uuid4

logger = logging.getLogger(__name__)


class TerminalState(str, Enum):
    STARTING = "starting"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"
    STREAM_LOST = "stream_lost"


_TERMINAL_STATES = frozenset(
    {
        TerminalState.SUCCEEDED.value,
        TerminalState.FAILED.value,
        TerminalState.CANCELLED.value,
        TerminalState.TIMED_OUT.value,
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
            TerminalState.TIMED_OUT.value,
            TerminalState.STREAM_LOST.value,
        }
    ),
    TerminalState.SUCCEEDED.value: frozenset(),
    TerminalState.FAILED.value: frozenset(),
    TerminalState.CANCELLED.value: frozenset(),
    TerminalState.TIMED_OUT.value: frozenset(),
    TerminalState.STREAM_LOST.value: frozenset(),
}


class TerminalStateTransitionError(ValueError):
    """Raised when a TerminalSession state change violates the state machine."""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


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
        now = _now_iso()
        if not self.created_at:
            self.created_at = now
        if not self.updated_at:
            self.updated_at = now


class TerminalSessionStore:
    """JSON-backed persistent store for TerminalSession records."""

    def __init__(self, path: Path):
        self._path = path
        self._terminals: dict[str, TerminalSession] = {}
        self._lock = threading.Lock()
        self._load()

    def add(self, terminal: TerminalSession) -> TerminalSession:
        with self._lock:
            if terminal.id in self._terminals:
                raise ValueError(f"TerminalSession already exists: {terminal.id}")
            self._terminals[terminal.id] = terminal
            self._save()
        return terminal

    def get(self, terminal_id: str) -> Optional[TerminalSession]:
        return self._terminals.get(terminal_id)

    def list_all(self) -> list[TerminalSession]:
        return list(self._terminals.values())

    def list_active(self) -> list[TerminalSession]:
        return [t for t in self._terminals.values() if t.state not in _TERMINAL_STATES]

    def list_for_parent(self, parent_session_id: str) -> list[TerminalSession]:
        return [t for t in self._terminals.values() if t.parent_session_id == parent_session_id]

    def update_state(self, terminal_id: str, new_state: str) -> TerminalSession:
        with self._lock:
            terminal = self._terminals.get(terminal_id)
            if terminal is None:
                raise KeyError(f"Unknown terminal: {terminal_id}")
            allowed = _VALID_TRANSITIONS.get(terminal.state, frozenset())
            if new_state not in allowed:
                raise TerminalStateTransitionError(
                    f"Invalid TerminalSession state transition: {terminal.state} -> {new_state} (id {terminal_id})"
                )
            terminal.state = new_state
            terminal.updated_at = _now_iso()
            if new_state in _TERMINAL_STATES and not terminal.resolved_at:
                terminal.resolved_at = terminal.updated_at
            self._save()
            return terminal

    def update(self, terminal_id: str, **fields) -> TerminalSession:
        with self._lock:
            terminal = self._terminals.get(terminal_id)
            if terminal is None:
                raise KeyError(f"Unknown terminal: {terminal_id}")
            for key, value in fields.items():
                if not hasattr(terminal, key):
                    raise ValueError(f"Unknown TerminalSession field: {key}")
                setattr(terminal, key, value)
            terminal.updated_at = _now_iso()
            self._save()
            return terminal

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            data = json.loads(self._path.read_text())
        except (json.JSONDecodeError, OSError) as e:
            logger.error("Failed to load terminals from %s: %s", self._path, e)
            return
        for entry in data.get("terminals", []):
            try:
                terminal = TerminalSession(**entry)
                self._terminals[terminal.id] = terminal
            except TypeError as e:
                logger.error("Skipping malformed TerminalSession record: %s (%s)", entry.get("id"), e)

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = {"terminals": [asdict(t) for t in self._terminals.values()]}
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2))
        os.replace(str(tmp), str(self._path))
