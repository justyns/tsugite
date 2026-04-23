"""Append-only event log storage for sessions.

One JSONL file per session. Each line is an `Event`. Writes are file-locked so
multiple processes (e.g., daemon + scheduler) can append safely. Reads stream
the file once and tolerate a malformed line (skip with stderr warning) so a
torn write can't poison the whole file.
"""

import hashlib
import json
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, List, Optional

import portalocker

from tsugite.config import get_xdg_data_path, load_config

from .models import Event


def get_history_dir() -> Path:
    return get_xdg_data_path("history")


def generate_session_id(agent_name: str, timestamp: Optional[datetime] = None) -> str:
    if timestamp is None:
        timestamp = datetime.now(timezone.utc)
    date_str = timestamp.strftime("%Y%m%d_%H%M%S")
    clean_agent = "".join(c if c.isalnum() or c == "-" else "_" for c in agent_name)[:20]
    digest = hashlib.sha256(f"{timestamp.isoformat()}_{agent_name}".encode()).hexdigest()[:6]
    return f"{date_str}_{clean_agent}_{digest}"


def get_machine_name() -> str:
    config = load_config()
    name = getattr(config, "machine_name", None)
    if name:
        return name
    try:
        return socket.gethostname()
    except Exception:
        return "unknown"


class SessionStorage:
    """Append-only event log for one session."""

    def __init__(self, session_path: Path):
        self.session_path = session_path
        self.session_id = session_path.stem

    @classmethod
    def create(
        cls,
        agent_name: str,
        model: str,
        workspace: Optional[str] = None,
        parent_session: Optional[str] = None,
        session_path: Optional[Path] = None,
        timestamp: Optional[datetime] = None,
    ) -> "SessionStorage":
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)
        if session_path is None:
            session_path = get_history_dir() / f"{generate_session_id(agent_name, timestamp)}.jsonl"
        session_path.parent.mkdir(parents=True, exist_ok=True)

        storage = cls(session_path)
        storage.record(
            "session_start",
            ts=timestamp,
            agent=agent_name,
            model=model,
            machine=get_machine_name(),
            workspace=workspace,
            parent_session=parent_session,
        )
        return storage

    @classmethod
    def load(cls, session_path: Path) -> "SessionStorage":
        if not session_path.exists():
            raise FileNotFoundError(f"Session not found: {session_path}")
        return cls(session_path)

    def record(self, type: str, *, ts: Optional[datetime] = None, **data: Any) -> None:
        """Append one event."""
        event = Event(
            type=type,
            ts=ts or datetime.now(timezone.utc),
            data={k: v for k, v in data.items() if v is not None},
        )
        self._write([event])

    def record_many(self, events: Iterable[Event]) -> None:
        """Append multiple events under one lock."""
        events = list(events)
        if events:
            self._write(events)

    def _write(self, events: List[Event]) -> None:
        self.session_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.session_path, "a", encoding="utf-8") as f:
            portalocker.lock(f, portalocker.LOCK_EX)
            try:
                for e in events:
                    f.write(e.model_dump_json(exclude_none=True))
                    f.write("\n")
                f.flush()
            finally:
                portalocker.unlock(f)

    def iter_events(self, types: Optional[Iterable[str]] = None) -> Iterator[Event]:
        """Yield events in file order, optionally filtered by type."""
        if not self.session_path.exists():
            return
        wanted = set(types) if types is not None else None
        with open(self.session_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    event = Event.model_validate(json.loads(line))
                except (json.JSONDecodeError, ValueError) as e:
                    print(f"Warning: skipping malformed event at {self.session_path}:{line_num}: {e}", file=sys.stderr)
                    continue
                if wanted is None or event.type in wanted:
                    yield event

    def load_events(self) -> List[Event]:
        return list(self.iter_events())

    @classmethod
    def load_meta_fast(cls, session_path: Path) -> Optional[Event]:
        """Read just the first event (the session_start) for fast list views."""
        try:
            with open(session_path, "r", encoding="utf-8") as f:
                line = f.readline().strip()
                if line:
                    return Event.model_validate(json.loads(line))
        except Exception:
            pass
        return None

    def summary(self) -> "SessionSummary":
        """Compute one-shot aggregates by walking the event log.

        Scans the file each call — sessions are append-only and small enough
        that this is fine for CLI/list views. If you need many properties,
        cache the returned summary; don't repeatedly access individual fields.
        """
        return SessionSummary.from_events(self.iter_events())


class SessionSummary:
    """Aggregates derived from an event log: agent, model, totals, status."""

    def __init__(self):
        self.agent: Optional[str] = None
        self.model: Optional[str] = None
        self.machine: Optional[str] = None
        self.workspace: Optional[str] = None
        self.created_at: Optional[datetime] = None
        self.parent_session: Optional[str] = None
        self.status: Optional[str] = None
        self.error_message: Optional[str] = None
        self.turn_count: int = 0  # number of user_input events
        self.total_tokens: int = 0
        self.total_cost: float = 0.0
        self.total_duration_ms: int = 0
        self.functions_called: set[str] = set()
        self.last_response_text: str = ""

    @classmethod
    def from_events(cls, events: Iterable[Event]) -> "SessionSummary":
        s = cls()
        for event in events:
            data = event.data
            if event.type == "session_start":
                s.agent = data.get("agent")
                s.model = data.get("model")
                s.machine = data.get("machine")
                s.workspace = data.get("workspace")
                s.created_at = event.ts
                s.parent_session = data.get("parent_session")
            elif event.type == "user_input":
                s.turn_count += 1
            elif event.type == "model_response":
                usage = data.get("usage") or {}
                if isinstance(usage, dict):
                    s.total_tokens += int(usage.get("total_tokens") or 0)
                cost = data.get("cost")
                if cost:
                    s.total_cost += float(cost)
                s.last_response_text = data.get("raw_content", s.last_response_text)
            elif event.type == "code_execution":
                s.total_duration_ms += int(data.get("duration_ms") or 0)
                for fn in data.get("tools_called") or []:
                    s.functions_called.add(fn)
            elif event.type == "tool_invocation":
                name = data.get("name")
                if name:
                    s.functions_called.add(name)
                s.total_duration_ms += int(data.get("duration_ms") or 0)
            elif event.type == "session_end":
                s.status = data.get("status")
                s.error_message = data.get("error_message")
        return s


def list_session_files() -> List[Path]:
    history_dir = get_history_dir()
    if not history_dir.exists():
        return []
    try:
        files = list(history_dir.glob("*.jsonl"))
        files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return files
    except OSError:
        return []
