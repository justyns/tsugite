"""Protocols for the swappable history battery.

A `HistoryBackend` owns session lifecycle (create / load / list / metadata) keyed
by conversation id; a `Session` is the per-conversation read/write handle it returns.
The default `JsonlHistoryBackend` (storage.py) wraps one JSONL file per session; a
plugin backend (e.g. postgres) answers the same calls from its own store.
"""

from datetime import datetime
from typing import Any, Iterable, Iterator, List, Optional, Protocol, runtime_checkable

from .models import Event
from .storage import SessionSummary


@runtime_checkable
class Session(Protocol):
    """Read/write handle for one conversation's event log."""

    session_id: str

    def record(self, type: str, *, ts: Optional[datetime] = None, **data: Any) -> None: ...

    def record_many(self, events: Iterable[Event]) -> None: ...

    def iter_events(self, types: Optional[Iterable[str]] = None) -> Iterator[Event]: ...

    def load_events(self) -> List[Event]: ...

    def summary(self) -> SessionSummary: ...


@runtime_checkable
class HistoryBackend(Protocol):
    """Session lifecycle + lookup, keyed by conversation id."""

    def create(
        self,
        agent_name: str,
        model: str,
        *,
        workspace: Optional[str] = None,
        parent_session: Optional[str] = None,
        session_id: Optional[str] = None,
        timestamp: Optional[datetime] = None,
    ) -> Session: ...

    def load(self, session_id: str) -> Session: ...

    def exists(self, session_id: str) -> bool: ...

    def get_meta(self, session_id: str) -> Optional[Event]: ...

    def list_sessions(self) -> List[str]: ...
