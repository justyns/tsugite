"""Protocols for the swappable history battery.

A `HistoryBackend` owns session lifecycle (create / load / list / metadata) keyed
by conversation id; a `Session` is the per-conversation read/write handle it returns.
`SqliteHistoryBackend` is the built-in implementation; a plugin backend (e.g.
postgres) answers the same calls from its own store.
"""

from datetime import datetime
from typing import Any, Iterable, Iterator, List, Optional, Protocol, runtime_checkable

from .models import Event, SessionSummary


@runtime_checkable
class Session(Protocol):
    """Read/write handle for one conversation's event log."""

    session_id: str

    def record(self, type: str, *, ts: Optional[datetime] = None, **data: Any) -> None: ...

    def record_many(self, events: Iterable[Event]) -> None: ...

    def iter_events(self, types: Optional[Iterable[str]] = None) -> Iterator[Event]: ...

    def load_events(self) -> List[Event]: ...

    def summary(self) -> SessionSummary: ...

    def read_events_window(
        self,
        *,
        after_id: Optional[int] = None,
        before_id: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> "tuple[List[Event], bool]":
        """A chronological window of events plus whether older ones exist before it.

        ``after_id``: forward delta (has_more always False). ``before_id`` + ``limit``:
        the newest ``limit`` below the cursor. ``limit`` alone: the newest ``limit``.
        None of them: every event.
        """
        ...


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

    def list_sessions(
        self,
        *,
        workspace: Optional[str] = None,
        agent: Optional[str] = None,
        status: Optional[str] = None,
        since: Optional[datetime] = None,
        before: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> List[str]: ...

    def count_events(self, session_id: str, *, type: Optional[str] = None) -> int: ...

    def ensure_session(self, session_id: str) -> Session: ...

    def delete_session(self, session_id: str) -> bool: ...

    def search(self, query: str, *, agent: Optional[str] = None, limit: int = 50) -> List[dict]: ...

    def export_jsonl(self, session_id: str) -> Iterator[str]: ...

    def create_branch(
        self,
        source_id: str,
        *,
        at_event_id: int,
        new_session_id: Optional[str] = None,
    ) -> str: ...
