"""Durable needs-attention records (daemon.db `attention` collection).

One record per thing that is waiting on the user. A row is in the worklist when
records are open against it.
"""

import logging
import threading
from dataclasses import asdict, dataclass
from dataclasses import fields as dataclass_fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from uuid import uuid4

from tsugite.core.record_store import SqliteCollectionStorage

logger = logging.getLogger(__name__)

# Where an open record came from. The pair (source, ref_id) identifies the thing
# waiting, so re-reporting it is idempotent.
SOURCE_DELIVERY = "delivery"
SOURCE_ASK = "ask"
SOURCE_JOB = "job"
SOURCE_ERROR = "error"

# Sources the user can dismiss. An ask clears when it is answered and a parked
# job when it resumes, so neither belongs here.
ACKNOWLEDGEABLE_SOURCES = (SOURCE_DELIVERY, SOURCE_ERROR)

OWNER_SESSION = "session"


@dataclass
class AttentionRecord:
    owner_kind: str
    owner_id: str
    source: str
    ref_id: str
    kind: str
    id: str = ""
    created_at: str = ""

    def __post_init__(self):
        if not self.id:
            self.id = f"attn-{uuid4().hex[:12]}"
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()


class AttentionStore:
    """In-memory records with write-through SQLite persistence. The set holds only
    what is still waiting: clearing a record removes it."""

    def __init__(self, path: Path):
        self._storage = SqliteCollectionStorage.for_state_file(path, "attention")
        self._records: dict[str, AttentionRecord] = {}
        self._lock = threading.RLock()
        self._load()

    def open(
        self,
        *,
        owner_kind: str,
        owner_id: str,
        source: str,
        ref_id: str,
        kind: str,
    ) -> Optional[AttentionRecord]:
        """Returns None when a record is already open against that ref, so a caller can
        tell an announcement from a re-report."""
        with self._lock:
            if self._open_for_ref_locked(source, ref_id) is not None:
                return None
            record = AttentionRecord(
                owner_kind=owner_kind,
                owner_id=owner_id,
                source=source,
                ref_id=ref_id,
                kind=kind,
            )
            self._records[record.id] = record
            self._storage.upsert(record.id, asdict(record))
            return record

    def clear_ref(self, source: str, ref_id: str) -> list[AttentionRecord]:
        with self._lock:
            record = self._open_for_ref_locked(source, ref_id)
            return [self._remove_locked(record)] if record else []

    def clear_owner(self, owner_id: str, *, source: Optional[str] = None) -> list[AttentionRecord]:
        with self._lock:
            targets = [
                r for r in self._records.values() if r.owner_id == owner_id and (source is None or r.source == source)
            ]
            return [self._remove_locked(r) for r in targets]

    def clear_stale_asks(self) -> list[AttentionRecord]:
        """Close ask records left by a previous process.

        A blocked ask_user call cannot outlive the process it blocked in, so nothing
        is left that could answer one read back from disk.
        """
        with self._lock:
            return [self._remove_locked(r) for r in list(self._records.values()) if r.source == SOURCE_ASK]

    def open_records(self, owner_id: Optional[str] = None) -> list[AttentionRecord]:
        with self._lock:
            return [r for r in self._records.values() if owner_id is None or r.owner_id == owner_id]

    def _open_for_ref_locked(self, source: str, ref_id: str) -> Optional[AttentionRecord]:
        return next((r for r in self._records.values() if r.source == source and r.ref_id == ref_id), None)

    def _remove_locked(self, record: AttentionRecord) -> AttentionRecord:
        del self._records[record.id]
        self._storage.delete(record.id)
        return record

    def _load(self) -> None:
        valid_fields = {f.name for f in dataclass_fields(AttentionRecord)}
        for entry in self._storage.load_all():
            try:
                record = AttentionRecord(**{k: v for k, v in entry.items() if k in valid_fields})
            except TypeError as e:
                logger.error("Skipping malformed attention record: %s", e)
                continue
            self._records[record.id] = record
