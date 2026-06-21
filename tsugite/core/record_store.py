"""Shared base for the daemon's JSON-file-backed record stores.

One copy of the locking, atomic tmpfile-swap persistence, and guarded
state-machine logic that JobStore and TerminalSessionStore both need - the two
had already started to drift (one locked its read accessors, the other didn't).
"""

from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class JsonRecordStore:
    """In-memory dict of dataclass records + atomic JSON persistence.

    Records must have: id, state, updated_at, resolved_at, parent_session_id.
    Subclasses define the class attributes below and may override `_load_entry`
    (per-entry migrations) and `_after_load` (post-load reconciliation; return
    True to persist the result).
    """

    record_cls: type
    collection_key: str  # top-level JSON key, e.g. "jobs"
    record_label: str  # for error messages, e.g. "job"
    valid_transitions: dict[str, frozenset[str]]
    terminal_states: frozenset[str]
    transition_error_cls: type[ValueError] = ValueError

    def __init__(self, path: Path):
        self._path = path
        self._records: dict = {}
        self._lock = threading.Lock()
        self._load()

    def add(self, record):
        with self._lock:
            if record.id in self._records:
                raise ValueError(f"{self.record_cls.__name__} already exists: {record.id}")
            self._records[record.id] = record
            self._save()
        return record

    def get(self, record_id: str):
        with self._lock:
            return self._records.get(record_id)

    def list_all(self) -> list:
        with self._lock:
            return list(self._records.values())

    def list_active(self) -> list:
        with self._lock:
            return [r for r in self._records.values() if r.state not in self.terminal_states]

    def list_for_parent(self, parent_session_id: str) -> list:
        with self._lock:
            return [r for r in self._records.values() if r.parent_session_id == parent_session_id]

    def update_state(self, record_id: str, new_state: str):
        with self._lock:
            record = self._records.get(record_id)
            if record is None:
                raise KeyError(f"Unknown {self.record_label}: {record_id}")
            allowed = self.valid_transitions.get(record.state, frozenset())
            if new_state not in allowed:
                raise self.transition_error_cls(
                    f"Invalid {self.record_cls.__name__} state transition: "
                    f"{record.state} -> {new_state} ({self.record_label} {record_id})"
                )
            record.state = new_state
            record.updated_at = now_iso()
            if new_state in self.terminal_states and not record.resolved_at:
                record.resolved_at = record.updated_at
            self._on_state_updated(record)
            self._save()
            return record

    def _on_state_updated(self, record) -> None:
        """Hook called (under the lock) after a state transition, before save."""

    def update(self, record_id: str, **fields):
        with self._lock:
            record = self._records.get(record_id)
            if record is None:
                raise KeyError(f"Unknown {self.record_label}: {record_id}")
            for key, value in fields.items():
                if not hasattr(record, key):
                    raise ValueError(f"Unknown {self.record_cls.__name__} field: {key}")
                setattr(record, key, self._coerce_update_value(key, value))
            record.updated_at = now_iso()
            self._save()
            return record

    def _coerce_update_value(self, key: str, value):
        """Hook for per-field validation/coercion in update()."""
        return value

    def _load_entry(self, entry: dict):
        """Build a record from a raw on-disk entry; return None to drop it.
        Subclasses apply schema migrations here."""
        try:
            return self.record_cls(**entry)
        except TypeError as e:
            logger.error("Skipping malformed %s record: %s (%s)", self.record_cls.__name__, entry.get("id"), e)
            return None

    def _after_load(self) -> bool:
        """Post-load reconciliation hook. Return True to persist the result."""
        return False

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            data = json.loads(self._path.read_text())
        except (json.JSONDecodeError, OSError) as e:
            logger.error("Failed to load %s from %s: %s", self.collection_key, self._path, e)
            return
        for entry in data.get(self.collection_key, []):
            record = self._load_entry(entry)
            if record is not None:
                self._records[record.id] = record
        if self._after_load():
            self._save()

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = {self.collection_key: [asdict(r) for r in self._records.values()]}
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2))
        os.replace(str(tmp), str(self._path))


__all__ = ["JsonRecordStore", "now_iso"]
