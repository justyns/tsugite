"""Shared base for the daemon's record stores.

One copy of the locking, persistence, and guarded state-machine logic that
JobStore and TerminalSessionStore both need - the two had already started to
drift (one locked its read accessors, the other didn't).

Persistence is a per-collection table in ``<dir>/daemon.db`` (SQLite, WAL,
write-through): each mutation upserts or deletes exactly the affected row, so
nothing is lost on a crash and a large store isn't re-serialized wholesale per
save. The legacy per-store ``*.json`` file remains the constructor argument:
it is read once as a migration source when the db collection is empty, and
left untouched afterwards as a backup.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
import threading
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

DAEMON_DB_FILENAME = "daemon.db"

_BUSY_TIMEOUT_MS = 5000
_COLLECTION_NAME_RE = re.compile(r"^[a-z_][a-z0-9_]*$")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class SqliteCollectionStorage:
    """One collection (table) of ``id -> JSON document`` rows in a shared SQLite db.

    Rows are opaque JSON documents rather than per-field columns so dataclass
    schema growth never needs a db migration; queries stay in the stores'
    in-memory dicts. The connection is NOT internally synchronized - callers
    serialize access with their own lock (every store method already runs
    under one).
    """

    def __init__(self, db_path: Path, collection: str):
        if not _COLLECTION_NAME_RE.match(collection):
            raise ValueError(f"Invalid collection name: {collection!r}")
        self._collection = collection
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False, isolation_level=None)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(f"PRAGMA busy_timeout={_BUSY_TIMEOUT_MS}")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute(f'CREATE TABLE IF NOT EXISTS "{collection}" (id TEXT PRIMARY KEY, data TEXT NOT NULL)')
        # Tracks which collections have completed their one-time legacy-JSON
        # import. Row-count can't stand in for this: a collection legitimately
        # emptied at runtime must NOT re-import the stale legacy file on the
        # next start.
        self._conn.execute('CREATE TABLE IF NOT EXISTS "_migrated" (collection TEXT PRIMARY KEY, at TEXT NOT NULL)')
        # Owner-only, like tokens.json was: the db holds session content and
        # auth token hashes. SQLite propagates the mode to -wal/-shm siblings.
        try:
            os.chmod(db_path, 0o600)
        except OSError:
            pass

    @classmethod
    def for_state_file(cls, legacy_path: Path, collection: str) -> "SqliteCollectionStorage":
        """Storage for a store whose legacy JSON lived at ``legacy_path``:
        the shared daemon.db sits next to it."""
        return cls(legacy_path.parent / DAEMON_DB_FILENAME, collection)

    def load_or_migrate(self, legacy_path: Path, collection_key: str, legacy_reader=None) -> tuple[list[dict], bool]:
        """Entries from the db, falling back to a one-time legacy JSON import.

        The import runs at most once per collection (tracked in the _migrated
        table), so a collection emptied at runtime stays empty across restarts
        instead of resurrecting the stale legacy file. ``legacy_reader``
        overrides the default ``{key: ...}`` file shape for stores whose legacy
        file was a bare array (push subscriptions, auth tokens).

        Returns (entries, migrating). When migrating is True the caller must
        persist its parsed records back (replace_all / _save) so the import
        becomes durable; the legacy file is left untouched as a backup.
        """
        entries = self.load_all()
        if entries or self._is_migrated():
            self._mark_migrated()  # stamp dbs created before the marker existed
            return entries, False
        entries = legacy_reader() if legacy_reader else load_legacy_json_entries(legacy_path, collection_key)
        if entries:
            logger.info(
                "Migrating %d %s record(s) from %s into daemon.db (legacy file kept as backup)",
                len(entries),
                collection_key,
                legacy_path.name,
            )
        self._mark_migrated()
        return entries, bool(entries)

    def _is_migrated(self) -> bool:
        row = self._conn.execute('SELECT 1 FROM "_migrated" WHERE collection = ?', (self._collection,)).fetchone()
        return row is not None

    def _mark_migrated(self) -> None:
        self._conn.execute(
            'INSERT OR IGNORE INTO "_migrated" (collection, at) VALUES (?, ?)',
            (self._collection, datetime.now(timezone.utc).isoformat()),
        )

    def get(self, record_id: str) -> dict | None:
        row = self._conn.execute(f'SELECT data FROM "{self._collection}" WHERE id = ?', (record_id,)).fetchone()
        if row is None:
            return None
        try:
            return json.loads(row[0])
        except json.JSONDecodeError as e:
            logger.error("Corrupt %s row %s: %s", self._collection, record_id, e)
            return None

    def load_all(self) -> list[dict]:
        rows = self._conn.execute(f'SELECT data FROM "{self._collection}"').fetchall()
        entries = []
        for (data,) in rows:
            try:
                entries.append(json.loads(data))
            except json.JSONDecodeError as e:
                logger.error("Skipping corrupt %s row: %s", self._collection, e)
        return entries

    def upsert(self, record_id: str, entry: dict) -> None:
        # default=str: a stray non-JSON value in a record field must degrade to
        # its string form, not crash the mutation that carries it.
        self._conn.execute(
            f'INSERT INTO "{self._collection}" (id, data) VALUES (?, ?) '
            "ON CONFLICT(id) DO UPDATE SET data = excluded.data",
            (record_id, json.dumps(entry, default=str)),
        )

    def delete(self, record_id: str) -> None:
        self._conn.execute(f'DELETE FROM "{self._collection}" WHERE id = ?', (record_id,))

    def replace_all(self, entries: dict[str, dict]) -> None:
        """Atomically make the collection equal to ``entries``."""
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            self._conn.execute(f'DELETE FROM "{self._collection}"')
            self._conn.executemany(
                f'INSERT INTO "{self._collection}" (id, data) VALUES (?, ?)',
                [(rid, json.dumps(entry, default=str)) for rid, entry in entries.items()],
            )
            self._conn.execute("COMMIT")
        except Exception:
            self._conn.execute("ROLLBACK")
            raise

    def close(self) -> None:
        self._conn.close()


def load_legacy_json_entries(path: Path, collection_key: str) -> list[dict]:
    """Entries from a legacy whole-file JSON store, tolerating both the list
    shape (``{key: [...]}``) and the dict shape (``{key: {id: {...}}}``).
    Returns [] when the file is absent or unreadable."""
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        logger.error("Failed to load legacy %s from %s: %s", collection_key, path, e)
        return []
    raw = data.get(collection_key, [])
    if isinstance(raw, dict):
        return list(raw.values())
    return raw if isinstance(raw, list) else []


class RecordStore:
    """In-memory dict of dataclass records + write-through SQLite persistence.

    Records must have: id, state, updated_at, resolved_at, parent_session_id.
    Subclasses define the class attributes below and may override `_load_entry`
    (per-entry migrations) and `_after_load` (post-load reconciliation; return
    True to persist the result).
    """

    record_cls: type
    collection_key: str  # table name / legacy top-level JSON key, e.g. "jobs"
    record_label: str  # for error messages, e.g. "job"
    valid_transitions: dict[str, frozenset[str]]
    terminal_states: frozenset[str]
    transition_error_cls: type[ValueError] = ValueError

    def __init__(self, path: Path):
        self._path = path  # legacy JSON location; migration source only
        self._storage = SqliteCollectionStorage.for_state_file(path, self.collection_key)
        self._records: dict = {}
        self._lock = threading.Lock()
        self._load()

    def _persist(self, record) -> None:
        """Write-through one record's row. Caller holds the lock."""
        self._storage.upsert(record.id, asdict(record))

    def add(self, record):
        with self._lock:
            if record.id in self._records:
                raise ValueError(f"{self.record_cls.__name__} already exists: {record.id}")
            self._records[record.id] = record
            self._persist(record)
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
            self._persist(record)
            return record

    def _on_state_updated(self, record) -> None:
        """Hook called (under the lock) after a state transition, before persist.
        Hooks that drop other records must go through `_delete_record` so the
        deletion is durable."""

    def _delete_record(self, record_id: str) -> None:
        """Remove a record from memory AND storage. Caller holds the lock."""
        self._records.pop(record_id, None)
        self._storage.delete(record_id)

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
            self._persist(record)
            return record

    def _coerce_update_value(self, key: str, value):
        """Hook for per-field validation/coercion in update()."""
        return value

    def _load_entry(self, entry: dict):
        """Build a record from a raw persisted entry; return None to drop it.
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
        entries, migrating = self._storage.load_or_migrate(self._path, self.collection_key)
        for entry in entries:
            record = self._load_entry(entry)
            if record is not None:
                self._records[record.id] = record
        if self._after_load() or migrating:
            self._storage.replace_all({r.id: asdict(r) for r in self._records.values()})


# Deprecated alias - external plugins may still import the old name.
JsonRecordStore = RecordStore

__all__ = ["RecordStore", "JsonRecordStore", "SqliteCollectionStorage", "load_legacy_json_entries", "now_iso"]
