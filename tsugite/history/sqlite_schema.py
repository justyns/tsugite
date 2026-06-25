"""Schema + hand-rolled migrations for the SQLite history battery.

Each entry in ``MIGRATIONS`` is ``(name, sql)`` applied in order and recorded in
``_migrations`` (the same pattern as simonw/llm, minus the sqlite-utils dependency).
Migration SQL must be idempotent (``CREATE ... IF NOT EXISTS``): the ``_migrations``
row records completion and gates re-application, but the DDL itself is not wrapped
in a single transaction, so a torn apply re-runs the idempotent statements cleanly.
"""

from __future__ import annotations

import sqlite3

from .models import iso_utc

# One row per event; ``id`` (rowid) gives atomic insertion order + identity, so no
# per-session counter is needed. Session metadata/aggregates/lineage live alongside.
SCHEMA_0001 = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id        TEXT PRIMARY KEY,
    agent             TEXT,
    model             TEXT,
    workspace         TEXT,
    created_at        TEXT NOT NULL,
    ended_at          TEXT,
    status            TEXT,
    error_message     TEXT,
    turn_count        INTEGER NOT NULL DEFAULT 0,
    total_tokens      INTEGER NOT NULL DEFAULT 0,
    total_cost        REAL    NOT NULL DEFAULT 0.0,
    total_duration_ms INTEGER NOT NULL DEFAULT 0,
    updated_at        TEXT,
    last_event_ts     TEXT,
    parent_session    TEXT,
    branched_from_session_id TEXT,
    branch_point_event_id    INTEGER
);

CREATE TABLE IF NOT EXISTS events (
    id         INTEGER PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    type       TEXT NOT NULL,
    ts         TEXT NOT NULL,
    data       TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_events_session      ON events(session_id, id);
CREATE INDEX IF NOT EXISTS idx_events_session_type ON events(session_id, type);
CREATE INDEX IF NOT EXISTS idx_sessions_updated    ON sessions(updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_sessions_created    ON sessions(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_sessions_agent      ON sessions(agent, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_sessions_branchfrom ON sessions(branched_from_session_id);
"""

MIGRATIONS: list[tuple[str, str]] = [
    ("0001_initial", SCHEMA_0001),
]


def apply_migrations(conn: sqlite3.Connection) -> None:
    """Apply any not-yet-recorded migrations. Idempotent and concurrency-safe."""
    conn.execute("CREATE TABLE IF NOT EXISTS _migrations (name TEXT PRIMARY KEY, applied_at TEXT NOT NULL)")
    applied = {row[0] for row in conn.execute("SELECT name FROM _migrations")}
    for name, sql in MIGRATIONS:
        if name in applied:
            continue
        conn.executescript(sql)
        try:
            conn.execute("INSERT INTO _migrations(name, applied_at) VALUES (?, ?)", (name, iso_utc()))
        except sqlite3.IntegrityError:
            # Another process recorded this migration first; the idempotent DDL above is a no-op.
            pass
