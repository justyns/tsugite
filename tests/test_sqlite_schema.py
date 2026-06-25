"""Connection lifecycle + schema/migration framework for the SQLite history battery."""

from datetime import datetime, timezone

import pytest

from tsugite.history.models import iso_utc
from tsugite.history.sqlite_conn import close_all, connect_history_db
from tsugite.history.sqlite_schema import MIGRATIONS, apply_migrations


@pytest.fixture
def db_path(tmp_path):
    yield tmp_path / "history.db"
    close_all()


def test_connect_creates_schema_and_passes_integrity(db_path):
    conn = connect_history_db(db_path)
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"sessions", "events", "_migrations"} <= tables
    assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"


def test_pragmas_set(db_path):
    conn = connect_history_db(db_path)
    assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
    assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    assert conn.execute("PRAGMA busy_timeout").fetchone()[0] >= 1000


def test_same_thread_connection_is_cached(db_path):
    assert connect_history_db(db_path) is connect_history_db(db_path)


def test_migrations_recorded_and_idempotent(db_path):
    conn = connect_history_db(db_path)
    names = [r[0] for r in conn.execute("SELECT name FROM _migrations ORDER BY name")]
    assert names == [m[0] for m in MIGRATIONS]
    apply_migrations(conn)  # re-applying must not error or duplicate
    assert conn.execute("SELECT COUNT(*) FROM _migrations").fetchone()[0] == len(MIGRATIONS)


def test_iso_utc_fixed_precision_and_sortable():
    a = iso_utc(datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc))
    b = iso_utc(datetime(2026, 1, 1, 0, 0, 0, 500000, tzinfo=timezone.utc))
    assert a.endswith("+00:00") and "." in a  # fixed microsecond precision
    assert a < b  # lexicographic ordering == chronological
    assert iso_utc(datetime(2026, 1, 1, 0, 0, 0)) == a  # naive treated as UTC
