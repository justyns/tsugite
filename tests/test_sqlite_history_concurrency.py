"""Concurrency: WAL + busy_timeout + an app-level retry keep writes correct under contention.

The daemon (worker threads), the scheduler, and the CLI all write the same history db.
SQLite serializes writers; these tests assert that under real contention every event lands,
ids stay unique, and the maintained aggregates match the event log.
"""

import sqlite3
import threading

import pytest

from tsugite.history.sqlite_backend import SqliteHistoryBackend, _is_locked, _run_write
from tsugite.history.sqlite_conn import close_all, connect_history_db


@pytest.fixture
def backend(tmp_path):
    b = SqliteHistoryBackend(db_path=tmp_path / "history.db")
    yield b
    close_all()


def test_retry_runs_body_once_when_unlocked(tmp_path):
    conn = connect_history_db(tmp_path / "x.db")
    calls = []
    _run_write(conn, lambda: calls.append(1), sleep=lambda s: None)
    close_all()
    assert calls == [1]


def test_retry_recovers_from_transient_lock(tmp_path):
    conn = connect_history_db(tmp_path / "x.db")
    calls = []

    def body():
        calls.append(1)
        if len(calls) < 2:
            raise sqlite3.OperationalError("database is locked")

    _run_write(conn, body, sleep=lambda s: None)
    close_all()
    assert len(calls) == 2  # retried once, then succeeded


def test_retry_reraises_non_lock_errors(tmp_path):
    conn = connect_history_db(tmp_path / "x.db")
    with pytest.raises(sqlite3.OperationalError):
        _run_write(
            conn, lambda: (_ for _ in ()).throw(sqlite3.OperationalError("no such table: nope")), sleep=lambda s: None
        )
    close_all()


def test_is_locked_matches_lock_and_busy():
    assert _is_locked(sqlite3.OperationalError("database is locked"))
    assert _is_locked(sqlite3.OperationalError("database table is busy"))
    assert not _is_locked(sqlite3.OperationalError("syntax error"))


def test_concurrent_writes_to_same_session(backend):
    s = backend.create("chat", "m")
    workers, per_worker = 6, 15

    def worker():
        sess = backend.load(s.session_id)  # picks up this thread's own connection
        for _ in range(per_worker):
            sess.record("user_input", text="x")

    threads = [threading.Thread(target=worker) for _ in range(workers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    reloaded = backend.load(s.session_id)
    assert backend.count_events(s.session_id, type="user_input") == workers * per_worker
    ids = [e.id for e in reloaded.iter_events()]
    assert len(ids) == len(set(ids))  # no duplicate rowids
    assert reloaded.summary().turn_count == workers * per_worker  # aggregate matches log


def test_concurrent_writes_to_different_sessions(backend):
    workers, per_worker = 8, 10
    created: list[str] = []
    lock = threading.Lock()

    def worker():
        sess = backend.create("chat", "m")
        for _ in range(per_worker):
            sess.record("user_input", text="x")
        with lock:
            created.append(sess.session_id)

    threads = [threading.Thread(target=worker) for _ in range(workers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(created) == workers
    assert len(set(created)) == workers  # unique session ids under concurrent create()
    for sid in created:
        assert backend.count_events(sid, type="user_input") == per_worker
