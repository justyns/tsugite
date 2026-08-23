"""Phase 1a: the history store sits behind a swappable HistoryBackend seam."""

import pytest

from tsugite.history import (
    Event,
    HistoryBackend,
    Session,
    get_history_backend,
    reset_history_backend,
    set_history_backend,
)


@pytest.fixture(autouse=True)
def _reset_backend():
    reset_history_backend()
    yield
    reset_history_backend()


def test_default_backend_is_sqlite_and_round_trips(isolate_config_files):
    backend = get_history_backend()
    assert isinstance(backend, HistoryBackend)
    assert type(backend).__name__ == "SqliteHistoryBackend"

    session = backend.create(agent_name="tester", model="test:model")
    assert isinstance(session, Session)
    sid = session.session_id
    session.record("user_input", text="hello")

    assert backend.exists(sid)
    assert sid in backend.list_sessions()

    meta = backend.get_meta(sid)
    assert meta is not None and meta.data.get("agent") == "tester"

    loaded = backend.load(sid)
    types = [e.type for e in loaded.load_events()]
    assert types == ["session_start", "user_input"]


class _DummyBackend:
    """Minimal in-memory backend used to prove the seam routes away from jsonl."""

    def __init__(self):
        self.sessions: dict[str, "_DummySession"] = {}

    def create(self, agent_name, model, *, workspace=None, parent_session=None, session_id=None, timestamp=None):
        sid = session_id or f"dummy-{len(self.sessions)}"
        s = _DummySession(sid)
        s.record("session_start", agent=agent_name, model=model)
        self.sessions[sid] = s
        return s

    def load(self, session_id):
        return self.sessions[session_id]

    def exists(self, session_id):
        return session_id in self.sessions

    def get_meta(self, session_id):
        events = self.sessions[session_id].events
        return events[0] if events else None

    def list_sessions(self):
        return list(self.sessions)


class _DummySession:
    def __init__(self, session_id):
        self.session_id = session_id
        self.events: list[Event] = []

    def record(self, type, *, ts=None, **data):
        from datetime import datetime, timezone

        self.events.append(
            Event(type=type, ts=ts or datetime.now(timezone.utc), data={k: v for k, v in data.items() if v is not None})
        )

    def record_many(self, events):
        self.events.extend(events)

    def iter_events(self, types=None):
        wanted = set(types) if types is not None else None
        return (e for e in self.events if wanted is None or e.type in wanted)

    def load_events(self):
        return list(self.events)

    def summary(self):
        from tsugite.history import SessionSummary

        return SessionSummary.from_events(self.events)

    def read_events_window(self, *, after_id=None, before_id=None, limit=None):
        return list(self.events), False


def test_set_backend_routes_sessions_to_it():
    dummy = _DummyBackend()
    set_history_backend(dummy)

    backend = get_history_backend()
    assert backend is dummy

    session = backend.create(agent_name="x", model="m", session_id="abc")
    assert dummy.sessions["abc"] is session
    assert isinstance(session, Session)


def test_backend_resolves_named_plugin_from_config(monkeypatch):
    """config.history.backend selects a backend registered under tsugite.history."""
    import tsugite.history.registry as registry
    import tsugite.plugins as plugins
    from tsugite.config import Config, HistoryConfig

    dummy = _DummyBackend()

    class _EP:
        name = "memory"

        def load(self):
            return lambda config: dummy

    monkeypatch.setattr(registry, "load_config", lambda: Config(history=HistoryConfig(backend="memory")))
    monkeypatch.setattr(
        plugins.importlib.metadata, "entry_points", lambda group: [_EP()] if group == "tsugite.history" else []
    )

    reset_history_backend()
    assert get_history_backend() is dummy


def test_sqlite_backend_selected_by_env(monkeypatch, tmp_path):
    """TSUGITE_HISTORY_BACKEND=sqlite resolves to the built-in SqliteHistoryBackend."""
    from tsugite.history.sqlite_backend import SqliteHistoryBackend
    from tsugite.history.sqlite_conn import close_all

    monkeypatch.setenv("TSUGITE_HISTORY_BACKEND", "sqlite")
    monkeypatch.setenv("TSUGITE_HISTORY_DB", str(tmp_path / "history.db"))
    reset_history_backend()
    try:
        backend = get_history_backend()
        assert isinstance(backend, SqliteHistoryBackend)
        session = backend.create(agent_name="tester", model="test:model")
        session.record("user_input", text="hi")
        assert [e.type for e in backend.load(session.session_id).load_events()] == ["session_start", "user_input"]
    finally:
        close_all()


@pytest.fixture
def sqlite_backend(tmp_path):
    """A SqliteHistoryBackend on its own database file."""
    from tsugite.history.sqlite_backend import SqliteHistoryBackend
    from tsugite.history.sqlite_conn import close_all

    yield SqliteHistoryBackend(tmp_path / "history.db")
    close_all()


def _seed_cross_session(backend):
    """Two sessions whose events interleave, so ordering can't come from grouping."""
    a = backend.create(agent_name="alpha", model="test:model", session_id="sess-a")
    b = backend.create(agent_name="beta", model="test:model", session_id="sess-b")
    a.record("user_input", text="one")
    b.record("user_input", text="two")
    a.record("session_end", status="success")
    b.record("session_end", status="error", error_message="boom")
    return a, b


def test_recent_events_reads_newest_first_across_sessions(sqlite_backend):
    _seed_cross_session(sqlite_backend)

    rows = sqlite_backend.recent_events(types=["user_input", "session_end"], limit=3)

    assert [(sid, event.type) for sid, event in rows] == [
        ("sess-b", "session_end"),
        ("sess-a", "session_end"),
        ("sess-b", "user_input"),
    ]


def test_latest_event_per_session_keeps_one_row_per_session(sqlite_backend):
    """A session with many matching events collapses to its newest one, so a busy
    session cannot push quieter ones out of the window."""
    session_a, _b = _seed_cross_session(sqlite_backend)
    for _ in range(5):
        session_a.record("session_end", status="success")

    rows = sqlite_backend.latest_event_per_session(types=["session_end"], limit=10)
    assert [sid for sid, _event in rows] == ["sess-a", "sess-b"]

    assert [sid for sid, _event in sqlite_backend.latest_event_per_session(types=["session_end"], limit=1)] == [
        "sess-a"
    ]


def test_recent_events_filters_by_type_and_carries_data(sqlite_backend):
    _seed_cross_session(sqlite_backend)

    rows = sqlite_backend.recent_events(types=["session_end"], limit=10)

    assert [(sid, event.data.get("status")) for sid, event in rows] == [
        ("sess-b", "error"),
        ("sess-a", "success"),
    ]


def test_recent_events_reads_the_index_without_a_sort(sqlite_backend):
    """recent_events reaches idx_events_type_id and needs no sort step.

    Both halves matter: an index that cannot serve the ORDER BY still gets named.
    Tracing the connection asserts the plan of the statement the method actually ran.
    """
    _seed_cross_session(sqlite_backend)
    conn = sqlite_backend._conn()
    seen: list[str] = []
    conn.set_trace_callback(seen.append)
    sqlite_backend.recent_events(types=["compaction"], limit=50)
    conn.set_trace_callback(None)
    plan = " ".join(row[-1] for row in conn.execute("EXPLAIN QUERY PLAN " + seen[-1]).fetchall())

    assert "idx_events_type_id" in plan, plan
    assert "TEMP B-TREE" not in plan, plan


def test_a_stale_migration_row_does_not_block_later_migrations(tmp_path):
    """A recorded name no longer in MIGRATIONS (a renamed dev-era migration) doesn't
    stop the current list from applying its indexes.
    """
    import sqlite3

    from tsugite.history.sqlite_schema import SCHEMA_0001, apply_migrations

    conn = sqlite3.connect(tmp_path / "history.db")
    conn.executescript(SCHEMA_0001)
    conn.execute("CREATE TABLE IF NOT EXISTS _migrations (name TEXT PRIMARY KEY, applied_at TEXT NOT NULL)")
    conn.execute("INSERT INTO _migrations(name, applied_at) VALUES ('0002_events_type_session_id', '')")
    conn.commit()

    apply_migrations(conn)
    indexes = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='events'")}
    assert "idx_events_type_id" in indexes, sorted(indexes)
