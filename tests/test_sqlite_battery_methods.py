"""Extended battery methods: list_sessions filters, search, count_events, ensure_session, purge."""

from datetime import datetime, timedelta, timezone

import pytest

from tsugite.history.models import iso_utc
from tsugite.history.sqlite_backend import SqliteHistoryBackend
from tsugite.history.sqlite_conn import close_all

NOW = datetime(2026, 6, 22, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def backend(tmp_path):
    b = SqliteHistoryBackend(db_path=tmp_path / "history.db")
    yield b
    close_all()


def test_count_events(backend):
    s = backend.create("chat", "m")
    s.record("user_input", text="a")
    s.record("user_input", text="b")
    s.record("model_response", raw_content="r")
    assert backend.count_events(s.session_id) == 4  # session_start + 2 + 1
    assert backend.count_events(s.session_id, type="user_input") == 2


def test_list_sessions_filters(backend):
    backend.create("alpha", "m", workspace="w1")
    backend.create("beta", "m", workspace="w2")
    backend.create("alpha", "m", workspace="w2")
    assert len(backend.list_sessions(agent="alpha")) == 2
    assert len(backend.list_sessions(workspace="w2")) == 2
    assert len(backend.list_sessions(agent="alpha", workspace="w2")) == 1
    assert len(backend.list_sessions(limit=1)) == 1


def test_search_matches_user_and_model_text(backend):
    s1 = backend.create("chat", "m")
    s1.record("user_input", text="please refactor the parser")
    s2 = backend.create("chat", "m")
    s2.record("model_response", raw_content="the parser is now faster")
    s3 = backend.create("chat", "m")
    s3.record("user_input", text="unrelated topic")

    hits = backend.search("parser")
    found = {h["session_id"] for h in hits}
    assert found == {s1.session_id, s2.session_id}
    assert all(h.get("snippet") for h in hits)


def test_search_agent_filter_and_limit(backend):
    a = backend.create("alpha", "m")
    a.record("user_input", text="shared keyword")
    b = backend.create("beta", "m")
    b.record("user_input", text="shared keyword")
    assert {h["session_id"] for h in backend.search("keyword", agent="alpha")} == {a.session_id}
    assert len(backend.search("keyword", limit=1)) == 1


def test_ensure_session_creates_bare_row_without_session_start(backend):
    s = backend.ensure_session("telemetry-1")
    assert backend.exists("telemetry-1")
    assert s.load_events() == []  # no session_start synthesized
    # Idempotent + never overwrites: a later session_start folds metadata.
    s2 = backend.ensure_session("telemetry-1")
    assert s2.session_id == "telemetry-1"


def test_purge_removes_old_sessions_and_their_events(backend):
    old = backend.create("chat", "m")
    old.record("user_input", text="ancient")
    fresh = backend.create("chat", "m")
    fresh.record("user_input", text="recent")
    # Retention is by last-write recency (updated_at); age the old session 40 days back.
    backend._conn().execute(
        "UPDATE sessions SET updated_at=? WHERE session_id=?",
        (iso_utc(NOW - timedelta(days=40)), old.session_id),
    )

    removed = backend.purge(older_than=NOW - timedelta(days=30))
    assert removed == 1
    assert not backend.exists(old.session_id)
    assert backend.exists(fresh.session_id)
    # events cascade-deleted with the session
    assert backend.count_events(old.session_id) == 0
