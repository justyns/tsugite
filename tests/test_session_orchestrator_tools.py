"""Tests for new session tools: session_events_since, session_summary, updated_since filter."""

from datetime import datetime, timedelta, timezone

import pytest

from tsugite.daemon.session_store import (
    Session,
    SessionSource,
    SessionStatus,
    SessionStore,
)


@pytest.fixture
def store(tmp_path):
    store_path = tmp_path / "session_store.json"
    return SessionStore(store_path)


@pytest.fixture
def session_with_events(store):
    """Create a session and add some events to it."""
    session = Session(
        id="test-session-1",
        agent="odyn",
        source=SessionSource.BACKGROUND.value,
        status=SessionStatus.COMPLETED.value,
        prompt="Do something useful",
        result="Done doing something useful",
    )
    store.create_session(session)

    # Add events with increasing timestamps
    base = datetime(2026, 3, 15, 10, 0, 0, tzinfo=timezone.utc)
    events = [
        {"type": "session_start", "timestamp": (base).isoformat(), "agent": "odyn", "prompt": "Do something"},
        {"type": "tool_call", "timestamp": (base + timedelta(seconds=10)).isoformat(), "name": "read_file"},
        {"type": "tool_call", "timestamp": (base + timedelta(seconds=20)).isoformat(), "name": "write_file"},
        {"type": "tool_call", "timestamp": (base + timedelta(seconds=30)).isoformat(), "name": "read_file"},
        {"type": "session_complete", "timestamp": (base + timedelta(seconds=40)).isoformat(), "result_preview": "Done"},
    ]
    for event in events:
        store.append_event("test-session-1", event)

    return session, events, base


class TestUpdatedSinceFilter:
    def test_list_sessions_no_filter(self, store):
        s1 = Session(id="s1", agent="a", last_active="2026-03-15T10:00:00+00:00")
        s2 = Session(id="s2", agent="a", last_active="2026-03-15T12:00:00+00:00")
        store.create_session(s1)
        store.create_session(s2)

        result = store.list_sessions()
        assert len(result) == 2

    def test_list_sessions_updated_since(self, store):
        s1 = Session(id="s1", agent="a", last_active="2026-03-15T10:00:00+00:00")
        s2 = Session(id="s2", agent="a", last_active="2026-03-15T12:00:00+00:00")
        s3 = Session(id="s3", agent="a", last_active="2026-03-15T14:00:00+00:00")
        store.create_session(s1)
        store.create_session(s2)
        store.create_session(s3)

        result = store.list_sessions(updated_since="2026-03-15T11:00:00+00:00")
        ids = {s.id for s in result}
        assert ids == {"s2", "s3"}

    def test_list_sessions_updated_since_none_match(self, store):
        s1 = Session(id="s1", agent="a", last_active="2026-03-15T10:00:00+00:00")
        store.create_session(s1)

        result = store.list_sessions(updated_since="2026-03-15T23:00:00+00:00")
        assert len(result) == 0

    def test_list_sessions_updated_since_combined_with_status(self, store):
        s1 = Session(id="s1", agent="a", status="completed", last_active="2026-03-15T12:00:00+00:00")
        s2 = Session(id="s2", agent="a", status="running", last_active="2026-03-15T12:00:00+00:00")
        store.create_session(s1)
        store.create_session(s2)

        result = store.list_sessions(updated_since="2026-03-15T11:00:00+00:00", status="completed")
        assert len(result) == 1
        assert result[0].id == "s1"


class TestSessionEventsSince:
    def test_all_events(self, store, session_with_events):
        _, events, _ = session_with_events
        result = store.session_events_since("test-session-1")
        assert len(result) == 5

    def test_events_since_timestamp(self, store, session_with_events):
        _, events, base = session_with_events
        since = (base + timedelta(seconds=15)).isoformat()
        result = store.session_events_since("test-session-1", since=since)
        assert len(result) == 3  # 20s, 30s, 40s events
        assert result[0]["type"] == "tool_call"
        assert result[-1]["type"] == "session_complete"

    def test_events_since_future(self, store, session_with_events):
        result = store.session_events_since("test-session-1", since="2099-01-01T00:00:00+00:00")
        assert len(result) == 0

    def test_events_nonexistent_session(self, store):
        # read_events returns [] for missing sessions, so this should work
        result = store.session_events_since("nonexistent")
        assert result == []


class TestSessionSummary:
    def test_basic_summary(self, store, session_with_events):
        session, _, _ = session_with_events
        summary = store.session_summary("test-session-1")

        assert summary["id"] == "test-session-1"
        assert summary["agent"] == "odyn"
        assert summary["source"] == "background"
        assert summary["status"] == "completed"
        assert "Do something useful" in summary["prompt"]
        assert "Done doing something useful" in summary["result"]
        assert summary["event_count"] == 5
        assert sorted(summary["tools_used"]) == ["read_file", "write_file"]

    def test_summary_no_events(self, store):
        session = Session(id="empty-session", agent="test", source="background", prompt="test")
        store.create_session(session)

        summary = store.session_summary("empty-session")
        assert summary["event_count"] == 0
        assert summary["tools_used"] == []

    def test_summary_with_error(self, store):
        session = Session(
            id="error-session",
            agent="test",
            source="background",
            status="failed",
            prompt="fail task",
            error="Something broke",
        )
        store.create_session(session)

        summary = store.session_summary("error-session")
        assert summary["status"] == "failed"
        assert summary["error"] == "Something broke"

    def test_summary_nonexistent_raises(self, store):
        with pytest.raises(ValueError):
            store.session_summary("nonexistent")
