"""Tests for session pinning, viewing, and supersession (issues #217, #218, #219)."""

from datetime import datetime, timezone

import pytest

from tsugite.daemon.session_store import Session, SessionSource, SessionStatus, SessionStore


@pytest.fixture
def store(tmp_path):
    return SessionStore(tmp_path / "session_store.json")


def _make(store: SessionStore, *, agent: str = "agent-x", user_id: str = "u1", title: str | None = None) -> Session:
    sid = f"s-{datetime.now(timezone.utc).isoformat()}-{title or 'anon'}"
    s = Session(
        id=sid,
        agent=agent,
        source=SessionSource.INTERACTIVE.value,
        user_id=user_id,
        title=title,
    )
    store.create_session(s)
    return s


# ── Schema round-trip ──


class TestSchemaFields:
    def test_session_has_pin_and_view_fields_with_defaults(self):
        s = Session(id="x", agent="a")
        assert s.pinned is False
        assert s.pin_position is None
        assert s.last_viewed_at == ""
        assert s.superseded_by is None

    def test_pin_fields_round_trip_through_persistence(self, tmp_path):
        path = tmp_path / "session_store.json"
        store = SessionStore(path)
        s = _make(store, title="t")
        store.set_pin(s.id, True)
        store.mark_viewed(s.id)

        # Reload from disk
        store.flush()
        store2 = SessionStore(path)
        loaded = store2.get_session(s.id)
        assert loaded.pinned is True
        assert loaded.pin_position == 0
        assert loaded.last_viewed_at != ""


# ── set_pin / unpin ──


class TestSetPin:
    def test_pin_appends_at_end(self, store):
        a = _make(store, title="a")
        b = _make(store, title="b")
        c = _make(store, title="c")
        store.set_pin(a.id, True)
        store.set_pin(b.id, True)
        store.set_pin(c.id, True)
        assert store.get_session(a.id).pin_position == 0
        assert store.get_session(b.id).pin_position == 1
        assert store.get_session(c.id).pin_position == 2

    def test_unpin_densifies_remaining_positions(self, store):
        a = _make(store, title="a")
        b = _make(store, title="b")
        c = _make(store, title="c")
        store.set_pin(a.id, True)
        store.set_pin(b.id, True)
        store.set_pin(c.id, True)
        store.set_pin(b.id, False)
        assert store.get_session(a.id).pin_position == 0
        assert store.get_session(b.id).pinned is False
        assert store.get_session(b.id).pin_position is None
        assert store.get_session(c.id).pin_position == 1

    def test_pin_idempotent(self, store):
        a = _make(store, title="a")
        store.set_pin(a.id, True)
        first = store.get_session(a.id).pin_position
        store.set_pin(a.id, True)
        assert store.get_session(a.id).pin_position == first

    def test_pin_at_explicit_position(self, store):
        a = _make(store, title="a")
        b = _make(store, title="b")
        store.set_pin(a.id, True)
        store.set_pin(b.id, True, position=0)
        assert store.get_session(b.id).pin_position == 0
        assert store.get_session(a.id).pin_position == 1


# ── reorder_pins ──


class TestReorderPins:
    def test_reorder_writes_positions(self, store):
        a = _make(store, title="a")
        b = _make(store, title="b")
        c = _make(store, title="c")
        store.set_pin(a.id, True)
        store.set_pin(b.id, True)
        store.set_pin(c.id, True)
        store.reorder_pins([c.id, a.id, b.id])
        assert store.get_session(c.id).pin_position == 0
        assert store.get_session(a.id).pin_position == 1
        assert store.get_session(b.id).pin_position == 2

    def test_reorder_ignores_unknown_and_unpinned_ids(self, store):
        a = _make(store, title="a")
        b = _make(store, title="b")
        c = _make(store, title="c")  # not pinned
        store.set_pin(a.id, True)
        store.set_pin(b.id, True)
        # c is unpinned, "missing" is unknown — both should be silently skipped
        store.reorder_pins(["missing", b.id, c.id, a.id])
        assert store.get_session(b.id).pin_position == 0
        assert store.get_session(a.id).pin_position == 1
        assert store.get_session(c.id).pinned is False


# ── mark_viewed ──


class TestMarkViewed:
    def test_mark_viewed_sets_timestamp(self, store):
        a = _make(store, title="a")
        store.mark_viewed(a.id)
        ts = store.get_session(a.id).last_viewed_at
        assert ts != ""
        # ISO 8601-ish, parseable
        datetime.fromisoformat(ts)

    def test_mark_viewed_with_explicit_ts(self, store):
        a = _make(store, title="a")
        ts_in = "2026-04-25T12:00:00+00:00"
        store.mark_viewed(a.id, ts=ts_in)
        assert store.get_session(a.id).last_viewed_at == ts_in

    def test_mark_viewed_persists(self, tmp_path):
        path = tmp_path / "session_store.json"
        store = SessionStore(path)
        s = _make(store, title="t")
        store.mark_viewed(s.id, ts="2026-04-25T01:02:03+00:00")
        store.flush()
        store2 = SessionStore(path)
        assert store2.get_session(s.id).last_viewed_at == "2026-04-25T01:02:03+00:00"


# ── Compaction transfer + supersession ──


class TestCompactionTransfersPinAndTitle:
    def test_compaction_preserves_title(self, store):
        s = _make(store, title="My Topic")
        new = store.compact_session(s.id)
        assert new.title == "My Topic"

    def test_compaction_transfers_pin(self, store):
        s = _make(store, title="My Topic")
        store.set_pin(s.id, True)
        new = store.compact_session(s.id)
        assert new.pinned is True
        assert new.pin_position == 0

    def test_compaction_supersedes_old_session(self, store):
        s = _make(store, title="My Topic")
        store.set_pin(s.id, True)
        new = store.compact_session(s.id)
        old = store.get_session(s.id)
        assert old.superseded_by == new.id
        assert old.pinned is False
        assert old.pin_position is None
        assert old.status == SessionStatus.COMPLETED.value

    def test_list_sessions_filters_superseded_by_default(self, store):
        s = _make(store, title="t")
        new = store.compact_session(s.id)
        ids = [r.id for r in store.list_sessions(agent=s.agent)]
        assert new.id in ids
        assert s.id not in ids

    def test_list_sessions_can_include_superseded(self, store):
        s = _make(store, title="t")
        new = store.compact_session(s.id)
        ids = [r.id for r in store.list_sessions(agent=s.agent, include_superseded=True)]
        assert new.id in ids
        assert s.id in ids
