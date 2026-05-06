"""Tests for primary-session metadata flag (Tier 2 of session routing refactor)."""

import threading
import time

import pytest

from tsugite.daemon.session_store import Session, SessionSource, SessionStatus, SessionStore


@pytest.fixture
def store(tmp_path):
    return SessionStore(tmp_path / "session_store.json")


def _make_session(store, sid, user_id="u1", agent="agent-x") -> Session:
    s = Session(id=sid, agent=agent, source=SessionSource.INTERACTIVE.value, user_id=user_id)
    store.create_session(s)
    return s


def test_find_primary_returns_none_when_absent(store):
    assert store.find_primary_session("u1", "agent-x") is None


def test_set_primary_marks_session(store):
    s = _make_session(store, "s-1")
    store.set_primary_session(s.id)
    found = store.find_primary_session("u1", "agent-x")
    assert found is not None
    assert found.id == "s-1"
    assert found.metadata.get("is_primary") is True


def test_set_primary_demotes_prior_primary(store):
    a = _make_session(store, "s-a")
    b = _make_session(store, "s-b")
    store.set_primary_session(a.id)
    store.set_primary_session(b.id)

    assert store.find_primary_session("u1", "agent-x").id == "s-b"
    assert not store.get_session("s-a").metadata.get("is_primary")
    assert store.get_session("s-b").metadata.get("is_primary") is True


def test_set_primary_isolated_per_user(store):
    a = _make_session(store, "s-u1", user_id="u1")
    b = _make_session(store, "s-u2", user_id="u2")
    store.set_primary_session(a.id)
    store.set_primary_session(b.id)

    assert store.find_primary_session("u1", "agent-x").id == "s-u1"
    assert store.find_primary_session("u2", "agent-x").id == "s-u2"


def test_set_primary_isolated_per_agent(store):
    a = _make_session(store, "s-x", agent="agent-x")
    b = _make_session(store, "s-y", agent="agent-y")
    store.set_primary_session(a.id)
    store.set_primary_session(b.id)

    assert store.find_primary_session("u1", "agent-x").id == "s-x"
    assert store.find_primary_session("u1", "agent-y").id == "s-y"


def test_find_primary_skips_finished(store):
    s = _make_session(store, "s-1")
    store.set_primary_session(s.id)
    store.update_session(s.id, status=SessionStatus.COMPLETED.value)
    assert store.find_primary_session("u1", "agent-x") is None


def test_find_primary_skips_superseded(store):
    s = _make_session(store, "s-1")
    store.set_primary_session(s.id)
    successor = _make_session(store, "s-2")
    store.update_session(s.id, superseded_by=successor.id)

    assert store.find_primary_session("u1", "agent-x") is None


def test_compaction_preserves_primary_flag(store):
    s = _make_session(store, "s-1")
    store.set_primary_session(s.id)

    new_session = store.compact_session(s.id)

    assert new_session.metadata.get("is_primary") is True
    found = store.find_primary_session("u1", "agent-x")
    assert found is not None
    assert found.id == new_session.id


def test_clear_primary_removes_flag(store):
    s = _make_session(store, "s-1")
    store.set_primary_session(s.id)
    store.clear_primary_session("u1", "agent-x")

    assert store.find_primary_session("u1", "agent-x") is None
    assert not store.get_session("s-1").metadata.get("is_primary")


def test_clear_primary_no_op_when_absent(store):
    """Clearing when nothing is primary should not raise."""
    store.clear_primary_session("u1", "agent-x")


def test_set_primary_unknown_session_raises(store):
    with pytest.raises(ValueError):
        store.set_primary_session("does-not-exist")


def test_set_primary_concurrent_lock(store):
    """Two threads racing set_primary on different sessions -> exactly one wins."""
    a = _make_session(store, "s-a")
    b = _make_session(store, "s-b")
    barrier = threading.Barrier(2)

    def race(sid):
        barrier.wait()
        store.set_primary_session(sid)

    t1 = threading.Thread(target=race, args=(a.id,))
    t2 = threading.Thread(target=race, args=(b.id,))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    primaries = [s for s in [store.get_session("s-a"), store.get_session("s-b")] if s.metadata.get("is_primary")]
    assert len(primaries) == 1


def test_get_or_create_interactive_routes_to_primary(store):
    """When primary is set, get_or_create_interactive returns it regardless of _interactive_index."""
    legacy = _make_session(store, "legacy-id")
    chosen = _make_session(store, "chosen-id")
    store._interactive_index[("u1", "agent-x")] = legacy.id
    store.set_primary_session(chosen.id)

    result = store.get_or_create_interactive("u1", "agent-x")
    assert result.id == "chosen-id"


def test_get_or_create_interactive_skips_primary_if_finished(store):
    """A finished primary should NOT be returned; fall through to existing logic."""
    s = _make_session(store, "s-finished")
    store.set_primary_session(s.id)
    store.update_session(s.id, status=SessionStatus.COMPLETED.value)

    result = store.get_or_create_interactive("u1", "agent-x")
    assert result.id != s.id


def test_set_primary_picks_latest_active(store):
    """If multiple sessions have is_primary set, the most recently active one wins."""
    a = _make_session(store, "s-a")
    b = _make_session(store, "s-b")
    a.metadata["is_primary"] = True
    b.metadata["is_primary"] = True

    time.sleep(0.01)
    store.update_session(b.id, scratchpad="bumped")

    found = store.find_primary_session("u1", "agent-x")
    assert found is not None
    assert found.id == "s-b"
