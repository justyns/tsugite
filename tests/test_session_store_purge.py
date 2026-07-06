"""Tests for `_purge_session_state` after the parallel-map consolidation (#299).

Per-session UI/runtime state (sticky_skills, suppressed_skills, reasoning_effort,
model_override, compacting) now lives on Session itself, so deleting the session
takes that state with it. What `_purge_session_state` still cleans up:
- reverse-lookup indexes (`_thread_index`, `_channel_index` entries pointing here)
- hot caches keyed by session_id (`_progress_cache`, `_event_count_cache`)
"""

import pytest
from tsugite_daemon.session_store import Session, SessionSource, SessionStatus, SessionStore


@pytest.fixture
def store(tmp_path):
    return SessionStore(tmp_path / "session_store.json", context_limits={"test-agent": 128_000})


def _make_session(store, *, sid, agent="test-agent", source=SessionSource.BACKGROUND.value, metadata=None):
    session = Session(
        id=sid,
        agent=agent,
        source=source,
        status=SessionStatus.COMPLETED.value,
        metadata=metadata or {},
    )
    store.create_session(session)
    return session


def _populate_state(store, sid):
    """Seed every per-session field plus an index and cache entry for `sid`."""
    store.mark_sticky(sid, "skill-a")
    store.suppress_skill(sid, "skill-b")
    store.set_reasoning_effort(sid, "high")
    store.set_model_override(sid, "openai:gpt-4o-mini")
    store.begin_compaction("user1", "test-agent", session_id=sid)
    store.append_event(sid, {"type": "user_input", "timestamp": "2026-05-15T00:00:00Z", "text": "hi"})
    store.session_progress_summary(sid)  # primes _progress_cache + _event_count_cache


def _assert_clean(store, sid):
    """After purge, neither the session nor any derived index/cache reference it."""
    assert sid not in store._sessions
    assert sid not in store._thread_index.values()
    assert not any(v == sid for v in store._channel_index.values())
    assert sid not in store._progress_cache
    assert sid not in store._event_count_cache


def test_purge_session_state_removes_session_and_derived_indexes(store):
    session = _make_session(store, sid="bg-1", metadata={"thread_id": "thread-1"})
    _populate_state(store, session.id)

    assert store._thread_index.get("thread-1") == session.id
    assert session.id in store._event_count_cache

    store._purge_session_state(session.id)
    _assert_clean(store, session.id)


def test_per_session_state_lives_on_session_and_drops_with_it(store):
    """Sticky/suppressed/effort/model/compacting are now Session fields, so
    deleting the session takes them along — no parallel maps to clean."""
    session = _make_session(store, sid="bg-2")
    _populate_state(store, session.id)

    assert session.sticky_skills == {"skill-a": 0}
    assert session.suppressed_skills == ["skill-b"]
    assert session.reasoning_effort == "high"
    assert session.model_override == "openai:gpt-4o-mini"
    assert session.compacting is True

    store._purge_session_state(session.id)
    # State is gone with the session — accessors return defaults.
    assert store.get_sticky_skills(session.id) == {}
    assert store.get_suppressed_skills(session.id) == set()
    assert store.get_reasoning_effort(session.id) is None
    assert store.get_model_override(session.id) is None
    assert store.is_compacting("user1", "test-agent", session_id=session.id) is False


def test_prune_background_purges_derived_indexes(store):
    """Pruning a background session must also drop its thread/channel index entries."""
    keep_ids = [f"bg-keep-{i}" for i in range(store.MAX_BACKGROUND_SESSIONS)]
    for sid in keep_ids:
        _make_session(store, sid=sid)

    leaked = _make_session(store, sid="bg-old", metadata={"thread_id": "leaked-thread"})
    _populate_state(store, leaked.id)
    leaked.created_at = "2020-01-01T00:00:00+00:00"

    _make_session(store, sid="bg-new")

    _assert_clean(store, leaked.id)


def test_prune_schedule_purges_derived_indexes(store):
    parent = _make_session(store, sid="parent-1", source=SessionSource.INTERACTIVE.value)
    keep_ids = [f"sched-keep-{i}" for i in range(store.MAX_SCHEDULE_SESSIONS)]
    for sid in keep_ids:
        session = Session(
            id=sid,
            agent="test-agent",
            source=SessionSource.SCHEDULE.value,
            status=SessionStatus.COMPLETED.value,
            parent_id=parent.id,
        )
        store.create_session(session)

    leaked = Session(
        id="sched-old",
        agent="test-agent",
        source=SessionSource.SCHEDULE.value,
        status=SessionStatus.COMPLETED.value,
        parent_id=parent.id,
        metadata={"thread_id": "leaked-sched-thread"},
    )
    store.create_session(leaked)
    _populate_state(store, leaked.id)
    leaked.created_at = "2020-01-01T00:00:00+00:00"

    overflow = Session(
        id="sched-new",
        agent="test-agent",
        source=SessionSource.SCHEDULE.value,
        status=SessionStatus.COMPLETED.value,
        parent_id=parent.id,
    )
    store.create_session(overflow)

    _assert_clean(store, leaked.id)


def test_recover_stale_sessions_clears_compacting_on_restart(tmp_path):
    """Persisted compacting=True from a crashed daemon must reset on reload —
    the in-memory lock didn't survive, so the flag would otherwise stick forever."""
    path = tmp_path / "session_store.json"
    first = SessionStore(path)
    session = first.get_or_create_interactive("alice", "test-agent")
    first.begin_compaction("alice", "test-agent", session_id=session.id)
    assert first.get_session(session.id).compacting is True

    second = SessionStore(path)
    assert second.get_session(session.id).compacting is False
