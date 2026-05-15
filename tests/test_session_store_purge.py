"""Tests for parallel-map cleanup on session removal (umbrella #299).

`_prune_schedule_sessions` and `_prune_background_sessions` currently delete
the session from `_sessions` but leave entries in the parallel maps
(`_thread_index`, `_channel_index`, `_sticky_skills`, `_suppressed_skills`,
`_reasoning_effort`, `_model_overrides`, `_compacting_session_ids`, hot
caches) — a slow leak over the daemon's uptime.

The fix consolidates cleanup into a single `_purge_session_state(session_id)`
helper so every removal path can call one method and not forget a map.
"""

import pytest

from tsugite.daemon.session_store import Session, SessionSource, SessionStatus, SessionStore


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


def _populate_parallel_state(store, sid):
    """Set every per-session map for `sid` so the cleanup assertions cover them all."""
    store.mark_sticky(sid, "skill-a")
    store.suppress_skill(sid, "skill-b")
    store.set_reasoning_effort(sid, "high")
    store.set_model_override(sid, "openai:gpt-4o-mini")
    store.begin_compaction("user1", "test-agent", session_id=sid)
    store.append_event(sid, {"type": "user_input", "timestamp": "2026-05-15T00:00:00Z", "text": "hi"})
    store.session_progress_summary(sid)  # primes _progress_cache + _event_count_cache


def _assert_clean(store, sid):
    assert sid not in store._sessions
    assert sid not in store._sticky_skills
    assert sid not in store._suppressed_skills
    assert sid not in store._reasoning_effort
    assert sid not in store._model_overrides
    assert sid not in store._compacting_session_ids
    assert sid not in store._progress_cache
    assert sid not in store._event_count_cache
    assert sid not in store._thread_index.values()
    assert not any(v == sid for v in store._channel_index.values())


def test_purge_session_state_clears_every_parallel_map(store):
    """Direct unit test for the purge helper."""
    session = _make_session(store, sid="bg-1", metadata={"thread_id": "thread-1"})
    _populate_parallel_state(store, session.id)

    assert session.id in store._sticky_skills  # sanity: populated
    assert session.id in store._compacting_session_ids
    assert store._thread_index.get("thread-1") == session.id

    store._purge_session_state(session.id)
    _assert_clean(store, session.id)


def test_prune_background_purges_parallel_maps(store):
    """Pruning a background session must not leak its routing/state entries."""
    keep_ids = [f"bg-keep-{i}" for i in range(store.MAX_BACKGROUND_SESSIONS)]
    for sid in keep_ids:
        _make_session(store, sid=sid)

    leaked = _make_session(store, sid="bg-old", metadata={"thread_id": "leaked-thread"})
    _populate_parallel_state(store, leaked.id)
    # Create_session bumps last_active; force this one to look like the oldest.
    leaked.created_at = "2020-01-01T00:00:00+00:00"

    _make_session(store, sid="bg-new")

    _assert_clean(store, leaked.id)


def test_prune_schedule_purges_parallel_maps(store):
    """Pruning a schedule child session must purge too."""
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
    _populate_parallel_state(store, leaked.id)
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
