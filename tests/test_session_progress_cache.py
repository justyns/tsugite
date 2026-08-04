"""Cache tests for session_progress_summary and event_count.

Both should be O(1) on the hot path: after the first read, subsequent calls
should not re-open the .jsonl file. ``append_event`` updates the cache
incrementally so a flood of session_update events on the sidebar refresh
doesn't re-parse megabytes of history per running session.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from tsugite_daemon.session_store import SessionStatus, SessionStore

from tests.history_helpers import seed_history_session
from tsugite.history import Session


@pytest.fixture
def store(tmp_path: Path, history_dir):
    return SessionStore(tmp_path / "session_store.json")


def _make_history_session(history_dir: Path, session_id: str) -> Session:
    return seed_history_session(session_id, agent="t", model="m")


def test_progress_summary_cold_then_warm(store, history_dir, jsonl_open_spy):
    sid = "warm-test"
    _make_history_session(history_dir, sid)
    store.append_event(sid, {"type": "user_input", "text": "hi", "timestamp": "2026-01-01T00:00:00+00:00"})
    store.append_event(sid, {"type": "tool_invocation", "name": "bash", "timestamp": "2026-01-01T00:00:01+00:00"})

    store.session_progress_summary(sid)  # warm the cache
    target = str(history_dir / f"{sid}.jsonl")
    jsonl_open_spy.clear()
    result = store.session_progress_summary(sid)
    assert target not in jsonl_open_spy
    assert result["tool_count"] == 1


def test_progress_updates_incrementally_on_append(store, history_dir, jsonl_open_spy):
    sid = "incr-test"
    _make_history_session(history_dir, sid)
    store.session_progress_summary(sid)  # prime cache with empty session

    target = str(history_dir / f"{sid}.jsonl")
    jsonl_open_spy.clear()
    store.append_event(sid, {"type": "tool_invocation", "name": "bash", "timestamp": "2026-01-01T00:00:00+00:00"})
    store.append_event(sid, {"type": "tool_invocation", "name": "read", "timestamp": "2026-01-01T00:00:01+00:00"})
    result = store.session_progress_summary(sid)

    # Two appends = two writes; a third open would be the (forbidden) summary re-read.
    summary_reads = [p for p in jsonl_open_spy if target in p]
    assert len(summary_reads) <= 2, f"Too many opens: {summary_reads}"
    assert result["tool_count"] == 2


def test_event_count_cold_then_warm(store, history_dir, jsonl_open_spy):
    sid = "ec-test"
    _make_history_session(history_dir, sid)
    for _ in range(3):
        store.append_event(sid, {"type": "user_input", "text": "x"})

    store.event_count(sid)  # cold
    target = str(history_dir / f"{sid}.jsonl")
    jsonl_open_spy.clear()
    count = store.event_count(sid)
    assert count == 4  # session_start + 3 appends
    assert target not in jsonl_open_spy


def test_event_count_increments_on_append(store, history_dir, jsonl_open_spy):
    sid = "ec-incr"
    _make_history_session(history_dir, sid)
    store.event_count(sid)  # prime to 1 (just session_start)
    store.append_event(sid, {"type": "info", "message": "hi"})
    target = str(history_dir / f"{sid}.jsonl")
    jsonl_open_spy.clear()
    n = store.event_count(sid)
    assert n == 2
    assert target not in jsonl_open_spy


def test_progress_session_end_resets_counts(store, history_dir):
    sid = "end-test"
    _make_history_session(history_dir, sid)
    store.append_event(sid, {"type": "tool_invocation", "name": "bash"})
    store.append_event(sid, {"type": "session_end", "status": "success"})

    summary = store.session_progress_summary(sid)
    assert summary["tool_count"] == 0
    assert summary["status_text"] == ""


def test_cache_survives_store_reload(tmp_path: Path, history_dir):
    """After a daemon restart, the cache rebuilds from the file on first call."""
    path = tmp_path / "session_store.json"
    s1 = SessionStore(path)
    sid = "reload-test"
    _make_history_session(history_dir, sid)
    s1.append_event(sid, {"type": "tool_invocation", "name": "bash"})
    s1.append_event(sid, {"type": "tool_invocation", "name": "read"})

    s2 = SessionStore(path)
    summary = s2.session_progress_summary(sid)
    assert summary["tool_count"] == 2


def test_progress_cache_evicted_on_session_finish(store, history_dir):
    """Once a session is marked COMPLETED/FAILED/CANCELLED it stops appending
    events and the sidebar stops reading its progress, so the cache entry
    just leaks memory until daemon restart. Verify it's evicted on the
    status transition."""
    from tsugite_daemon.session_store import Session

    session = store.create_session(Session(id="finish-test", agent="t", status=SessionStatus.ACTIVE.value))
    sid = session.id
    _make_history_session(history_dir, sid)
    store.append_event(sid, {"type": "tool_invocation", "name": "bash"})
    store.session_progress_summary(sid)  # populate cache
    assert sid in store._progress_cache

    store.update_session(sid, status=SessionStatus.COMPLETED.value)
    assert sid not in store._progress_cache


def _mid_turn_tail(store, sid):
    """Events ending mid-turn: a status-bearing model_request with no terminator after it."""
    store.append_event(sid, {"type": "final_result", "result": "done"})
    store.append_event(sid, {"type": "model_response", "content": "ok"})
    store.append_event(sid, {"type": "model_request", "model": "m"})
    store.append_event(sid, {"type": "model_response", "content": "ok"})


def test_progress_summary_stays_a_pure_fold_when_idle(store, history_dir):
    """A retained slice can end mid-turn (compaction drops per-turn session_end
    markers). The fold still reports the last status-bearing event - suppressing
    it is the caller's job, because liveness is not an event."""
    from tsugite_daemon.session_store import Session

    session = store.create_session(Session(id="idle-tail", agent="t", status=SessionStatus.ACTIVE.value))
    _make_history_session(history_dir, session.id)
    _mid_turn_tail(store, session.id)

    assert store.session_progress_summary(session.id)["status_text"] == "Waiting on LLM..."
    assert session.has_live_work is False


class TestSessionHasLiveWork:
    """The durable half of "is this session busy"; `_session_busy` builds on it."""

    def _session(self, **kw):
        from tsugite_daemon.session_store import Session

        return Session(id="s", agent="t", **kw)

    def test_idle_interactive_session_has_no_live_work(self):
        assert self._session(status=SessionStatus.ACTIVE.value).has_live_work is False

    def test_in_flight_turn_is_live_work(self):
        session = self._session(status=SessionStatus.ACTIVE.value)
        session.turn_in_flight = True
        assert session.has_live_work is True

    def test_running_background_session_is_live_work(self):
        """Scheduled and background runs never call begin_turn, so the turn
        marker alone would report them idle for their whole run."""
        assert self._session(status=SessionStatus.RUNNING.value).has_live_work is True

    def test_finished_session_has_no_live_work(self):
        assert self._session(status=SessionStatus.COMPLETED.value).has_live_work is False

    def test_turn_marker_wins_over_a_non_running_status(self):
        session = self._session(status=SessionStatus.COMPLETED.value)
        session.turn_in_flight = True
        assert session.has_live_work is True
