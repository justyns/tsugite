"""Cache tests for session_progress_summary and event_count.

Both should be O(1) on the hot path: after the first read, subsequent calls
should not re-open the .jsonl file. ``append_event`` updates the cache
incrementally so a flood of session_update events on the sidebar refresh
doesn't re-parse megabytes of history per running session.
"""

from __future__ import annotations

import builtins
from pathlib import Path
from unittest.mock import patch

import pytest

from tsugite.daemon.session_store import SessionStore
from tsugite.history import SessionStorage


@pytest.fixture
def history_dir(tmp_path: Path):
    h = tmp_path / "history"
    h.mkdir()
    with patch("tsugite.history.storage.get_history_dir", return_value=h):
        yield h


@pytest.fixture
def store(tmp_path: Path, history_dir):
    return SessionStore(tmp_path / "session_store.json", history_dir=history_dir)


def _make_history_session(history_dir: Path, session_id: str) -> SessionStorage:
    return SessionStorage.create(
        agent_name="t",
        model="m",
        session_path=history_dir / f"{session_id}.jsonl",
    )


class _OpenSpy:
    """Track every file opened — both via builtin open() and Path.open."""

    def __init__(self):
        self.paths: list[str] = []
        self._real_builtin = builtins.open
        self._real_path = Path.open

    def __enter__(self):
        spy = self

        def _track(p):
            try:
                spy.paths.append(str(p))
            except Exception:
                pass

        def builtin_open(file, *a, **kw):
            _track(file)
            return spy._real_builtin(file, *a, **kw)

        def path_open(self, *a, **kw):
            _track(self)
            return spy._real_path(self, *a, **kw)

        self._patch_b = patch.object(builtins, "open", builtin_open)
        self._patch_p = patch.object(Path, "open", path_open)
        self._patch_b.start()
        self._patch_p.start()
        return self

    def __exit__(self, *exc):
        self._patch_b.stop()
        self._patch_p.stop()

    def jsonl_opens(self) -> list[str]:
        return [p for p in self.paths if p.endswith(".jsonl")]


def test_progress_summary_cold_then_warm(store, history_dir):
    sid = "warm-test"
    _make_history_session(history_dir, sid)
    store.append_event(sid, {"type": "user_input", "text": "hi", "timestamp": "2026-01-01T00:00:00+00:00"})
    store.append_event(sid, {"type": "tool_invocation", "name": "bash", "timestamp": "2026-01-01T00:00:01+00:00"})

    # First call may read the file (cold). Second call MUST be cache-only.
    store.session_progress_summary(sid)  # warm the cache
    target = history_dir / f"{sid}.jsonl"

    with _OpenSpy() as spy:
        result = store.session_progress_summary(sid)

    assert str(target) not in spy.jsonl_opens(), (
        f"warm session_progress_summary opened {target} — cache miss: {spy.jsonl_opens()}"
    )
    assert result["tool_count"] == 1


def test_progress_updates_incrementally_on_append(store, history_dir):
    sid = "incr-test"
    _make_history_session(history_dir, sid)
    # Prime the cache with no events
    store.session_progress_summary(sid)

    target = history_dir / f"{sid}.jsonl"
    with _OpenSpy() as spy:
        store.append_event(sid, {"type": "tool_invocation", "name": "bash", "timestamp": "2026-01-01T00:00:00+00:00"})
        store.append_event(sid, {"type": "tool_invocation", "name": "read", "timestamp": "2026-01-01T00:00:01+00:00"})
        result = store.session_progress_summary(sid)

    # append_event itself writes (one open per append). What we forbid is the
    # progress summary call doing its own read after an append-driven update.
    summary_reads = [p for p in spy.jsonl_opens() if str(target) in p]
    # Two appends = two opens. A third would be the (forbidden) summary re-read.
    assert len(summary_reads) <= 2, f"Too many opens: {summary_reads}"
    assert result["tool_count"] == 2


def test_event_count_cold_then_warm(store, history_dir):
    sid = "ec-test"
    _make_history_session(history_dir, sid)
    for _ in range(3):
        store.append_event(sid, {"type": "user_input", "text": "x"})

    # session_start + 3 = 4
    store.event_count(sid)  # cold
    target = history_dir / f"{sid}.jsonl"
    with _OpenSpy() as spy:
        count = store.event_count(sid)
    assert count == 4
    assert str(target) not in spy.jsonl_opens()


def test_event_count_increments_on_append(store, history_dir):
    sid = "ec-incr"
    _make_history_session(history_dir, sid)
    store.event_count(sid)  # prime to 1 (just session_start)
    store.append_event(sid, {"type": "reaction", "emoji": "🎉"})
    target = history_dir / f"{sid}.jsonl"
    with _OpenSpy() as spy:
        n = store.event_count(sid)
    assert n == 2
    assert str(target) not in spy.jsonl_opens()


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
    s1 = SessionStore(path, history_dir=history_dir)
    sid = "reload-test"
    _make_history_session(history_dir, sid)
    s1.append_event(sid, {"type": "tool_invocation", "name": "bash"})
    s1.append_event(sid, {"type": "tool_invocation", "name": "read"})
    s1.flush()

    # Simulate restart with a fresh store instance
    s2 = SessionStore(path, history_dir=history_dir)
    summary = s2.session_progress_summary(sid)
    assert summary["tool_count"] == 2
