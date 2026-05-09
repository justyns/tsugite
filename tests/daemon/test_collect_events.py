"""Regression tests for HTTPServer._collect_events.

Single-file semantics: only the requested session's JSONL is read. The
predecessor chain (via `parent_session` in `session_start`) is intentionally
not walked — the new file's leading `compaction` event already carries the
canonical pre-compaction summary, so re-rendering predecessor events would
duplicate context the agent already received.

Two correctness axes:
- chronological order of returned events (oldest first) within the single file
- limit-driven trimming applied within the single file
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from tsugite.daemon.adapters.http import HTTPServer
from tsugite.history import SessionStorage


@pytest.fixture
def history_dir(tmp_path: Path):
    h = tmp_path / "history"
    h.mkdir()
    with patch("tsugite.daemon.adapters.http.get_history_dir", return_value=h):
        yield h


def _seed_session(history_dir: Path, sid: str, *, parent: str | None = None, n_user_inputs: int = 5) -> Path:
    path = history_dir / f"{sid}.jsonl"
    storage = SessionStorage.create(
        agent_name="t",
        model="m",
        session_path=path,
        parent_session=parent,
    )
    for i in range(n_user_inputs):
        storage.record("user_input", text=f"{sid}-input-{i}")
        storage.record("model_response", raw_content=f"{sid}-resp-{i}")
    return path


def test_single_file_returns_chronological(history_dir):
    _seed_session(history_dir, "solo", n_user_inputs=3)
    events = HTTPServer._collect_events("solo", limit=0)
    user_inputs = [e for e in events if e.get("type") == "user_input"]
    assert [e["data"]["text"] for e in user_inputs] == ["solo-input-0", "solo-input-1", "solo-input-2"]


def test_predecessor_file_not_read(history_dir, jsonl_open_spy):
    """Parent session's events must NOT appear in the result. The new file's
    leading `compaction` event is the canonical pre-compaction context — the
    raw predecessor events would duplicate it."""
    _seed_session(history_dir, "p", n_user_inputs=2)
    _seed_session(history_dir, "c", parent="p", n_user_inputs=2)
    jsonl_open_spy.clear()

    events = HTTPServer._collect_events("c", limit=0)

    user_inputs = [e["data"]["text"] for e in events if e.get("type") == "user_input"]
    assert user_inputs == [
        "c-input-0",
        "c-input-1",
    ], "Only the requested session's user_inputs should appear; predecessor must not be merged"
    parent_opens = [p for p in jsonl_open_spy if p.endswith("p.jsonl")]
    assert parent_opens == [], f"Predecessor file should never be opened: {jsonl_open_spy}"


def test_limit_keeps_last_n_user_inputs(history_dir):
    _seed_session(history_dir, "long", n_user_inputs=10)
    events = HTTPServer._collect_events("long", limit=3)
    user_inputs = [e["data"]["text"] for e in events if e.get("type") == "user_input"]
    assert user_inputs == ["long-input-7", "long-input-8", "long-input-9"]


def test_limit_does_not_walk_into_parent(history_dir, jsonl_open_spy):
    """Even when the newest file has fewer user_inputs than `limit`, the
    parent file is not opened. The result is just the newest file's events,
    trimmed to whatever it has."""
    _seed_session(history_dir, "old2", n_user_inputs=5)
    _seed_session(history_dir, "new2", parent="old2", n_user_inputs=2)
    jsonl_open_spy.clear()

    events = HTTPServer._collect_events("new2", limit=4)

    user_inputs = [e["data"]["text"] for e in events if e.get("type") == "user_input"]
    assert user_inputs == ["new2-input-0", "new2-input-1"]
    parent_opens = [p for p in jsonl_open_spy if p.endswith("old2.jsonl")]
    assert parent_opens == [], f"Parent file should never be opened: {jsonl_open_spy}"


def test_missing_session_returns_empty(history_dir):
    assert HTTPServer._collect_events("nope", limit=0) == []


def test_circular_parent_self_reference_handled(history_dir):
    """Defensive: a malformed file whose session_start.parent_session points
    back at itself shouldn't matter (we don't walk parents anyway), but the
    single-file read must still complete cleanly."""
    p = history_dir / "loop.jsonl"
    p.write_text(
        json.dumps({"type": "session_start", "ts": "2026-01-01T00:00:00+00:00", "data": {"parent_session": "loop"}})
        + "\n"
        + json.dumps({"type": "user_input", "ts": "2026-01-01T00:00:01+00:00", "data": {"text": "hi"}})
        + "\n"
    )
    events = HTTPServer._collect_events("loop", limit=0)
    user_inputs = [e for e in events if e.get("type") == "user_input"]
    assert len(user_inputs) == 1
