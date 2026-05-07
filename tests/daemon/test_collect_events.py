"""Regression tests for HTTPServer._collect_events.

Two correctness axes:
- chronological order of returned events (oldest first, with parent chain merged)
- limit-driven short-circuit: when the newest file has enough user_inputs to
  satisfy `limit`, parent chain files must not be fully read.
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


def test_chain_merged_in_chronological_order(history_dir):
    _seed_session(history_dir, "p", n_user_inputs=2)
    _seed_session(history_dir, "c", parent="p", n_user_inputs=2)
    events = HTTPServer._collect_events("c", limit=0)
    user_inputs = [e["data"]["text"] for e in events if e.get("type") == "user_input"]
    assert user_inputs == ["p-input-0", "p-input-1", "c-input-0", "c-input-1"]


def test_limit_keeps_last_n_user_inputs(history_dir):
    _seed_session(history_dir, "long", n_user_inputs=10)
    events = HTTPServer._collect_events("long", limit=3)
    user_inputs = [e["data"]["text"] for e in events if e.get("type") == "user_input"]
    assert user_inputs == ["long-input-7", "long-input-8", "long-input-9"]


def test_limit_short_circuits_chain_walk(history_dir, jsonl_open_spy):
    """When the newest file's user_input count >= limit, the parent file must
    not be opened. This is the user-reported case: legacy chained sessions
    shouldn't pull old files into memory when only the recent tail is needed."""
    _seed_session(history_dir, "old", n_user_inputs=5)
    _seed_session(history_dir, "new", parent="old", n_user_inputs=10)
    jsonl_open_spy.clear()

    events = HTTPServer._collect_events("new", limit=3)

    user_inputs = [e["data"]["text"] for e in events if e.get("type") == "user_input"]
    assert user_inputs == ["new-input-7", "new-input-8", "new-input-9"]
    parent_opens = [p for p in jsonl_open_spy if p.endswith("old.jsonl")]
    assert parent_opens == [], f"Parent file should not be opened when limit covers newest: {jsonl_open_spy}"


def test_limit_walks_into_parent_when_newest_too_short(history_dir):
    """If the newest file has fewer user_inputs than limit, parent must be read
    to fill the remainder."""
    _seed_session(history_dir, "old2", n_user_inputs=5)
    _seed_session(history_dir, "new2", parent="old2", n_user_inputs=2)
    events = HTTPServer._collect_events("new2", limit=4)
    user_inputs = [e["data"]["text"] for e in events if e.get("type") == "user_input"]
    assert user_inputs == ["old2-input-3", "old2-input-4", "new2-input-0", "new2-input-1"]


def test_missing_session_returns_empty(history_dir):
    assert HTTPServer._collect_events("nope", limit=0) == []


def test_circular_parent_does_not_loop(history_dir):
    """Defensive: malformed data with a cycle shouldn't infinite-loop."""
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
