"""Regression tests for HTTPServer._collect_events.

Single-session semantics: only the requested session's events are read. The
predecessor chain (via `parent_session` in `session_start`) is intentionally
not walked — the new session's leading `compaction` event already carries the
canonical pre-compaction summary, so re-rendering predecessor events would
duplicate context the agent already received.

Two correctness axes:
- chronological order of returned events (oldest first) within the session
- limit-driven trimming applied within the session
"""

from __future__ import annotations

from pathlib import Path

from tsugite_daemon.adapters.http import HTTPServer

from tests.history_helpers import seed_history_session


def _seed_session(history_dir: Path, sid: str, *, parent: str | None = None, n_user_inputs: int = 5) -> None:
    storage = seed_history_session(sid, agent="t", model="m", parent_session=parent)
    for i in range(n_user_inputs):
        storage.record("user_input", text=f"{sid}-input-{i}")
        storage.record("model_response", raw_content=f"{sid}-resp-{i}")


def test_single_file_returns_chronological(history_dir):
    _seed_session(history_dir, "solo", n_user_inputs=3)
    events = HTTPServer._collect_events("solo", limit=0)
    user_inputs = [e for e in events if e.get("type") == "user_input"]
    assert [e["data"]["text"] for e in user_inputs] == ["solo-input-0", "solo-input-1", "solo-input-2"]


def test_predecessor_session_not_merged(history_dir):
    """Parent session's events must NOT appear in the result. The new session's
    leading `compaction` event is the canonical pre-compaction context — the
    raw predecessor events would duplicate it."""
    _seed_session(history_dir, "p", n_user_inputs=2)
    _seed_session(history_dir, "c", parent="p", n_user_inputs=2)

    events = HTTPServer._collect_events("c", limit=0)

    user_inputs = [e["data"]["text"] for e in events if e.get("type") == "user_input"]
    assert user_inputs == [
        "c-input-0",
        "c-input-1",
    ], "Only the requested session's user_inputs should appear; predecessor must not be merged"
    assert not [e for e in events if e.get("data", {}).get("text", "").startswith("p-")], (
        "No predecessor event should appear in the result"
    )


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
    """A session naming itself as parent must not send the walk into a loop."""
    storage = seed_history_session("loop", agent="t", model="m", parent_session="loop")
    storage.record("user_input", text="hi")

    events = HTTPServer._collect_events("loop", limit=0)
    user_inputs = [e for e in events if e.get("type") == "user_input"]
    assert len(user_inputs) == 1
