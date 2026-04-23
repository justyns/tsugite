"""Tests for compaction emitting an event into the new session JSONL."""

from datetime import datetime, timezone
from pathlib import Path

from tsugite.daemon.memory import (
    extract_file_paths_from_events,
    split_events_for_compaction,
)
from tsugite.history import SessionStorage, events_to_messages
from tsugite.history.models import Event


def _ev(type_: str, **data) -> Event:
    return Event(type=type_, ts=datetime.now(timezone.utc), data=data)


class TestSplitEventsForCompaction:
    def test_empty_events_returns_empty_pair(self):
        assert split_events_for_compaction([], "openai:gpt-4o-mini", 1000) == ([], [])

    def test_few_turns_keeps_all(self):
        events = [
            _ev("user_input", text="hi"),
            _ev("model_response", raw_content="hello"),
        ]
        old, recent = split_events_for_compaction(events, "openai:gpt-4o-mini", 1000)
        assert old == []
        assert len(recent) == 2

    def test_splits_when_many_turns(self):
        events = []
        for i in range(10):
            events.append(_ev("user_input", text=f"msg {i}"))
            events.append(_ev("model_response", raw_content=f"reply {i}"))
        # Tiny budget so most turns get summarized
        old, recent = split_events_for_compaction(events, "openai:gpt-4o-mini", retention_budget_tokens=10)
        assert len(old) > 0
        assert len(recent) > 0
        # Both halves split on a user_input boundary
        if recent:
            assert recent[0].type == "user_input"

    def test_min_retained_floor(self):
        events = []
        for i in range(5):
            events.append(_ev("user_input", text="x" * 1000))
            events.append(_ev("model_response", raw_content="y" * 1000))
        # Budget = 0 — should still keep MIN_RETAINED_TURNS
        old, recent = split_events_for_compaction(events, "openai:gpt-4o-mini", retention_budget_tokens=0)
        # Recent count includes whole turn(s)
        recent_user_inputs = sum(1 for e in recent if e.type == "user_input")
        assert recent_user_inputs >= 2


class TestExtractFilePaths:
    def test_finds_paths_in_event_data(self):
        events = [
            _ev("code_execution", code='read_file(path="/tmp/foo.txt")'),
            _ev("model_response", raw_content='Looking at file_path="/etc/hosts"'),
        ]
        paths = extract_file_paths_from_events(events)
        assert "/tmp/foo.txt" in paths
        assert "/etc/hosts" in paths


class TestEventsToMessagesAfterCompaction:
    def test_compaction_event_replaces_prior_events(self):
        events = [
            _ev("user_input", text="old"),
            _ev("model_response", raw_content="old reply"),
            _ev("compaction", summary="we worked on cats", retained_count=0),
            _ev("user_input", text="new"),
            _ev("model_response", raw_content="new reply"),
        ]
        msgs = events_to_messages(events)
        # Pre-compaction events drop out; the synthetic summary appears.
        assert "we worked on cats" in msgs[0]["content"]
        assert msgs[2] == {"role": "user", "content": "new"}


class TestCompactionWritesEventInPlace:
    def test_new_session_first_body_event_is_compaction(self, tmp_path: Path):
        """A compacted session's JSONL begins with session_start, then a
        compaction event whose data carries the summary, then the retained
        events."""
        new_session_path = tmp_path / "post.jsonl"
        storage = SessionStorage.create(
            agent_name="agent",
            model="openai:gpt-4o-mini",
            parent_session="old-session-id",
            session_path=new_session_path,
        )
        storage.record(
            "compaction",
            summary="prior conversation about deployment",
            replaced_count=10,
            retained_count=2,
            reason="token_threshold",
        )
        # Retained events copied over verbatim
        storage.record("user_input", text="last user msg")
        storage.record("model_response", raw_content="last assistant reply")

        events = storage.load_events()
        assert events[0].type == "session_start"
        assert events[0].data["parent_session"] == "old-session-id"
        assert events[1].type == "compaction"
        assert events[1].data["summary"] == "prior conversation about deployment"
        assert events[1].data["replaced_count"] == 10
        assert events[1].data["retained_count"] == 2
        assert events[2].type == "user_input"
        assert events[3].type == "model_response"
