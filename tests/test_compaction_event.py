"""Tests for compaction emitting an event into the new session JSONL."""

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from tsugite_daemon.memory import (
    _llm_complete,
    extract_file_paths_from_events,
    split_events_for_compaction,
    track_compaction_usage,
)

from tsugite.history import SessionStorage, events_to_messages
from tsugite.history.models import Event
from tsugite.providers.base import CompletionResponse, Usage


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

    def test_all_turns_fit_returns_empty_old_despite_prefix(self):
        """When every turn fits the retention budget there is nothing to
        summarize, even though a session_start/compaction prefix precedes the
        first user_input. Must return empty old_events so the caller's
        `if not old_events` gate skips the no-op compaction — otherwise an
        unchanged session is re-rotated (and re-summarized) on every scheduled
        run, collapsing its timestamps and burning tokens.
        """
        events = [
            _ev("session_start", agent="a", model="openai:gpt-4o-mini"),
            _ev("compaction", summary="prior conversation"),
            _ev("user_input", text="one"),
            _ev("model_response", raw_content="r1"),
            _ev("user_input", text="two"),
            _ev("model_response", raw_content="r2"),
            _ev("user_input", text="three"),
            _ev("model_response", raw_content="r3"),
        ]
        old, recent = split_events_for_compaction(events, "openai:gpt-4o-mini", retention_budget_tokens=1_000_000)
        assert old == []
        assert recent == events


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
        assert msgs[2]["role"] == "user"
        assert msgs[2]["content"].endswith("] new")


class TestCompactionSummaryDateRange:
    """The <previous_conversation> block tells the agent what time period the
    summary covers and when the compaction itself happened. After multiple
    compactions in a long-running conversation this is the only way to anchor
    recalled facts in time.
    """

    def test_summary_block_includes_compacted_range(self):
        compaction_ts = datetime(2026, 5, 4, 8, 15, tzinfo=timezone.utc)
        events = [
            Event(
                type="compaction",
                ts=compaction_ts,
                data={
                    "summary": "we discussed a refactor",
                    "range_start": "2026-04-27T11:26:00+00:00",
                    "range_end": "2026-05-01T14:30:00+00:00",
                },
            ),
            _ev("user_input", text="continuing"),
        ]
        msgs = events_to_messages(events)
        block = msgs[0]["content"]
        assert "2026-04-27" in block
        assert "2026-05-01" in block
        assert "2026-05-04" in block  # compaction time
        assert "we discussed a refactor" in block

    def test_summary_block_omits_range_when_missing(self):
        # Pre-existing JSONLs from before this change won't have range_start /
        # range_end. The block must still render gracefully.
        events = [
            Event(
                type="compaction",
                ts=datetime(2026, 5, 4, tzinfo=timezone.utc),
                data={"summary": "old-format compaction"},
            ),
            _ev("user_input", text="continuing"),
        ]
        msgs = events_to_messages(events)
        block = msgs[0]["content"]
        assert "old-format compaction" in block
        # Should not crash and should not produce empty/None placeholders
        assert "None" not in block


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


class TestCompactionUsageTracking:
    @pytest.mark.asyncio
    async def test_llm_complete_accumulates_usage_only_inside_block(self):
        """Compaction summarization token usage (previously discarded) is summed
        across calls inside `track_compaction_usage`, and calls outside the block
        are not tracked.
        """
        provider = AsyncMock()
        provider.acompletion = AsyncMock(
            return_value=CompletionResponse(content="ok", usage=Usage(prompt_tokens=100, completion_tokens=25))
        )

        with patch(
            "tsugite.models.get_provider_and_model",
            return_value=("openai:gpt-4o-mini", provider, "gpt-4o-mini"),
        ):
            with track_compaction_usage() as usage:
                await _llm_complete("sys", "u1", "openai:gpt-4o-mini")
                await _llm_complete("sys", "u2", "openai:gpt-4o-mini")
            # Outside the block the accumulator is cleared, so this call is untracked.
            await _llm_complete("sys", "u3", "openai:gpt-4o-mini")

        assert usage == {"prompt_tokens": 200, "completion_tokens": 50, "calls": 2, "cost": None}
