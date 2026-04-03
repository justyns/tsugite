"""Tests for prompt snapshot event and token breakdown."""

from tsugite.core.agent import CONTEXT_ACK, TsugiteAgent, estimate_content_tokens
from tsugite.events import EventType, PromptSnapshotEvent
from tsugite.ui.jsonl import JSONLUIHandler


class TestPromptSnapshotEvent:
    def test_serializes_correctly(self):
        event = PromptSnapshotEvent(
            token_breakdown={"system": 4, "task": 2, "total": 6},
        )
        assert event.event_type == EventType.PROMPT_SNAPSHOT
        assert event.token_breakdown["total"] == 6

    def test_defaults_to_empty(self):
        event = PromptSnapshotEvent()
        assert event.messages == []
        assert event.token_breakdown == {}


class TestEstimateContentTokens:
    def test_string(self):
        assert estimate_content_tokens("x" * 400) == 100

    def test_list_content(self):
        content = [{"type": "text", "text": "x" * 200}, {"type": "text", "text": "y" * 100}]
        assert estimate_content_tokens(content) == 75

    def test_non_string(self):
        assert estimate_content_tokens(42) == 100


class _FakeAgent:
    """Minimal stand-in for TsugiteAgent to test _compute_token_breakdown."""

    _compute_token_breakdown = TsugiteAgent._compute_token_breakdown

    def __init__(self, task=""):
        class M:
            pass

        self.memory = M()
        self.memory.task = task


class TestComputeTokenBreakdown:
    def test_system_only(self):
        messages = [{"role": "system", "content": "x" * 400}]
        result = _FakeAgent()._compute_token_breakdown(messages)
        assert result["system"] == 100
        assert result["total"] == 100

    def test_full_prompt_structure(self):
        messages = [
            {"role": "system", "content": "x" * 400},
            {"role": "user", "content": "x" * 200},
            {"role": "assistant", "content": CONTEXT_ACK},
            {"role": "user", "content": "x" * 100},
            {"role": "assistant", "content": "x" * 100},
            {"role": "user", "content": "task content here"},
            {"role": "assistant", "content": "```python\ncode\n```"},
            {"role": "user", "content": "<observation>result</observation>"},
        ]
        result = _FakeAgent(task="task content here")._compute_token_breakdown(messages)
        assert result["system"] == 100
        assert result["context"] > 0
        assert result["history"] > 0
        assert result["task"] > 0
        assert result["steps"] > 0
        assert result["total"] == sum(v for k, v in result.items() if k != "total")

    def test_no_context_turn(self):
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hello"},
        ]
        result = _FakeAgent(task="hello")._compute_token_breakdown(messages)
        assert result["context"] == 0
        assert result["history"] == 0
        assert result["task"] > 0

    def test_context_update_counted_as_context(self):
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "<context>attachments</context>"},
            {"role": "assistant", "content": CONTEXT_ACK},
            {"role": "user", "content": "<context_update>new file</context_update>"},
            {"role": "assistant", "content": "Context updated."},
            {"role": "user", "content": "my task"},
        ]
        result = _FakeAgent(task="my task")._compute_token_breakdown(messages)
        assert result["context"] > 0
        assert result["history"] > 0  # "Context updated." assistant msg


class TestJSONLHandler:
    def test_emits_prompt_snapshot(self):
        handler = JSONLUIHandler()
        emitted = []
        handler._emit = lambda t, d: emitted.append((t, d))

        event = PromptSnapshotEvent(
            token_breakdown={"system": 1, "total": 1},
        )
        handler._handle_prompt_snapshot(event)

        assert len(emitted) == 1
        assert emitted[0][0] == "prompt_snapshot"
        assert emitted[0][1]["token_breakdown"]["total"] == 1
        assert "messages" not in emitted[0][1]
