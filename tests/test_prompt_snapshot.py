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

    def __init__(self, task="", tools=None, attachments=None, skills=None):
        from types import SimpleNamespace

        self.memory = SimpleNamespace(task=task)
        self.instructions = "test instructions"
        self.tools = tools or []
        self.attachments = attachments or []
        self.skills = skills or []


def _cat(result, name):
    """Get a category dict from a breakdown result by name."""
    return next((c for c in result["categories"] if c["name"] == name), {"tokens": 0, "items": []})


class TestComputeTokenBreakdown:
    def test_has_categories_and_total(self):
        messages = [{"role": "system", "content": "x" * 400}]
        result = _FakeAgent()._compute_token_breakdown(messages)
        assert "categories" in result
        assert "total" in result
        assert result["total"] > 0

    def test_instructions_category(self):
        messages = [{"role": "system", "content": "x" * 400}]
        result = _FakeAgent()._compute_token_breakdown(messages)
        assert _cat(result, "instructions")["tokens"] > 0

    def test_tools_with_items(self):
        from unittest.mock import MagicMock

        tool = MagicMock()
        tool.name = "read_file"
        tool.to_code_prompt.return_value = "x" * 200
        messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "task"}]
        result = _FakeAgent(task="task", tools=[tool])._compute_token_breakdown(messages)
        tools_cat = _cat(result, "tools")
        assert tools_cat["tokens"] == 50
        assert len(tools_cat["items"]) == 1
        assert tools_cat["items"][0]["name"] == "read_file"

    def test_attachments_with_items(self):
        from types import SimpleNamespace

        att = SimpleNamespace(name="README.md", content="x" * 400)
        messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "task"}]
        result = _FakeAgent(task="task", attachments=[att])._compute_token_breakdown(messages)
        att_cat = _cat(result, "attachments")
        assert att_cat["tokens"] == 100
        assert att_cat["items"][0]["name"] == "README.md"

    def test_history_and_task(self):
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "x" * 200},
            {"role": "assistant", "content": CONTEXT_ACK},
            {"role": "user", "content": "x" * 100},
            {"role": "assistant", "content": "x" * 100},
            {"role": "user", "content": "my task"},
        ]
        result = _FakeAgent(task="my task")._compute_token_breakdown(messages)
        assert _cat(result, "history")["tokens"] > 0
        assert _cat(result, "task")["tokens"] > 0

    def test_total_is_sum_of_categories(self):
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "task"},
        ]
        result = _FakeAgent(task="task")._compute_token_breakdown(messages)
        assert result["total"] == sum(c["tokens"] for c in result["categories"])


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
