"""Tests for prompt snapshot event and token breakdown."""

from pathlib import Path

import pytest

from tsugite.core.agent import CONTEXT_ACK, TsugiteAgent, estimate_content_tokens
from tsugite.events import EventType, PromptSnapshotEvent
from tsugite.history import SessionStorage
from tsugite.providers.base import CompletionResponse, Usage
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

    def __init__(self, task="", tools=None, attachments=None, skills=None, hook_vars=None):
        from types import SimpleNamespace

        self.memory = SimpleNamespace(task=task)
        self.instructions = "test instructions"
        self.tools = tools or []
        self.attachments = attachments or []
        self.skills = skills or []
        self.hook_vars = hook_vars or {}


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


def _resp(content: str) -> CompletionResponse:
    return CompletionResponse(content=content, usage=Usage(total_tokens=10), cost=0.0)


def _patch(agent, *, side_effect=None, return_value=None):
    from unittest.mock import AsyncMock, MagicMock

    mock = AsyncMock(side_effect=side_effect, return_value=return_value)
    agent._provider = MagicMock()
    agent._provider.acompletion = mock
    agent._provider.stop = AsyncMock()
    agent._provider.get_state = MagicMock(return_value=None)
    agent._provider.set_context = MagicMock()
    return mock


def _agent(tmp_path: Path, **kw) -> tuple[TsugiteAgent, SessionStorage]:
    storage = SessionStorage.create(agent_name="t", model="openai:gpt-4o-mini", session_path=tmp_path / "s.jsonl")
    agent = TsugiteAgent(
        model_string="openai:gpt-4o-mini",
        tools=[],
        instructions="be a helpful assistant",
        max_turns=kw.pop("max_turns", 3),
        storage=storage,
        **kw,
    )
    return agent, storage


def _snaps(storage):
    return [e for e in storage.iter_events() if e.type == "prompt_snapshot"]


class TestPromptSnapshotDurability:
    """The breakdown must be durable, not live-only. Scheduled / subprocess /
    restarted-daemon sessions run with no live SSE handler (event_bus is None),
    so the agent itself must record the snapshot - otherwise the inspector only
    ever works for the interactive chat that happened to have an SSE persist
    path attached ("only works for live chats").
    """

    @pytest.mark.asyncio
    async def test_headless_run_persists_a_real_breakdown(self, tmp_path: Path):
        agent, storage = _agent(tmp_path)
        assert agent.event_bus is None  # headless: no live SSE handler
        _patch(agent, return_value=_resp("just prose, no code"))

        await agent.run("summarize the repo")

        snaps = _snaps(storage)
        assert len(snaps) >= 1
        bd = snaps[-1].data["token_breakdown"]
        assert bd["total"] > 0
        assert "categories" in bd
        # Carries the turn index for the inspector's staleness readout.
        assert snaps[-1].data["turn"] == 0

    @pytest.mark.asyncio
    async def test_one_snapshot_per_turn_matches_model_requests(self, tmp_path: Path):
        # "stopped working often": the live SSE persist path is lossy (a real
        # session logged 37 model_requests but only 27 snapshots). The durable
        # channel must record exactly one snapshot per turn, like model_request.
        agent, storage = _agent(tmp_path)
        _patch(
            agent,
            side_effect=[
                _resp('```python-exec\nprint("working")\n```'),
                _resp("all done"),
            ],
        )

        await agent.run("do a two-turn task")

        reqs = [e for e in storage.iter_events() if e.type == "model_request"]
        assert len(reqs) == 2
        assert len(_snaps(storage)) == len(reqs)

    @pytest.mark.asyncio
    async def test_breakdown_failure_never_crashes_the_turn(self, tmp_path: Path):
        # A breakdown computation error must be swallowed (logged), never crash
        # the run and never persist a bogus empty snapshot.
        from unittest.mock import MagicMock

        agent, storage = _agent(tmp_path)
        agent._compute_token_breakdown = MagicMock(side_effect=RuntimeError("boom"))
        _patch(agent, return_value=_resp("answer"))

        await agent.run("task")

        finals = [e for e in storage.iter_events() if e.type == "final_result"]
        assert len(finals) == 1  # the run completed
        assert _snaps(storage) == []  # no bogus snapshot recorded


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
