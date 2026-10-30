"""Tests for the event-driven TsugiteAgent loop.

Behaviors verified:
- LLM raw responses are recorded verbatim (no parser-rebuild loss)
- No-code response ends the loop and returns the text as the answer
- return_value(x) ends the loop and returns the (possibly non-string) value
- Multi-code-block response runs nothing, emits a warning, and re-prompts
- Each turn appends model_request + model_response + code_execution events
"""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from tsugite.core.agent import TsugiteAgent
from tsugite.events import EventBus, EventType
from tsugite.history import Session, get_history_backend
from tsugite.providers.base import CompletionResponse, Usage


def _resp(content: str) -> CompletionResponse:
    return CompletionResponse(content=content, usage=Usage(total_tokens=10), cost=0.0)


def _patch(agent, *, side_effect=None, return_value=None):
    mock = AsyncMock(side_effect=side_effect, return_value=return_value)
    agent._provider = MagicMock()
    agent._provider.acompletion = mock
    agent._provider.stop = AsyncMock()
    agent._provider.get_state = MagicMock(return_value=None)
    agent._provider.set_context = MagicMock()
    return mock


@pytest.fixture
def storage(tmp_path: Path) -> Session:
    return get_history_backend().create(agent_name="t", model="openai:gpt-4o-mini")


@pytest.mark.asyncio
async def test_no_code_response_ends_loop_and_returns_text(storage):
    agent = TsugiteAgent(
        model_string="openai:gpt-4o-mini",
        tools=[],
        instructions="",
        max_turns=5,
        storage=storage,
    )
    _patch(agent, return_value=_resp("Hello, this is just text."))

    result = await agent.run("hi")

    assert result == "Hello, this is just text."
    types = [e.type for e in storage.iter_events()]
    # Recording the user's message belongs to record_user_input(), which knows the
    # text the person actually typed and holds the one-per-turn guard. The loop only
    # ever sees the fully rendered prompt (tools, skills, instructions), so a
    # user_input from here would be both a duplicate and template noise in the bubble.
    assert "user_input" not in types
    assert "model_request" in types
    assert "model_response" in types
    # No code = no execution event
    assert "code_execution" not in types


@pytest.mark.asyncio
async def test_code_then_no_code_ends_loop(storage):
    agent = TsugiteAgent(
        model_string="openai:gpt-4o-mini",
        tools=[],
        instructions="",
        max_turns=5,
        storage=storage,
    )
    calls = 0

    async def side(*a, **k):
        nonlocal calls
        calls += 1
        if calls == 1:
            return _resp("Working on it.\n\n```python-exec\nx = 1\nprint(x)\n```")
        return _resp("All done. The answer is 1.")

    _patch(agent, side_effect=side)

    result = await agent.run("compute")

    assert result == "All done. The answer is 1."
    types = [e.type for e in storage.iter_events()]
    assert types.count("code_execution") == 1
    assert types.count("model_response") == 2


@pytest.mark.asyncio
async def test_return_value_ends_loop_with_structured_value(storage):
    agent = TsugiteAgent(
        model_string="openai:gpt-4o-mini",
        tools=[],
        instructions="",
        max_turns=5,
        storage=storage,
    )
    _patch(
        agent,
        return_value=_resp('```python-exec\nreturn_value({"a": 1, "b": [1,2,3]})\n```'),
    )

    result = await agent.run("structured")

    assert result == {"a": 1, "b": [1, 2, 3]}


@pytest.mark.asyncio
async def test_max_turns_returns_last_response_text(storage):
    agent = TsugiteAgent(
        model_string="openai:gpt-4o-mini",
        tools=[],
        instructions="",
        max_turns=3,
        storage=storage,
    )
    _patch(agent, return_value=_resp("Still going.\n```python-exec\nx = 1\n```"))

    result = await agent.run("loop forever")

    # max_turns reached. Last assistant text is returned (no exception).
    assert "Still going." in str(result) or result is None
    types = [e.type for e in storage.iter_events()]
    assert types.count("model_response") == 3


@pytest.mark.asyncio
async def test_multi_code_block_runs_nothing_and_warns(storage):
    """A response with several ```python-exec blocks is rejected before anything
    runs: no code_execution event, a WarningEvent, and the loop re-prompts.
    """
    event_bus = EventBus()
    warnings: list = []
    event_bus.subscribe(lambda e: warnings.append(e) if e.event_type == EventType.WARNING else None)

    agent = TsugiteAgent(
        model_string="openai:gpt-4o-mini",
        tools=[],
        instructions="",
        max_turns=5,
        storage=storage,
        event_bus=event_bus,
    )
    calls = 0

    async def side(*a, **k):
        nonlocal calls
        calls += 1
        if calls == 1:
            return _resp("```python-exec\na=1\n```\n\n```python-exec\nb=2\n```")
        return _resp("Recovered.")

    _patch(agent, side_effect=side)

    result = await agent.run("multi block")

    assert result == "Recovered."
    assert calls == 2
    assert list(storage.iter_events(types=["code_execution"])) == []
    assert any("2" in w.message and "none were executed" in w.message for w in warnings)


@pytest.mark.asyncio
async def test_multi_code_block_rejection_lands_in_next_turn_and_retry_runs(storage):
    """The next model call includes the rejection, and a single-block retry executes."""
    agent = TsugiteAgent(
        model_string="openai:gpt-4o-mini",
        tools=[],
        instructions="",
        max_turns=5,
        storage=storage,
    )
    calls = 0

    async def side(*a, **k):
        nonlocal calls
        calls += 1
        if calls == 1:
            return _resp(
                "```python-exec\na = 1\nprint(a)\n```\n\n```python-exec\nb = 2\n```\n\n```python-exec\nc = 3\n```"
            )
        if calls == 2:
            return _resp("```python-exec\na = 1\nprint(a)\n```")
        return _resp("All done.")

    provider_mock = _patch(agent, side_effect=side)

    await agent.run("multi block obs")

    turn2_call = provider_mock.call_args_list[1]
    turn2_messages = turn2_call.kwargs.get("messages") or turn2_call.args[0]
    user_msgs = [m for m in turn2_messages if m.get("role") == "user"]
    obs_text = "\n".join(str(m.get("content", "")) for m in user_msgs)

    assert '<tsugite_multi_block_warning total="3">' in obs_text
    assert "None of them were executed" in obs_text

    code_execs = list(storage.iter_events(types=["code_execution"]))
    assert len(code_execs) == 1
    assert code_execs[0].data["code"].strip() == "a = 1\nprint(a)"


@pytest.mark.asyncio
async def test_single_block_response_has_no_multi_block_warning(storage):
    """Sanity counterpart to the above: a normal one-block response must not
    sprout a multi-block warning. Keeps the fix from leaking into the common
    case.
    """
    agent = TsugiteAgent(
        model_string="openai:gpt-4o-mini",
        tools=[],
        instructions="",
        max_turns=5,
        storage=storage,
    )
    calls = 0

    async def side(*a, **k):
        nonlocal calls
        calls += 1
        if calls == 1:
            return _resp("```python-exec\nx = 1\nprint(x)\n```")
        return _resp("done")

    provider_mock = _patch(agent, side_effect=side)

    await agent.run("single block")

    turn2_call = provider_mock.call_args_list[1]
    turn2_messages = turn2_call.kwargs.get("messages") or turn2_call.args[0]
    obs_text = "\n".join(str(m.get("content", "")) for m in turn2_messages if m.get("role") == "user")
    assert "tsugite_multi_block_warning" not in obs_text, (
        f"single-block response leaked a multi-block warning: {obs_text!r}"
    )


@pytest.mark.asyncio
async def test_raw_assistant_text_recorded_verbatim(storage):
    agent = TsugiteAgent(
        model_string="openai:gpt-4o-mini",
        tools=[],
        instructions="",
        max_turns=3,
        storage=storage,
    )
    raw = "Some prose with `inline backticks` and ```python-exec\nx = 1\n```\nMore prose after."
    calls = 0

    async def side(*a, **k):
        nonlocal calls
        calls += 1
        if calls == 1:
            return _resp(raw)
        return _resp("done")

    _patch(agent, side_effect=side)

    await agent.run("verbatim test")

    responses = list(storage.iter_events(types=["model_response"]))
    assert responses[0].data["raw_content"] == raw


@pytest.mark.asyncio
async def test_provider_state_delta_recorded(storage):
    agent = TsugiteAgent(
        model_string="claude_code:opus",
        tools=[],
        instructions="",
        max_turns=2,
        storage=storage,
    )
    _patch(agent, return_value=_resp("done"))
    agent._provider.get_state = MagicMock(return_value={"session_id": "abc123", "compacted": False})

    await agent.run("hi")

    responses = list(storage.iter_events(types=["model_response"]))
    assert responses[0].data["state_delta"]["session_id"] == "abc123"
