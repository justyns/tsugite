"""Tests for the event-driven TsugiteAgent loop.

Behaviors verified:
- LLM raw responses are recorded verbatim (no parser-rebuild loss)
- No-code response ends the loop and returns the text as the answer
- return_value(x) ends the loop and returns the (possibly non-string) value
- Multi-code-block response is rejected as a format_error event, loop continues
- Each turn appends model_request + model_response + (code_execution|format_error) events
"""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from tsugite.core.agent import TsugiteAgent
from tsugite.history import SessionStorage
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
def storage(tmp_path: Path) -> SessionStorage:
    return SessionStorage.create(agent_name="t", model="openai:gpt-4o-mini", session_path=tmp_path / "s.jsonl")


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
    assert "user_input" in types
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
            return _resp("Working on it.\n\n```python\nx = 1\nprint(x)\n```")
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
        return_value=_resp('```python\nreturn_value({"a": 1, "b": [1,2,3]})\n```'),
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
    _patch(agent, return_value=_resp("Still going.\n```python\nx = 1\n```"))

    result = await agent.run("loop forever")

    # max_turns reached. Last assistant text is returned (no exception).
    assert "Still going." in str(result) or result is None
    types = [e.type for e in storage.iter_events()]
    assert types.count("model_response") == 3


@pytest.mark.asyncio
async def test_multi_code_block_emits_format_error_and_continues(storage):
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
            return _resp("```python\na=1\n```\n\n```python\nb=2\n```")
        return _resp("Recovered.")

    _patch(agent, side_effect=side)

    result = await agent.run("multi block")

    assert result == "Recovered."
    types = [e.type for e in storage.iter_events()]
    assert "format_error" in types


@pytest.mark.asyncio
async def test_raw_assistant_text_recorded_verbatim(storage):
    agent = TsugiteAgent(
        model_string="openai:gpt-4o-mini",
        tools=[],
        instructions="",
        max_turns=3,
        storage=storage,
    )
    raw = "Some prose with `inline backticks` and ```python\nx = 1\n```\nMore prose after."
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
