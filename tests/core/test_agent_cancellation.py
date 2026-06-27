"""Cooperative cancellation of the agent loop.

Regression coverage for the web-UI Stop button: clicking Stop must halt the
agent's actual work, not just tear down the SSE stream. The agent loop checks a
bound cancel Event at two checkpoints - between turns and before each code
execution - and exits cleanly with a ``cancelled`` status.
"""

import threading
from unittest.mock import AsyncMock, MagicMock

import pytest

from tsugite.cancellation import is_cancelled, set_cancel_event
from tsugite.core.agent import AgentResult, TsugiteAgent
from tsugite.providers.base import CompletionResponse, Usage

# Code that never calls final_answer, so the loop would run to max_turns absent
# cancellation.
LOOPING_RESPONSE = """Thought: keep working

```python
print("tick")
```"""


def _mock_response(content: str) -> CompletionResponse:
    return CompletionResponse(content=content, usage=Usage(total_tokens=10), cost=0.0)


def _patch_provider(agent, side_effect):
    mock = AsyncMock(side_effect=side_effect)
    agent._provider = MagicMock()
    agent._provider.acompletion = mock
    agent._provider.stop = AsyncMock()
    agent._provider.get_state = MagicMock(return_value=None)
    agent._provider.set_context = MagicMock()
    return mock


def _make_agent(max_turns: int = 10) -> TsugiteAgent:
    return TsugiteAgent(model_string="openai:gpt-4o-mini", tools=[], instructions="", max_turns=max_turns)


@pytest.fixture(autouse=True)
def _isolate_cancel_event():
    """ContextVar isolation so a bound Event never leaks across tests."""
    set_cancel_event(None)
    yield
    set_cancel_event(None)


def test_is_cancelled_reflects_bound_event():
    assert is_cancelled() is False
    event = threading.Event()
    set_cancel_event(event)
    assert is_cancelled() is False
    event.set()
    assert is_cancelled() is True


@pytest.mark.asyncio
async def test_cancel_before_first_turn_runs_no_model_calls():
    """An already-set Event stops the loop before the first model call."""
    agent = _make_agent()
    mock = _patch_provider(agent, side_effect=lambda *a, **k: _mock_response(LOOPING_RESPONSE))

    event = threading.Event()
    event.set()
    set_cancel_event(event)

    result = await agent.run("go", return_full_result=True)

    assert mock.call_count == 0
    assert isinstance(result, AgentResult)
    assert "ancel" in (result.error or "")


@pytest.mark.asyncio
async def test_cancel_between_tool_calls_stops_before_executing_code():
    """Event set while the model responds → loop bails before running that turn's code."""
    agent = _make_agent()
    event = threading.Event()

    async def respond(*args, **kwargs):
        event.set()  # user clicks Stop mid-generation
        return _mock_response(LOOPING_RESPONSE)

    mock = _patch_provider(agent, side_effect=respond)
    set_cancel_event(event)

    result = await agent.run("go", return_full_result=True)

    assert mock.call_count == 1  # one model call happened...
    assert len(agent.memory.steps) == 0  # ...but its code never executed
    assert "ancel" in (result.error or "")


@pytest.mark.asyncio
async def test_cancel_between_turns_stops_before_next_turn():
    """Event set after the first turn's code runs → loop bails before the next model call."""
    agent = _make_agent()
    event = threading.Event()
    mock = _patch_provider(agent, side_effect=lambda *a, **k: _mock_response(LOOPING_RESPONSE))

    real_execute = agent.executor.execute
    executions = {"n": 0}

    async def wrapped_execute(code):
        result = await real_execute(code)
        executions["n"] += 1
        if executions["n"] == 1:
            event.set()  # Stop clicked between turns
        return result

    agent.executor.execute = wrapped_execute
    set_cancel_event(event)

    result = await agent.run("go", return_full_result=True)

    assert mock.call_count == 1  # the second turn's model call never happened
    assert len(agent.memory.steps) == 1  # exactly one turn's code ran
    assert "ancel" in (result.error or "")


@pytest.mark.asyncio
async def test_no_cancel_runs_to_max_turns():
    """Sanity: without a cancel Event the loop is unaffected (runs to max_turns)."""
    agent = _make_agent(max_turns=3)
    mock = _patch_provider(agent, side_effect=lambda *a, **k: _mock_response(LOOPING_RESPONSE))

    result = await agent.run("go", return_full_result=True)

    assert mock.call_count == 3
    assert "max_turns" in (result.error or "")
