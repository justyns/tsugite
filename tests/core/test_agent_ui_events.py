"""Tests for TsugiteAgent UI event integration."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from tsugite.core.agent import TsugiteAgent
from tsugite.events import EventBus, EventType
from tsugite.providers.base import CompletionResponse, Usage


def _resp(content: str, reasoning_content: str = None, reasoning_tokens: int = None) -> CompletionResponse:
    usage = Usage(total_tokens=100, reasoning_tokens=reasoning_tokens)
    return CompletionResponse(content=content, reasoning_content=reasoning_content, usage=usage, cost=0.001)


def _patch_provider(agent, side_effect=None, return_value=None):
    mock = AsyncMock(side_effect=side_effect, return_value=return_value)
    agent._provider = MagicMock()
    agent._provider.acompletion = mock
    agent._provider.stop = AsyncMock()
    agent._provider.get_state = MagicMock(return_value=None)
    agent._provider.set_context = MagicMock()
    return mock


@pytest.fixture
def mock_ui_handler():
    handler = MagicMock()
    handler.events = []

    def track_event(event):
        handler.events.append({"event": event.event_type, "event_obj": event})

    handler.handle_event = track_event
    return handler


@pytest.fixture
def event_bus_with_handler(mock_ui_handler):
    bus = EventBus()
    bus.subscribe(mock_ui_handler.handle_event)
    return bus, mock_ui_handler


@pytest.mark.asyncio
async def test_ui_event_task_start(event_bus_with_handler):
    event_bus, mock_ui_handler = event_bus_with_handler

    agent = TsugiteAgent(
        model_string="openai:gpt-4o-mini", tools=[], instructions="", max_turns=5,
        event_bus=event_bus, model_name="gpt-4o-mini",
    )
    _patch_provider(agent, return_value=_resp("""Thought: Calculate the answer.

```python
final_answer(42)
```"""))

    await agent.run("Test task")

    task_start_events = [e for e in mock_ui_handler.events if e["event"] == EventType.TASK_START]
    assert len(task_start_events) == 1
    assert task_start_events[0]["event_obj"].task == "Test task"
    assert task_start_events[0]["event_obj"].model == "gpt-4o-mini"


@pytest.mark.asyncio
async def test_ui_event_step_start(event_bus_with_handler):
    event_bus, mock_ui_handler = event_bus_with_handler

    call_count = 0

    async def mock_acompletion(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return _resp("""Thought: First step.

```python
x = 5
print(x)
```""")
        else:
            return _resp("""Thought: Second step.

```python
final_answer(x * 2)
```""")

    agent = TsugiteAgent(
        model_string="openai:gpt-4o-mini", tools=[], instructions="", max_turns=5,
        event_bus=event_bus,
    )
    _patch_provider(agent, side_effect=mock_acompletion)

    await agent.run("Test task")

    step_start_events = [e for e in mock_ui_handler.events if e["event"] == EventType.STEP_START]
    assert len(step_start_events) == 2
    assert step_start_events[0]["event_obj"].step == 1
    assert step_start_events[1]["event_obj"].step == 2


@pytest.mark.asyncio
async def test_ui_event_code_execution(event_bus_with_handler):
    event_bus, mock_ui_handler = event_bus_with_handler

    agent = TsugiteAgent(
        model_string="openai:gpt-4o-mini", tools=[], instructions="", max_turns=5,
        event_bus=event_bus,
    )
    _patch_provider(agent, return_value=_resp("""Thought: Execute code.

```python
result = 5 + 3
final_answer(result)
```"""))

    await agent.run("Test task")

    code_exec_events = [e for e in mock_ui_handler.events if e["event"] == EventType.CODE_EXECUTION]
    assert len(code_exec_events) == 1
    assert "result = 5 + 3" in code_exec_events[0]["event_obj"].code
    assert "final_answer(result)" in code_exec_events[0]["event_obj"].code


@pytest.mark.asyncio
async def test_ui_event_observation(event_bus_with_handler):
    event_bus, mock_ui_handler = event_bus_with_handler

    agent = TsugiteAgent(
        model_string="openai:gpt-4o-mini", tools=[], instructions="", max_turns=5,
        event_bus=event_bus,
    )
    _patch_provider(agent, return_value=_resp("""Thought: Print something.

```python
print("Hello, World!")
final_answer("done")
```"""))

    await agent.run("Test task")

    observation_events = [e for e in mock_ui_handler.events if e["event"] == EventType.OBSERVATION]
    assert len(observation_events) == 1
    assert "Hello, World!" in observation_events[0]["event_obj"].observation


@pytest.mark.asyncio
async def test_ui_event_final_answer(event_bus_with_handler):
    event_bus, mock_ui_handler = event_bus_with_handler

    agent = TsugiteAgent(
        model_string="openai:gpt-4o-mini", tools=[], instructions="", max_turns=5,
        event_bus=event_bus,
    )
    _patch_provider(agent, return_value=_resp("""Thought: Return the answer.

```python
final_answer("The answer is 42")
```"""))

    await agent.run("Test task")

    final_answer_events = [e for e in mock_ui_handler.events if e["event"] == EventType.FINAL_ANSWER]
    assert len(final_answer_events) == 1
    assert final_answer_events[0]["event_obj"].answer == "The answer is 42"


@pytest.mark.asyncio
async def test_ui_event_error_on_execution_failure(event_bus_with_handler):
    event_bus, mock_ui_handler = event_bus_with_handler

    call_count = 0

    async def mock_acompletion(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return _resp("""Thought: Try to divide by zero.

```python
result = 1 / 0
```""")
        else:
            return _resp("""Thought: Fix the error.

```python
final_answer(1)
```""")

    agent = TsugiteAgent(
        model_string="openai:gpt-4o-mini", tools=[], instructions="", max_turns=5,
        event_bus=event_bus,
    )
    _patch_provider(agent, side_effect=mock_acompletion)

    await agent.run("Test task")

    warning_events = [e for e in mock_ui_handler.events if e["event"] == EventType.WARNING]
    assert len(warning_events) == 1
    assert "ZeroDivisionError" in warning_events[0]["event_obj"].message
    assert "will retry" in warning_events[0]["event_obj"].message


@pytest.mark.asyncio
async def test_ui_event_error_on_max_turns(event_bus_with_handler):
    event_bus, mock_ui_handler = event_bus_with_handler

    agent = TsugiteAgent(
        model_string="openai:gpt-4o-mini", tools=[], instructions="", max_turns=2,
        event_bus=event_bus,
    )
    _patch_provider(agent, return_value=_resp("""Thought: Still working...

```python
x = 1
print(x)
```"""))

    with pytest.raises(RuntimeError):
        await agent.run("Test task")

    error_events = [e for e in mock_ui_handler.events if e["event"] == EventType.ERROR]
    assert len(error_events) == 1
    assert "max_turns" in error_events[0]["event_obj"].error
    assert error_events[0]["event_obj"].error_type == "RuntimeError"


@pytest.mark.asyncio
async def test_ui_event_reasoning_content(event_bus_with_handler):
    event_bus, mock_ui_handler = event_bus_with_handler

    agent = TsugiteAgent(
        model_string="anthropic:claude-3-7-sonnet", tools=[], instructions="", max_turns=5,
        event_bus=event_bus,
    )
    _patch_provider(agent, return_value=_resp(
        """Thought: Solve this.

```python
final_answer(100)
```""",
        reasoning_content="Deep thinking process here...",
    ))

    await agent.run("Test task")

    reasoning_events = [e for e in mock_ui_handler.events if e["event"] == EventType.REASONING_CONTENT]
    assert len(reasoning_events) == 1
    assert reasoning_events[0]["event_obj"].content == "Deep thinking process here..."
    assert reasoning_events[0]["event_obj"].step == 1


@pytest.mark.asyncio
async def test_ui_event_reasoning_tokens(event_bus_with_handler):
    event_bus, mock_ui_handler = event_bus_with_handler

    agent = TsugiteAgent(
        model_string="openai:o1", tools=[], instructions="", max_turns=5,
        event_bus=event_bus,
    )
    _patch_provider(agent, return_value=_resp(
        """Thought: Solve this.

```python
final_answer(100)
```""",
        reasoning_tokens=256,
    ))

    await agent.run("Test task")

    reasoning_token_events = [e for e in mock_ui_handler.events if e["event"] == EventType.REASONING_TOKENS]
    assert len(reasoning_token_events) == 1
    assert reasoning_token_events[0]["event_obj"].tokens == 256
    assert reasoning_token_events[0]["event_obj"].step == 1


@pytest.mark.asyncio
async def test_agent_without_ui_handler():
    agent = TsugiteAgent(
        model_string="openai:gpt-4o-mini", tools=[], instructions="", max_turns=5,
        event_bus=None,
    )
    _patch_provider(agent, return_value=_resp("""Thought: Calculate.

```python
final_answer(42)
```"""))

    result = await agent.run("Test task")
    assert result == 42


@pytest.mark.asyncio
async def test_ui_event_order(event_bus_with_handler):
    event_bus, mock_ui_handler = event_bus_with_handler

    agent = TsugiteAgent(
        model_string="openai:gpt-4o-mini", tools=[], instructions="", max_turns=5,
        event_bus=event_bus,
    )
    _patch_provider(agent, return_value=_resp("""Thought: Calculate.

```python
print("Hello")
final_answer(42)
```"""))

    await agent.run("Test task")

    event_types = [e["event"] for e in mock_ui_handler.events]

    assert event_types[0] == EventType.TASK_START
    assert event_types[1] == EventType.STEP_START
    assert event_types[2] == EventType.LLM_MESSAGE
    assert event_types[3] == EventType.CODE_EXECUTION
    assert event_types[4] == EventType.OBSERVATION
    assert event_types[5] == EventType.FINAL_ANSWER


@pytest.mark.asyncio
async def test_ui_event_error_on_no_code_generation(event_bus_with_handler):
    event_bus, mock_ui_handler = event_bus_with_handler

    call_count = 0

    async def mock_acompletion(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return _resp(
                """Thought: I'm thinking about this problem and how to solve it, but I'm not providing any code."""
            )
        else:
            return _resp("""Thought: Now I'll provide code.

```python
final_answer(42)
```""")

    agent = TsugiteAgent(
        model_string="openai:gpt-4o-mini", tools=[], instructions="", max_turns=5,
        event_bus=event_bus,
    )
    _patch_provider(agent, side_effect=mock_acompletion)

    await agent.run("Test task")

    error_events = [e for e in mock_ui_handler.events if e["event"] == EventType.ERROR]
    assert len(error_events) == 1
    assert "LLM did not generate code" in error_events[0]["event_obj"].error
    assert error_events[0]["event_obj"].error_type == "Format Error"
