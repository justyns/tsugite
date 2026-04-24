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
        model_string="openai:gpt-4o-mini",
        tools=[],
        instructions="",
        max_turns=5,
        event_bus=event_bus,
        model_name="gpt-4o-mini",
    )
    _patch_provider(
        agent,
        return_value=_resp("""Thought: Calculate the answer.

```python
final_answer(42)
```"""),
    )

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
state['x'] = 5
print(state['x'])
```""")
        else:
            return _resp("""Thought: Second step.

```python
final_answer(state['x'] * 2)
```""")

    agent = TsugiteAgent(
        model_string="openai:gpt-4o-mini",
        tools=[],
        instructions="",
        max_turns=5,
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
        model_string="openai:gpt-4o-mini",
        tools=[],
        instructions="",
        max_turns=5,
        event_bus=event_bus,
    )
    _patch_provider(
        agent,
        return_value=_resp("""Thought: Execute code.

```python
result = 5 + 3
final_answer(result)
```"""),
    )

    await agent.run("Test task")

    code_exec_events = [e for e in mock_ui_handler.events if e["event"] == EventType.CODE_EXECUTION]
    assert len(code_exec_events) == 1
    assert "result = 5 + 3" in code_exec_events[0]["event_obj"].code
    assert "final_answer(result)" in code_exec_events[0]["event_obj"].code


@pytest.mark.asyncio
async def test_ui_event_observation(event_bus_with_handler):
    event_bus, mock_ui_handler = event_bus_with_handler

    agent = TsugiteAgent(
        model_string="openai:gpt-4o-mini",
        tools=[],
        instructions="",
        max_turns=5,
        event_bus=event_bus,
    )
    _patch_provider(
        agent,
        return_value=_resp("""Thought: Print something.

```python
print("Hello, World!")
final_answer("done")
```"""),
    )

    await agent.run("Test task")

    observation_events = [e for e in mock_ui_handler.events if e["event"] == EventType.OBSERVATION]
    assert len(observation_events) == 1
    assert "Hello, World!" in observation_events[0]["event_obj"].observation


@pytest.mark.asyncio
async def test_ui_event_final_answer(event_bus_with_handler):
    event_bus, mock_ui_handler = event_bus_with_handler

    agent = TsugiteAgent(
        model_string="openai:gpt-4o-mini",
        tools=[],
        instructions="",
        max_turns=5,
        event_bus=event_bus,
    )
    _patch_provider(
        agent,
        return_value=_resp("""Thought: Return the answer.

```python
final_answer("The answer is 42")
```"""),
    )

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
        model_string="openai:gpt-4o-mini",
        tools=[],
        instructions="",
        max_turns=5,
        event_bus=event_bus,
    )
    _patch_provider(agent, side_effect=mock_acompletion)

    await agent.run("Test task")

    warning_events = [e for e in mock_ui_handler.events if e["event"] == EventType.WARNING]
    assert len(warning_events) == 1
    assert "ZeroDivisionError" in warning_events[0]["event_obj"].message
    assert "will retry" in warning_events[0]["event_obj"].message


@pytest.mark.asyncio
async def test_ui_event_warning_on_max_turns(event_bus_with_handler):
    event_bus, mock_ui_handler = event_bus_with_handler

    agent = TsugiteAgent(
        model_string="openai:gpt-4o-mini",
        tools=[],
        instructions="",
        max_turns=2,
        event_bus=event_bus,
    )
    _patch_provider(
        agent,
        return_value=_resp("""Thought: Still working...

```python
x = 1
print(x)
```"""),
    )

    # max_turns no longer raises — last response text is returned and a warning is emitted.
    await agent.run("Test task")

    warning_events = [e for e in mock_ui_handler.events if e["event"] == EventType.WARNING]
    assert any("max_turns" in w["event_obj"].message for w in warning_events)


@pytest.mark.asyncio
async def test_ui_event_reasoning_content(event_bus_with_handler):
    event_bus, mock_ui_handler = event_bus_with_handler

    agent = TsugiteAgent(
        model_string="anthropic:claude-3-7-sonnet",
        tools=[],
        instructions="",
        max_turns=5,
        event_bus=event_bus,
    )
    _patch_provider(
        agent,
        return_value=_resp(
            """Thought: Solve this.

```python
final_answer(100)
```""",
            reasoning_content="Deep thinking process here...",
        ),
    )

    await agent.run("Test task")

    reasoning_events = [e for e in mock_ui_handler.events if e["event"] == EventType.REASONING_CONTENT]
    assert len(reasoning_events) == 1
    assert reasoning_events[0]["event_obj"].content == "Deep thinking process here..."
    assert reasoning_events[0]["event_obj"].step == 1


@pytest.mark.asyncio
async def test_ui_event_reasoning_tokens(event_bus_with_handler):
    event_bus, mock_ui_handler = event_bus_with_handler

    agent = TsugiteAgent(
        model_string="openai:o1",
        tools=[],
        instructions="",
        max_turns=5,
        event_bus=event_bus,
    )
    _patch_provider(
        agent,
        return_value=_resp(
            """Thought: Solve this.

```python
final_answer(100)
```""",
            reasoning_tokens=256,
        ),
    )

    await agent.run("Test task")

    reasoning_token_events = [e for e in mock_ui_handler.events if e["event"] == EventType.REASONING_TOKENS]
    assert len(reasoning_token_events) == 1
    assert reasoning_token_events[0]["event_obj"].tokens == 256
    assert reasoning_token_events[0]["event_obj"].step == 1


@pytest.mark.asyncio
async def test_agent_without_ui_handler():
    agent = TsugiteAgent(
        model_string="openai:gpt-4o-mini",
        tools=[],
        instructions="",
        max_turns=5,
        event_bus=None,
    )
    _patch_provider(
        agent,
        return_value=_resp("""Thought: Calculate.

```python
final_answer(42)
```"""),
    )

    result = await agent.run("Test task")
    assert result == 42


@pytest.mark.asyncio
async def test_ui_event_order(event_bus_with_handler):
    event_bus, mock_ui_handler = event_bus_with_handler

    agent = TsugiteAgent(
        model_string="openai:gpt-4o-mini",
        tools=[],
        instructions="",
        max_turns=5,
        event_bus=event_bus,
    )
    _patch_provider(
        agent,
        return_value=_resp("""Thought: Calculate.

```python
print("Hello")
final_answer(42)
```"""),
    )

    await agent.run("Test task")

    event_types = [e["event"] for e in mock_ui_handler.events]

    assert event_types[0] == EventType.TASK_START
    assert event_types[1] == EventType.STEP_START
    assert event_types[2] == EventType.PROMPT_SNAPSHOT
    assert event_types[3] == EventType.LLM_MESSAGE
    assert event_types[4] == EventType.PROMPT_SNAPSHOT  # post-LLM snapshot with response
    assert event_types[5] == EventType.CODE_EXECUTION
    assert event_types[6] == EventType.OBSERVATION
    assert event_types[7] == EventType.FINAL_ANSWER


@pytest.mark.asyncio
async def test_llm_message_does_not_contain_code_fence(event_bus_with_handler):
    """When the LLM response is code-only (no prose thought), the emitted
    LLMMessageEvent must NOT contain the raw markdown code fence.  Otherwise
    the UI renders the code block twice: once inside the "thought" markdown,
    and once as a separate code-execution event.
    """
    event_bus, mock_ui_handler = event_bus_with_handler

    agent = TsugiteAgent(
        model_string="openai:gpt-4o-mini",
        tools=[],
        instructions="",
        max_turns=5,
        event_bus=event_bus,
    )
    _patch_provider(
        agent,
        return_value=_resp("```python\nfinal_answer(42)\n```"),
    )

    await agent.run("Test task")

    llm_message_events = [
        e["event_obj"] for e in mock_ui_handler.events if e["event"] == EventType.LLM_MESSAGE
    ]
    for ev in llm_message_events:
        assert "```python" not in ev.content, (
            f"LLMMessageEvent carried a code fence as its thought content. "
            f"This causes the UI to render the code block twice. content={ev.content!r}"
        )


@pytest.mark.asyncio
async def test_ui_event_warning_on_multiple_code_blocks(event_bus_with_handler):
    """If the LLM emits two ```python blocks in one response, the agent executes
    the first block (parser already does this) and emits a WarningEvent noting
    the extras were dropped. It does NOT reject the turn or emit an ErrorEvent —
    that behavior burned a retry per multi-block response with no upside, since
    the parser was never going to run the extras anyway.
    """
    event_bus, mock_ui_handler = event_bus_with_handler

    async def mock_acompletion(*args, **kwargs):
        return _resp(
            "Thought: doing work and then replying.\n\n"
            "```python\n"
            "final_answer(1)\n"
            "```\n\n"
            "```python\n"
            "# this block will be dropped\n"
            "x = 99\n"
            "```"
        )

    agent = TsugiteAgent(
        model_string="openai:gpt-4o-mini",
        tools=[],
        instructions="",
        max_turns=5,
        event_bus=event_bus,
    )
    _patch_provider(agent, side_effect=mock_acompletion)

    await agent.run("Test task")

    error_events = [e for e in mock_ui_handler.events if e["event"] == EventType.ERROR]
    format_errors = [e for e in error_events if getattr(e["event_obj"], "error_type", None) == "Format Error"]
    assert not format_errors, f"expected no Format Error events, got: {format_errors}"

    warning_events = [e for e in mock_ui_handler.events if e["event"] == EventType.WARNING]
    multi_block_warnings = [e for e in warning_events if "block" in e["event_obj"].message.lower()]
    assert len(multi_block_warnings) >= 1, (
        f"expected a warning mentioning dropped blocks. got warnings: "
        f"{[e['event_obj'].message for e in warning_events]}"
    )
    assert "2" in multi_block_warnings[0]["event_obj"].message, (
        f"warning should mention the count. got: {multi_block_warnings[0]['event_obj'].message!r}"
    )


@pytest.mark.asyncio
async def test_no_code_response_ends_loop_cleanly(event_bus_with_handler):
    """A no-code response is now the answer; no error event is emitted."""
    event_bus, mock_ui_handler = event_bus_with_handler

    agent = TsugiteAgent(
        model_string="openai:gpt-4o-mini",
        tools=[],
        instructions="",
        max_turns=5,
        event_bus=event_bus,
    )
    _patch_provider(
        agent,
        return_value=_resp("Just a plain text answer with no code block."),
    )

    result = await agent.run("Test task")

    assert result == "Just a plain text answer with no code block."
    error_events = [e for e in mock_ui_handler.events if e["event"] == EventType.ERROR]
    assert error_events == []
