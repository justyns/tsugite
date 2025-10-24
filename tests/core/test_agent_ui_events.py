"""Tests for TsugiteAgent UI event integration."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tsugite.core.agent import TsugiteAgent
from tsugite.ui import UIEvent


@pytest.fixture
def mock_ui_handler():
    """Create a mock UI handler that tracks events."""
    handler = MagicMock()
    handler.events = []

    def track_event(event, data):
        handler.events.append({"event": event, "data": data})

    handler.handle_event = track_event
    return handler


@pytest.fixture
def mock_litellm_response():
    """Create a mock LiteLLM response."""

    def _create_response(content: str, reasoning_content: str = None):
        """Factory to create mock responses with different content."""
        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message = MagicMock(spec=[])
        response.choices[0].message.content = content

        if reasoning_content:
            response.choices[0].message.reasoning_content = reasoning_content

        # Create usage object with spec to prevent auto-creation of attributes
        response.usage = MagicMock(spec=["total_tokens"])
        response.usage.total_tokens = 100

        return response

    return _create_response


@pytest.mark.asyncio
async def test_ui_event_task_start(mock_ui_handler, mock_litellm_response):
    """Test that TASK_START event is triggered when agent starts."""

    with patch("tsugite.core.agent.litellm") as mock_litellm:
        mock_litellm.acompletion = AsyncMock(
            return_value=mock_litellm_response(
                """Thought: Calculate the answer.

```python
final_answer(42)
```"""
            )
        )

        agent = TsugiteAgent(
            model_string="openai:gpt-4o-mini",
            tools=[],
            instructions="",
            max_turns=5,
            ui_handler=mock_ui_handler,
            model_name="gpt-4o-mini",
        )

        await agent.run("Test task")

        # Verify TASK_START was triggered
        task_start_events = [e for e in mock_ui_handler.events if e["event"] == UIEvent.TASK_START]
        assert len(task_start_events) == 1
        assert task_start_events[0]["data"]["task"] == "Test task"
        assert task_start_events[0]["data"]["model"] == "gpt-4o-mini"


@pytest.mark.asyncio
async def test_ui_event_step_start(mock_ui_handler, mock_litellm_response):
    """Test that STEP_START event is triggered for each step."""

    call_count = 0

    async def mock_acompletion(*args, **kwargs):
        nonlocal call_count
        call_count += 1

        if call_count == 1:
            return mock_litellm_response(
                """Thought: First step.

```python
x = 5
print(x)
```"""
            )
        elif call_count == 2:
            return mock_litellm_response(
                """Thought: Second step.

```python
final_answer(x * 2)
```"""
            )

    with patch("tsugite.core.agent.litellm") as mock_litellm:
        mock_litellm.acompletion = mock_acompletion

        agent = TsugiteAgent(
            model_string="openai:gpt-4o-mini",
            tools=[],
            instructions="",
            max_turns=5,
            ui_handler=mock_ui_handler,
        )

        await agent.run("Test task")

        # Verify STEP_START was triggered twice
        step_start_events = [e for e in mock_ui_handler.events if e["event"] == UIEvent.STEP_START]
        assert len(step_start_events) == 2
        assert step_start_events[0]["data"]["step"] == 1
        assert step_start_events[1]["data"]["step"] == 2


@pytest.mark.asyncio
async def test_ui_event_code_execution(mock_ui_handler, mock_litellm_response):
    """Test that CODE_EXECUTION event is triggered before code execution."""

    with patch("tsugite.core.agent.litellm") as mock_litellm:
        mock_litellm.acompletion = AsyncMock(
            return_value=mock_litellm_response(
                """Thought: Execute code.

```python
result = 5 + 3
final_answer(result)
```"""
            )
        )

        agent = TsugiteAgent(
            model_string="openai:gpt-4o-mini",
            tools=[],
            instructions="",
            max_turns=5,
            ui_handler=mock_ui_handler,
        )

        await agent.run("Test task")

        # Verify CODE_EXECUTION was triggered
        code_exec_events = [e for e in mock_ui_handler.events if e["event"] == UIEvent.CODE_EXECUTION]
        assert len(code_exec_events) == 1
        assert "result = 5 + 3" in code_exec_events[0]["data"]["code"]
        assert "final_answer(result)" in code_exec_events[0]["data"]["code"]


@pytest.mark.asyncio
async def test_ui_event_observation(mock_ui_handler, mock_litellm_response):
    """Test that OBSERVATION event is triggered after code execution."""

    with patch("tsugite.core.agent.litellm") as mock_litellm:
        mock_litellm.acompletion = AsyncMock(
            return_value=mock_litellm_response(
                """Thought: Print something.

```python
print("Hello, World!")
final_answer("done")
```"""
            )
        )

        agent = TsugiteAgent(
            model_string="openai:gpt-4o-mini",
            tools=[],
            instructions="",
            max_turns=5,
            ui_handler=mock_ui_handler,
        )

        await agent.run("Test task")

        # Verify OBSERVATION was triggered
        observation_events = [e for e in mock_ui_handler.events if e["event"] == UIEvent.OBSERVATION]
        assert len(observation_events) == 1
        assert "Hello, World!" in observation_events[0]["data"]["observation"]


@pytest.mark.asyncio
async def test_ui_event_final_answer(mock_ui_handler, mock_litellm_response):
    """Test that FINAL_ANSWER event is triggered when agent completes."""

    with patch("tsugite.core.agent.litellm") as mock_litellm:
        mock_litellm.acompletion = AsyncMock(
            return_value=mock_litellm_response(
                """Thought: Return the answer.

```python
final_answer("The answer is 42")
```"""
            )
        )

        agent = TsugiteAgent(
            model_string="openai:gpt-4o-mini",
            tools=[],
            instructions="",
            max_turns=5,
            ui_handler=mock_ui_handler,
        )

        await agent.run("Test task")

        # Verify FINAL_ANSWER was triggered
        final_answer_events = [e for e in mock_ui_handler.events if e["event"] == UIEvent.FINAL_ANSWER]
        assert len(final_answer_events) == 1
        assert final_answer_events[0]["data"]["answer"] == "The answer is 42"


@pytest.mark.asyncio
async def test_ui_event_error_on_execution_failure(mock_ui_handler, mock_litellm_response):
    """Test that ERROR event is triggered when code execution fails."""

    with patch("tsugite.core.agent.litellm") as mock_litellm:
        # First call: code with error
        # Second call: fix the error
        call_count = 0

        async def mock_acompletion(*args, **kwargs):
            nonlocal call_count
            call_count += 1

            if call_count == 1:
                return mock_litellm_response(
                    """Thought: Try to divide by zero.

```python
result = 1 / 0
```"""
                )
            else:
                return mock_litellm_response(
                    """Thought: Fix the error.

```python
final_answer(1)
```"""
                )

        mock_litellm.acompletion = mock_acompletion

        agent = TsugiteAgent(
            model_string="openai:gpt-4o-mini",
            tools=[],
            instructions="",
            max_turns=5,
            ui_handler=mock_ui_handler,
        )

        await agent.run("Test task")

        # Verify ERROR was triggered
        error_events = [e for e in mock_ui_handler.events if e["event"] == UIEvent.ERROR]
        assert len(error_events) == 1
        assert "ZeroDivisionError" in error_events[0]["data"]["error"]
        assert error_events[0]["data"]["error_type"] == "Execution Error"


@pytest.mark.asyncio
async def test_ui_event_error_on_max_turns(mock_ui_handler, mock_litellm_response):
    """Test that ERROR event is triggered when max_turns is reached."""

    with patch("tsugite.core.agent.litellm") as mock_litellm:
        # Always return code without final_answer
        mock_litellm.acompletion = AsyncMock(
            return_value=mock_litellm_response(
                """Thought: Still working...

```python
x = 1
print(x)
```"""
            )
        )

        agent = TsugiteAgent(
            model_string="openai:gpt-4o-mini",
            tools=[],
            instructions="",
            max_turns=2,
            ui_handler=mock_ui_handler,
        )

        with pytest.raises(RuntimeError):
            await agent.run("Test task")

        # Verify ERROR was triggered
        error_events = [e for e in mock_ui_handler.events if e["event"] == UIEvent.ERROR]
        assert len(error_events) == 1
        assert "max_turns" in error_events[0]["data"]["error"]
        assert error_events[0]["data"]["error_type"] == "RuntimeError"


@pytest.mark.asyncio
async def test_ui_event_reasoning_content(mock_ui_handler, mock_litellm_response):
    """Test that REASONING_CONTENT event is triggered for reasoning models."""

    with patch("tsugite.core.agent.litellm") as mock_litellm:
        mock_litellm.acompletion = AsyncMock(
            return_value=mock_litellm_response(
                """Thought: Solve this.

```python
final_answer(100)
```""",
                reasoning_content="Deep thinking process here...",
            )
        )

        agent = TsugiteAgent(
            model_string="anthropic:claude-3-7-sonnet",
            tools=[],
            instructions="",
            max_turns=5,
            ui_handler=mock_ui_handler,
        )

        await agent.run("Test task")

        # Verify REASONING_CONTENT was triggered
        reasoning_events = [e for e in mock_ui_handler.events if e["event"] == UIEvent.REASONING_CONTENT]
        assert len(reasoning_events) == 1
        assert reasoning_events[0]["data"]["content"] == "Deep thinking process here..."
        assert reasoning_events[0]["data"]["step"] == 1


@pytest.mark.asyncio
async def test_ui_event_reasoning_tokens(mock_ui_handler, mock_litellm_response):
    """Test that REASONING_TOKENS event is triggered for o1/o3 models."""

    with patch("tsugite.core.agent.litellm") as mock_litellm:
        response = mock_litellm_response(
            """Thought: Solve this.

```python
final_answer(100)
```"""
        )
        # Add reasoning token details (o1/o3 format)
        response.usage.completion_tokens_details = MagicMock()
        response.usage.completion_tokens_details.reasoning_tokens = 256

        mock_litellm.acompletion = AsyncMock(return_value=response)

        agent = TsugiteAgent(
            model_string="openai:o1",
            tools=[],
            instructions="",
            max_turns=5,
            ui_handler=mock_ui_handler,
        )

        await agent.run("Test task")

        # Verify REASONING_TOKENS was triggered
        reasoning_token_events = [e for e in mock_ui_handler.events if e["event"] == UIEvent.REASONING_TOKENS]
        assert len(reasoning_token_events) == 1
        assert reasoning_token_events[0]["data"]["tokens"] == 256
        assert reasoning_token_events[0]["data"]["step"] == 1


@pytest.mark.asyncio
async def test_agent_without_ui_handler(mock_litellm_response):
    """Test that agent works correctly without a UI handler."""

    with patch("tsugite.core.agent.litellm") as mock_litellm:
        mock_litellm.acompletion = AsyncMock(
            return_value=mock_litellm_response(
                """Thought: Calculate.

```python
final_answer(42)
```"""
            )
        )

        # Create agent without ui_handler
        agent = TsugiteAgent(
            model_string="openai:gpt-4o-mini",
            tools=[],
            instructions="",
            max_turns=5,
            ui_handler=None,  # No UI handler
        )

        # Should work fine without errors
        result = await agent.run("Test task")
        assert result == 42


@pytest.mark.asyncio
async def test_ui_event_order(mock_ui_handler, mock_litellm_response):
    """Test that UI events are triggered in the correct order."""

    with patch("tsugite.core.agent.litellm") as mock_litellm:
        mock_litellm.acompletion = AsyncMock(
            return_value=mock_litellm_response(
                """Thought: Calculate.

```python
print("Hello")
final_answer(42)
```"""
            )
        )

        agent = TsugiteAgent(
            model_string="openai:gpt-4o-mini",
            tools=[],
            instructions="",
            max_turns=5,
            ui_handler=mock_ui_handler,
        )

        await agent.run("Test task")

        # Extract event types in order
        event_types = [e["event"] for e in mock_ui_handler.events]

        # Verify order: TASK_START -> STEP_START -> LLM_MESSAGE -> CODE_EXECUTION -> OBSERVATION -> FINAL_ANSWER
        assert event_types[0] == UIEvent.TASK_START
        assert event_types[1] == UIEvent.STEP_START
        assert event_types[2] == UIEvent.LLM_MESSAGE  # LLM reasoning is now shown before code execution
        assert event_types[3] == UIEvent.CODE_EXECUTION
        assert event_types[4] == UIEvent.OBSERVATION
        assert event_types[5] == UIEvent.FINAL_ANSWER


@pytest.mark.asyncio
async def test_ui_event_error_on_no_code_generation(mock_ui_handler, mock_litellm_response):
    """Test that ERROR event is triggered when LLM doesn't generate code."""

    with patch("tsugite.core.agent.litellm") as mock_litellm:
        # First call: LLM only provides thought, no code
        # Second call: LLM provides proper format
        call_count = 0

        async def mock_acompletion(*args, **kwargs):
            nonlocal call_count
            call_count += 1

            if call_count == 1:
                return mock_litellm_response(
                    """Thought: I'm thinking about this problem and how to solve it, but I'm not providing any code."""
                )
            else:
                return mock_litellm_response(
                    """Thought: Now I'll provide code.

```python
final_answer(42)
```"""
                )

        mock_litellm.acompletion = mock_acompletion

        agent = TsugiteAgent(
            model_string="openai:gpt-4o-mini",
            tools=[],
            instructions="",
            max_turns=5,
            ui_handler=mock_ui_handler,
        )

        await agent.run("Test task")

        # Verify ERROR was triggered for missing code
        error_events = [e for e in mock_ui_handler.events if e["event"] == UIEvent.ERROR]
        assert len(error_events) == 1
        assert "LLM did not generate code" in error_events[0]["data"]["error"]
        assert "Expected format" in error_events[0]["data"]["error"]
        assert error_events[0]["data"]["error_type"] == "Format Error"
