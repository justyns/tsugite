"""Tests for TsugiteAgent."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tsugite.core.agent import AgentResult, TsugiteAgent
from tsugite.core.executor import LocalExecutor
from tsugite.core.tools import create_tool_from_function
from tsugite.providers.base import CompletionResponse, Usage


def _mock_response(content: str, reasoning_content: str = None) -> CompletionResponse:
    return CompletionResponse(
        content=content,
        reasoning_content=reasoning_content,
        usage=Usage(total_tokens=100),
        cost=0.001,
    )


def _patch_provider(agent, side_effect=None, return_value=None):
    """Patch an agent's provider.acompletion with an AsyncMock."""
    mock = AsyncMock(side_effect=side_effect, return_value=return_value)
    agent._provider = MagicMock()
    agent._provider.acompletion = mock
    return mock


@pytest.mark.asyncio
async def test_agent_creation():
    """Test creating a TsugiteAgent."""

    def add(a: int, b: int) -> int:
        """Add two numbers"""
        return a + b

    tool = create_tool_from_function(add)

    agent = TsugiteAgent(
        model_string="openai:gpt-4o-mini",
        tools=[tool],
        instructions="You are a helpful assistant",
        max_turns=10,
    )

    assert agent.model_string == "openai:gpt-4o-mini"
    assert len(agent.tools) == 1
    assert agent.instructions == "You are a helpful assistant"
    assert agent.max_turns == 10
    assert isinstance(agent.executor, LocalExecutor)
    assert agent.tool_map["add"] == tool


@pytest.mark.asyncio
async def test_agent_simple_calculation():
    """Test agent can do basic calculation and return final_answer."""

    agent = TsugiteAgent(
        model_string="openai:gpt-4o-mini",
        tools=[],
        instructions="",
        max_turns=5,
    )

    mock = _patch_provider(agent, return_value=_mock_response("""Thought: I'll calculate 5 + 3 using Python.

```python
result = 5 + 3
final_answer(result)
```"""))

    result = await agent.run("What is 5 + 3?")
    assert result == 8
    mock.assert_called_once()


@pytest.mark.asyncio
async def test_agent_multi_step_reasoning():
    """Test agent can do multi-step reasoning."""

    agent = TsugiteAgent(
        model_string="openai:gpt-4o-mini",
        tools=[],
        instructions="",
        max_turns=5,
    )

    call_count = 0

    async def mock_acompletion(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return _mock_response("""Thought: I'll start by setting x to 5.

```python
x = 5
print(f"x = {x}")
```""")
        else:
            return _mock_response("""Thought: Now I'll multiply x by 2 and return the result.

```python
result = x * 2
final_answer(result)
```""")

    _patch_provider(agent, side_effect=mock_acompletion)
    result = await agent.run("Calculate 5 * 2")

    assert result == 10
    assert len(agent.memory.steps) == 2
    assert "x = 5" in agent.memory.steps[0].code
    assert "x = 5" in agent.memory.steps[0].output
    assert "result = x * 2" in agent.memory.steps[1].code


@pytest.mark.asyncio
async def test_agent_with_tools():
    """Test agent can use tools."""

    def multiply(a: int, b: int) -> int:
        """Multiply two numbers"""
        return a * b

    tool = create_tool_from_function(multiply)

    agent = TsugiteAgent(
        model_string="openai:gpt-4o-mini",
        tools=[tool],
        instructions="",
        max_turns=5,
    )

    _patch_provider(agent, return_value=_mock_response("""Thought: I'll use the multiply tool.

```python
result = multiply(5, 3)
final_answer(result)
```"""))

    agent.executor.namespace["multiply"] = tool.function
    result = await agent.run("What is 5 * 3?")
    assert result == 15


@pytest.mark.asyncio
async def test_agent_max_turns_reached():
    """Test agent raises error when max_turns is reached."""

    agent = TsugiteAgent(
        model_string="openai:gpt-4o-mini",
        tools=[],
        instructions="",
        max_turns=3,
    )

    _patch_provider(agent, return_value=_mock_response("""Thought: Still working...

```python
x = 1
print(x)
```"""))

    with pytest.raises(RuntimeError) as exc_info:
        await agent.run("Some task")

    assert "max_turns" in str(exc_info.value)
    assert "3" in str(exc_info.value)


@pytest.mark.asyncio
async def test_agent_return_full_result():
    """Test agent can return full result with metadata."""

    agent = TsugiteAgent(
        model_string="openai:gpt-4o-mini",
        tools=[],
        instructions="",
        max_turns=5,
    )

    _patch_provider(agent, return_value=_mock_response("""Thought: Calculate the answer.

```python
final_answer(42)
```"""))

    result = await agent.run("What is the answer?", return_full_result=True)

    assert isinstance(result, AgentResult)
    assert result.output == 42
    assert result.token_usage == 100
    assert result.steps is not None


@pytest.mark.asyncio
async def test_agent_reasoning_model_support():
    """Test agent supports reasoning models with reasoning_effort."""

    agent = TsugiteAgent(
        model_string="openai:o1",
        tools=[],
        instructions="",
        max_turns=5,
        model_kwargs={"reasoning_effort": "high"},
    )

    _patch_provider(agent, return_value=_mock_response(
        """Thought: Using reasoning to solve this.

```python
final_answer(100)
```""",
        reasoning_content="[Hidden reasoning process...]",
    ))

    await agent.run("Solve this problem")

    # Should capture reasoning content
    assert len(agent.memory.reasoning_history) == 1
    assert "[Hidden reasoning process...]" in agent.memory.reasoning_history[0]


@pytest.mark.asyncio
async def test_agent_model_kwargs_passed_to_provider():
    """Test that model_kwargs are passed to provider.acompletion."""

    agent = TsugiteAgent(
        model_string="openai:gpt-4o-mini",
        tools=[],
        instructions="",
        max_turns=5,
        model_kwargs={
            "temperature": 0.7,
            "max_tokens": 1000,
            "response_format": {"type": "json_object"},
        },
    )

    mock = _patch_provider(agent, return_value=_mock_response("""Thought: Done.

```python
final_answer("test")
```"""))

    await agent.run("Task")

    # Verify kwargs were passed
    call_kwargs = mock.call_args[1]
    assert call_kwargs["temperature"] == 0.7
    assert call_kwargs["max_tokens"] == 1000
    assert call_kwargs["response_format"] == {"type": "json_object"}


@pytest.mark.asyncio
async def test_agent_error_handling():
    """Test agent handles code execution errors."""

    agent = TsugiteAgent(
        model_string="openai:gpt-4o-mini",
        tools=[],
        instructions="",
        max_turns=5,
    )

    call_count = 0

    async def mock_acompletion(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return _mock_response("""Thought: Try to divide by zero.

```python
result = 1 / 0
print(result)
```""")
        else:
            return _mock_response("""Thought: That failed. Let me try a different approach.

```python
result = 1 / 1
final_answer(result)
```""")

    _patch_provider(agent, side_effect=mock_acompletion)
    result = await agent.run("Calculate something")

    assert result == 1.0
    assert agent.memory.steps[0].error is not None
    assert "ZeroDivisionError" in agent.memory.steps[0].error
    assert agent.memory.steps[1].error is None


@pytest.mark.asyncio
async def test_agent_build_system_prompt():
    """Test system prompt includes tools and instructions."""

    def search(query: str) -> str:
        """Search for information"""
        return "results"

    tool = create_tool_from_function(search)

    agent = TsugiteAgent(
        model_string="openai:gpt-4o-mini",
        tools=[tool],
        instructions="You are an expert researcher.",
        max_turns=5,
    )

    prompt = agent._build_system_prompt()

    assert "search" in prompt
    assert "Search for information" in prompt
    assert "You are an expert researcher." in prompt
    assert "tsugite_execution_result" in prompt
    assert "final_answer" in prompt
    assert "```python" in prompt


@pytest.mark.asyncio
async def test_agent_parse_response():
    """Test parsing LLM responses."""
    agent = TsugiteAgent(
        model_string="openai:gpt-4o-mini",
        tools=[],
        instructions="",
        max_turns=5,
    )

    parsed = agent._parse_response_from_text("""Thought: I need to calculate this.

```python
x = 5 + 3
print(x)
```""")

    assert "calculate" in parsed.thought.lower()
    assert "x = 5 + 3" in parsed.code
    assert "print(x)" in parsed.code

    parsed = agent._parse_response_from_text("""Thought: Return the result.

```python
final_answer(42)
```""")

    assert "result" in parsed.thought.lower()
    assert "final_answer(42)" in parsed.code


@pytest.mark.asyncio
async def test_agent_model_kwargs():
    """Test that model_kwargs are correctly filtered for reasoning models."""
    agent = TsugiteAgent(
        model_string="openai:gpt-4o-mini",
        tools=[],
        instructions="",
        max_turns=5,
    )

    assert agent._model_id == "gpt-4o-mini"

    # Test with reasoning model - should filter unsupported params
    agent_o1 = TsugiteAgent(
        model_string="openai:o1",
        tools=[],
        instructions="",
        max_turns=5,
        model_kwargs={"temperature": 0.7, "reasoning_effort": "high"},
    )

    # Temperature should be filtered out for o1
    assert "temperature" not in agent_o1._model_kwargs
    # reasoning_effort should be kept
    assert agent_o1._model_kwargs.get("reasoning_effort") == "high"


@pytest.mark.asyncio
async def test_agent_build_messages():
    """Test building message history from memory."""
    agent = TsugiteAgent(
        model_string="openai:gpt-4o-mini",
        tools=[],
        instructions="",
        max_turns=5,
    )

    agent.memory.add_task("Calculate 5 + 3")

    agent.memory.add_step(
        thought="I'll use Python to calculate",
        code="result = 5 + 3\nprint(result)",
        output="8",
        error=None,
        xml_observation='<tsugite_execution_result status="success">\n<output>8</output>\n</tsugite_execution_result>',
    )

    messages = agent._build_messages()

    assert len(messages) == 4
    assert messages[0]["role"] == "system"
    assert "tsugite_execution_result" in messages[0]["content"]
    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == "Calculate 5 + 3"
    assert messages[2]["role"] == "assistant"
    assert "```python" in messages[2]["content"]
    assert "result = 5 + 3" in messages[2]["content"]
    assert messages[3]["role"] == "user"
    assert "<tsugite_execution_result" in messages[3]["content"]
    assert "<output>8</output>" in messages[3]["content"]


@pytest.mark.asyncio
async def test_agent_build_messages_no_code_includes_thought():
    """Test that steps with no code show the LLM's actual thought in history."""
    agent = TsugiteAgent(
        model_string="openai:gpt-4o-mini",
        tools=[],
        instructions="",
        max_turns=5,
    )

    agent.memory.add_task("Hello")

    agent.memory.add_step(
        thought="I want to greet the user but forgot to use a code block.",
        code="",
        output="",
        error=None,
        xml_observation='<tsugite_execution_result status="error">\n<error>Format Error</error>\n</tsugite_execution_result>',
    )

    messages = agent._build_messages()

    assistant_msg = messages[2]
    assert assistant_msg["role"] == "assistant"
    assert "```python" not in assistant_msg["content"]
    assert "I want to greet the user" in assistant_msg["content"]


@pytest.mark.asyncio
async def test_agent_custom_executor():
    """Test agent can use custom executor."""

    mock_executor = MagicMock()
    mock_executor.execute = AsyncMock(return_value=MagicMock(output="custom output", error=None, final_answer=42))
    mock_executor.namespace = {}

    agent = TsugiteAgent(
        model_string="openai:gpt-4o-mini",
        tools=[],
        instructions="",
        max_turns=5,
        executor=mock_executor,
    )

    _patch_provider(agent, return_value=_mock_response("""Thought: Test.

```python
final_answer(42)
```"""))

    result = await agent.run("Test task")

    mock_executor.execute.assert_called_once()
    assert result == 42


def test_build_budget_tag():
    """Test budget tag includes turn and token info."""
    agent = TsugiteAgent(
        model_string="openai:gpt-4o-mini",
        tools=[],
        instructions="",
        max_turns=10,
    )

    tag = agent._build_budget_tag(0)
    assert '<tsugite_budget turn="1" max_turns="10" />' in tag
    assert "tokens_used" not in tag
    assert "warning" not in tag

    agent.total_tokens = 5000
    tag = agent._build_budget_tag(4)
    assert 'turn="5"' in tag
    assert 'max_turns="10"' in tag
    assert 'tokens_used="5000"' in tag
    assert "warning" not in tag

    tag = agent._build_budget_tag(8)
    assert 'turn="9"' in tag
    assert 'warning="approaching turn limit, wrap up soon"' in tag

    tag = agent._build_budget_tag(9)
    assert 'turn="10"' in tag
    assert "warning=" in tag


@pytest.mark.asyncio
async def test_budget_tag_in_observations():
    """Test that observations include the budget tag with accumulated tokens."""

    agent = TsugiteAgent(
        model_string="openai:gpt-4o-mini",
        tools=[],
        instructions="",
        max_turns=5,
    )

    call_count = 0

    async def mock_acompletion(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return _mock_response("""Thought: Step 1.

```python
x = 42
print(x)
```""")
        else:
            return _mock_response("""Thought: Done.

```python
final_answer(x)
```""")

    _patch_provider(agent, side_effect=mock_acompletion)
    result = await agent.run("Test task")
    assert result == 42

    step1_output = agent.memory.steps[0].output
    assert "<tsugite_budget" in step1_output
    assert 'turn="1"' in step1_output
    assert 'max_turns="5"' in step1_output
    assert 'tokens_used="100"' in step1_output

    step1_xml = agent.memory.steps[0].xml_observation
    assert "<tsugite_budget" in step1_xml
    assert 'turn="1"' in step1_xml

    assert agent.total_tokens == 200  # 100 per call * 2 calls


def test_tool_execution_no_task_warnings():
    """Test that calling async tools from sync context doesn't produce Task warnings."""
    import asyncio
    import sys
    from io import StringIO

    from tsugite.core.tools import Tool

    async def async_search(query: str) -> str:
        """Search for information"""
        await asyncio.sleep(0.001)
        return f"Results for: {query}"

    tool = Tool(
        name="async_search",
        description="Search for information",
        parameters={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
        function=async_search,
    )

    agent = TsugiteAgent(
        model_string="openai:gpt-4o-mini",
        tools=[tool],
        instructions="",
        max_turns=5,
    )

    old_stderr = sys.stderr
    stderr_capture = StringIO()
    sys.stderr = stderr_capture

    try:
        result = agent.executor.namespace["async_search"](query="test")
        assert result == "Results for: test"
    finally:
        sys.stderr = old_stderr

    stderr_output = stderr_capture.getvalue()
    filtered_stderr = "\n".join(
        line
        for line in stderr_output.split("\n")
        if "async_search" in line or "Task pending" in line or "never retrieved" in line
    )

    assert "Task pending" not in filtered_stderr, f"Unexpected Task pending warning:\n{stderr_output}"
    assert "Task exception was never retrieved" not in filtered_stderr, (
        f"Unexpected Task exception warning:\n{stderr_output}"
    )


def test_tool_exception_propagation_from_async():
    """Test that exceptions from async tools are properly propagated."""
    import asyncio
    import sys
    from io import StringIO

    from tsugite.core.tools import Tool

    async def failing_tool(value: str) -> str:
        """Tool that fails on purpose"""
        await asyncio.sleep(0.001)
        raise RuntimeError(f"Tool failed with value: {value}")

    tool = Tool(
        name="failing_tool",
        description="Tool that fails",
        parameters={
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
        },
        function=failing_tool,
    )

    agent = TsugiteAgent(
        model_string="openai:gpt-4o-mini",
        tools=[tool],
        instructions="",
        max_turns=5,
    )

    old_stderr = sys.stderr
    stderr_capture = StringIO()
    sys.stderr = stderr_capture

    try:
        with pytest.raises(RuntimeError) as exc_info:
            agent.executor.namespace["failing_tool"](value="test")
        assert "Tool failed with value: test" in str(exc_info.value)
    finally:
        sys.stderr = old_stderr

    stderr_output = stderr_capture.getvalue()
    filtered_stderr = "\n".join(
        line for line in stderr_output.split("\n") if "failing_tool" in line or "never retrieved" in line
    )

    assert "exception was never retrieved" not in filtered_stderr.lower(), (
        f"Exception handling broken:\n{stderr_output}"
    )
