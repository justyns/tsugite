"""Tests for TsugiteAgent."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tsugite.core.agent import AgentResult, TsugiteAgent
from tsugite.core.executor import LocalExecutor
from tsugite.core.tools import create_tool_from_function


@pytest.fixture
def mock_litellm_response():
    """Create a mock LiteLLM response."""

    def _create_response(content: str, reasoning_content: str = None):
        """Factory to create mock responses with different content."""
        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message = MagicMock(spec=[])  # Start with empty spec
        response.choices[0].message.content = content

        # Add reasoning content if provided
        if reasoning_content:
            # Add reasoning_content attribute
            response.choices[0].message.reasoning_content = reasoning_content

        # Add token usage
        response.usage = MagicMock()
        response.usage.total_tokens = 100

        return response

    return _create_response


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
        max_steps=10,
    )

    assert agent.model_string == "openai:gpt-4o-mini"
    assert len(agent.tools) == 1
    assert agent.instructions == "You are a helpful assistant"
    assert agent.max_steps == 10
    assert isinstance(agent.executor, LocalExecutor)
    assert agent.tool_map["add"] == tool


@pytest.mark.asyncio
async def test_agent_simple_calculation(mock_litellm_response):
    """Test agent can do basic calculation and return final_answer."""

    # Mock litellm.acompletion to return a response with final_answer
    with patch("tsugite.core.agent.litellm") as mock_litellm:
        mock_litellm.acompletion = AsyncMock(
            return_value=mock_litellm_response(
                """Thought: I'll calculate 5 + 3 using Python.

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
            max_steps=5,
        )

        result = await agent.run("What is 5 + 3?")

        # Should return the final answer (8)
        assert result == 8

        # Verify litellm was called
        mock_litellm.acompletion.assert_called_once()


@pytest.mark.asyncio
async def test_agent_multi_step_reasoning(mock_litellm_response):
    """Test agent can do multi-step reasoning."""

    call_count = 0

    async def mock_acompletion(*args, **kwargs):
        nonlocal call_count
        call_count += 1

        if call_count == 1:
            # First step: Set a variable
            return mock_litellm_response(
                """Thought: I'll start by setting x to 5.

```python
x = 5
print(f"x = {x}")
```"""
            )
        elif call_count == 2:
            # Second step: Use the variable and call final_answer
            return mock_litellm_response(
                """Thought: Now I'll multiply x by 2 and return the result.

```python
result = x * 2
final_answer(result)
```"""
            )

    with patch("tsugite.core.agent.litellm") as mock_litellm:
        mock_litellm.acompletion = mock_acompletion

        agent = TsugiteAgent(
            model_string="openai:gpt-4o-mini",
            tools=[],
            instructions="",
            max_steps=5,
        )

        result = await agent.run("Calculate 5 * 2")

        # Should return 10
        assert result == 10

        # Should have taken 2 steps
        assert len(agent.memory.steps) == 2

        # Check first step
        assert "x = 5" in agent.memory.steps[0].code
        assert "x = 5" in agent.memory.steps[0].output

        # Check second step
        assert "result = x * 2" in agent.memory.steps[1].code


@pytest.mark.asyncio
async def test_agent_with_tools(mock_litellm_response):
    """Test agent can use tools."""

    def multiply(a: int, b: int) -> int:
        """Multiply two numbers"""
        return a * b

    tool = create_tool_from_function(multiply)

    with patch("tsugite.core.agent.litellm") as mock_litellm:
        mock_litellm.acompletion = AsyncMock(
            return_value=mock_litellm_response(
                """Thought: I'll use the multiply tool.

```python
result = multiply(5, 3)
final_answer(result)
```"""
            )
        )

        agent = TsugiteAgent(
            model_string="openai:gpt-4o-mini",
            tools=[tool],
            instructions="",
            max_steps=5,
        )

        # Inject the tool into the executor namespace so it can be called
        agent.executor.namespace["multiply"] = tool.function

        result = await agent.run("What is 5 * 3?")

        # Should return 15
        assert result == 15


@pytest.mark.asyncio
async def test_agent_max_steps_reached(mock_litellm_response):
    """Test agent raises error when max_steps is reached."""

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
            max_steps=3,
        )

        # Should raise RuntimeError
        with pytest.raises(RuntimeError) as exc_info:
            await agent.run("Some task")

        assert "max_steps" in str(exc_info.value)
        assert "3" in str(exc_info.value)


@pytest.mark.asyncio
async def test_agent_return_full_result(mock_litellm_response):
    """Test agent can return full result with metadata."""

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
            max_steps=5,
        )

        result = await agent.run("What is the answer?", return_full_result=True)

        # Should return AgentResult
        assert isinstance(result, AgentResult)
        assert result.output == 42
        assert result.token_usage == 100
        assert result.steps is not None


@pytest.mark.asyncio
async def test_agent_reasoning_model_support(mock_litellm_response):
    """Test agent supports reasoning models with reasoning_effort."""

    with patch("tsugite.core.agent.litellm") as mock_litellm:
        mock_litellm.acompletion = AsyncMock(
            return_value=mock_litellm_response(
                """Thought: Using reasoning to solve this.

```python
final_answer(100)
```""",
                reasoning_content="[Hidden reasoning process...]",
            )
        )

        agent = TsugiteAgent(
            model_string="openai:o1",
            tools=[],
            instructions="",
            max_steps=5,
            model_kwargs={"reasoning_effort": "high"},
        )

        await agent.run("Solve this problem")

        # Should capture reasoning content
        assert len(agent.memory.reasoning_history) == 1
        assert "[Hidden reasoning process...]" in agent.memory.reasoning_history[0]

        # Should have called litellm with reasoning_effort
        call_kwargs = mock_litellm.acompletion.call_args[1]
        assert "reasoning_effort" in call_kwargs
        assert call_kwargs["reasoning_effort"] == "high"


@pytest.mark.asyncio
async def test_agent_model_kwargs_passed_to_litellm(mock_litellm_response):
    """Test that model_kwargs are passed to litellm.acompletion."""

    with patch("tsugite.core.agent.litellm") as mock_litellm:
        mock_litellm.acompletion = AsyncMock(
            return_value=mock_litellm_response(
                """Thought: Done.

```python
final_answer("test")
```"""
            )
        )

        agent = TsugiteAgent(
            model_string="openai:gpt-4o-mini",
            tools=[],
            instructions="",
            max_steps=5,
            model_kwargs={
                "temperature": 0.7,
                "max_tokens": 1000,
                "response_format": {"type": "json_object"},
            },
        )

        await agent.run("Task")

        # Verify kwargs were passed
        call_kwargs = mock_litellm.acompletion.call_args[1]
        assert call_kwargs["temperature"] == 0.7
        assert call_kwargs["max_tokens"] == 1000
        assert call_kwargs["response_format"] == {"type": "json_object"}


@pytest.mark.asyncio
async def test_agent_error_handling(mock_litellm_response):
    """Test agent handles code execution errors."""

    call_count = 0

    async def mock_acompletion(*args, **kwargs):
        nonlocal call_count
        call_count += 1

        if call_count == 1:
            # First step: Code with error
            return mock_litellm_response(
                """Thought: Try to divide by zero.

```python
result = 1 / 0
print(result)
```"""
            )
        elif call_count == 2:
            # Second step: Fix the error
            return mock_litellm_response(
                """Thought: That failed. Let me try a different approach.

```python
result = 1 / 1
final_answer(result)
```"""
            )

    with patch("tsugite.core.agent.litellm") as mock_litellm:
        mock_litellm.acompletion = mock_acompletion

        agent = TsugiteAgent(
            model_string="openai:gpt-4o-mini",
            tools=[],
            instructions="",
            max_steps=5,
        )

        result = await agent.run("Calculate something")

        # Should have recovered from error
        assert result == 1.0

        # Check that first step has error
        assert agent.memory.steps[0].error is not None
        assert "ZeroDivisionError" in agent.memory.steps[0].error

        # Check that second step succeeded
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
        max_steps=5,
    )

    prompt = agent._build_system_prompt()

    # Check that prompt includes tool definition
    assert "search" in prompt
    assert "Search for information" in prompt

    # Check that prompt includes instructions
    assert "You are an expert researcher." in prompt

    # Check that prompt has basic structure
    assert "Thought" in prompt
    assert "Code" in prompt
    assert "Observation" in prompt
    assert "final_answer" in prompt


@pytest.mark.asyncio
async def test_agent_parse_response():
    """Test parsing LLM responses."""
    agent = TsugiteAgent(
        model_string="openai:gpt-4o-mini",
        tools=[],
        instructions="",
        max_steps=5,
    )

    # Mock response object
    response = MagicMock()
    response.choices = [MagicMock()]

    # Test parsing thought and code
    response.choices[0].message.content = """Thought: I need to calculate this.

```python
x = 5 + 3
print(x)
```"""

    thought, code, final_answer = agent._parse_response(response)

    assert "calculate" in thought.lower()
    assert "x = 5 + 3" in code
    assert "print(x)" in code
    assert final_answer is None

    # Test parsing with final_answer
    response.choices[0].message.content = """Thought: Return the result.

```python
final_answer(42)
```"""

    thought, code, final_answer = agent._parse_response(response)

    assert "result" in thought.lower()
    assert "final_answer(42)" in code
    assert final_answer is None  # Will be extracted after execution


@pytest.mark.asyncio
async def test_agent_litellm_params():
    """Test that LiteLLM params are pre-computed correctly."""
    agent = TsugiteAgent(
        model_string="openai:gpt-4o-mini",
        tools=[],
        instructions="",
        max_steps=5,
    )

    # Should have pre-computed litellm_params with model key
    assert "model" in agent.litellm_params
    assert agent.litellm_params["model"] == "openai/gpt-4o-mini"

    # Test with reasoning model - should filter unsupported params
    agent_o1 = TsugiteAgent(
        model_string="openai:o1",
        tools=[],
        instructions="",
        max_steps=5,
        model_kwargs={"temperature": 0.7, "reasoning_effort": "high"},
    )

    # Temperature should be filtered out for o1
    assert "temperature" not in agent_o1.litellm_params
    # reasoning_effort should be kept
    assert agent_o1.litellm_params.get("reasoning_effort") == "high"


@pytest.mark.asyncio
async def test_agent_build_messages():
    """Test building message history from memory."""
    agent = TsugiteAgent(
        model_string="openai:gpt-4o-mini",
        tools=[],
        instructions="",
        max_steps=5,
    )

    # Set task
    agent.memory.add_task("Calculate 5 + 3")

    # Add a step
    agent.memory.add_step(
        thought="I'll use Python to calculate",
        code="result = 5 + 3\nprint(result)",
        output="8",
        error=None,
    )

    messages = agent._build_messages()

    # Should have system, user (task), assistant (thought+code), user (observation)
    assert len(messages) == 4

    # Check system message
    assert messages[0]["role"] == "system"
    assert "Thought" in messages[0]["content"]

    # Check task
    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == "Calculate 5 + 3"

    # Check assistant response
    assert messages[2]["role"] == "assistant"
    assert "Thought: I'll use Python to calculate" in messages[2]["content"]
    assert "```python" in messages[2]["content"]

    # Check observation
    assert messages[3]["role"] == "user"
    assert "Observation: 8" in messages[3]["content"]


@pytest.mark.asyncio
async def test_agent_extract_reasoning_content():
    """Test extracting reasoning content from responses."""
    agent = TsugiteAgent(
        model_string="openai:o1",
        tools=[],
        instructions="",
        max_steps=5,
    )

    # Mock response with reasoning_content
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.reasoning_content = "Deep thinking process..."

    reasoning = agent._extract_reasoning_content(response)
    assert reasoning == "Deep thinking process..."

    # Mock response without reasoning_content
    response2 = MagicMock()
    response2.choices = [MagicMock()]
    response2.choices[0].message = MagicMock(spec=["content"])  # No reasoning_content

    reasoning2 = agent._extract_reasoning_content(response2)
    assert reasoning2 is None


@pytest.mark.asyncio
async def test_agent_custom_executor():
    """Test agent can use custom executor."""

    # Create mock executor
    mock_executor = AsyncMock()
    mock_executor.execute = AsyncMock(return_value=MagicMock(output="custom output", error=None, final_answer=42))

    with patch("tsugite.core.agent.litellm") as mock_litellm:
        mock_litellm.acompletion = AsyncMock(
            return_value=MagicMock(
                choices=[
                    MagicMock(
                        message=MagicMock(
                            content="""Thought: Test.

```python
final_answer(42)
```"""
                        )
                    )
                ],
                usage=MagicMock(total_tokens=50),
            )
        )

        agent = TsugiteAgent(
            model_string="openai:gpt-4o-mini",
            tools=[],
            instructions="",
            max_steps=5,
            executor=mock_executor,
        )

        result = await agent.run("Test task")

        # Should use custom executor
        mock_executor.execute.assert_called_once()
        assert result == 42


def test_tool_execution_no_task_warnings():
    """Test that calling async tools from sync context doesn't produce Task warnings.

    This tests the fix for the issue where tool execution would create asyncio
    Tasks that were never awaited, producing warning messages like:
    - '💡 <Task pending name='Task-7' coro=<Tool.execute()>>'
    - 'Task exception was never retrieved'
    """
    import asyncio
    import sys
    from io import StringIO

    from tsugite.core.tools import Tool

    # Create async tool (like real tools)
    async def async_search(query: str) -> str:
        """Search for information"""
        await asyncio.sleep(0.001)  # Simulate async work
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

    # Create agent and inject tool
    agent = TsugiteAgent(
        model_string="openai:gpt-4o-mini",
        tools=[tool],
        instructions="",
        max_steps=5,
    )

    # Capture stderr to check for Task warnings
    old_stderr = sys.stderr
    stderr_capture = StringIO()
    sys.stderr = stderr_capture

    try:
        # Call the tool from sync context (simulating what exec() does)
        # This should NOT produce Task warnings
        result = agent.executor.namespace["async_search"](query="test")

        # Verify result is correct
        assert result == "Results for: test"

    finally:
        sys.stderr = old_stderr

    # Check that no Task warnings were produced
    stderr_output = stderr_capture.getvalue()

    # Filter out unrelated warnings (like litellm cleanup)
    filtered_stderr = "\n".join(
        line for line in stderr_output.split("\n") if "async_search" in line or "Task pending" in line or "never retrieved" in line
    )

    assert "Task pending" not in filtered_stderr, f"Unexpected Task pending warning:\n{stderr_output}"
    assert "Task exception was never retrieved" not in filtered_stderr, f"Unexpected Task exception warning:\n{stderr_output}"


def test_tool_exception_propagation_from_async():
    """Test that exceptions from async tools are properly propagated.

    This ensures that when an async tool raises an exception, it's properly
    raised in the sync context rather than being lost or creating warnings.
    """
    import asyncio
    import sys
    from io import StringIO

    from tsugite.core.tools import Tool

    # Create async tool that raises exception
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

    # Create agent and inject tool
    agent = TsugiteAgent(
        model_string="openai:gpt-4o-mini",
        tools=[tool],
        instructions="",
        max_steps=5,
    )

    # Capture stderr
    old_stderr = sys.stderr
    stderr_capture = StringIO()
    sys.stderr = stderr_capture

    try:
        # Call the tool and expect exception
        with pytest.raises(RuntimeError) as exc_info:
            agent.executor.namespace["failing_tool"](value="test")

        # Verify exception message
        assert "Tool failed with value: test" in str(exc_info.value)

    finally:
        sys.stderr = old_stderr

    # Check that no "exception was never retrieved" warnings appeared
    stderr_output = stderr_capture.getvalue()
    filtered_stderr = "\n".join(
        line for line in stderr_output.split("\n") if "failing_tool" in line or "never retrieved" in line
    )

    assert "exception was never retrieved" not in filtered_stderr.lower(), f"Exception handling broken:\n{stderr_output}"
