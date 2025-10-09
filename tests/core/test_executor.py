"""Tests for code execution backends."""

import pytest

from tsugite.core.executor import ExecutionResult, LocalExecutor


@pytest.mark.asyncio
async def test_execution_result_creation():
    """Test creating an ExecutionResult."""
    result = ExecutionResult(
        output="Hello, world!",
        error=None,
        stdout="Hello, world!",
        stderr="",
    )

    assert result.output == "Hello, world!"
    assert result.error is None
    assert result.stdout == "Hello, world!"
    assert result.stderr == ""
    assert result.final_answer is None


@pytest.mark.asyncio
async def test_execution_result_with_error():
    """Test ExecutionResult with error."""
    result = ExecutionResult(
        output="",
        error="NameError: name 'x' is not defined",
        stdout="",
        stderr="NameError: name 'x' is not defined",
    )

    assert result.error == "NameError: name 'x' is not defined"


@pytest.mark.asyncio
async def test_local_executor_basic():
    """Test basic code execution."""
    executor = LocalExecutor()

    result = await executor.execute("print('Hello from executor')")

    assert result.error is None
    assert "Hello from executor" in result.output


@pytest.mark.asyncio
async def test_local_executor_state_persistence():
    """Test that variables persist between executions."""
    executor = LocalExecutor()

    # First execution - set variable
    result1 = await executor.execute("x = 42")
    assert result1.error is None

    # Second execution - use the variable
    result2 = await executor.execute("print(x)")
    assert result2.error is None
    assert "42" in result2.output

    # Third execution - modify the variable
    result3 = await executor.execute("x = x * 2; print(x)")
    assert result3.error is None
    assert "84" in result3.output


@pytest.mark.asyncio
async def test_local_executor_error_handling():
    """Test error handling in executor."""
    executor = LocalExecutor()

    # Execute code that will fail
    result = await executor.execute("1 / 0")

    assert result.error is not None
    assert "ZeroDivisionError" in result.error


@pytest.mark.asyncio
async def test_local_executor_final_answer():
    """Test final_answer() function capture."""
    executor = LocalExecutor()

    # Execute code that calls final_answer()
    result = await executor.execute("final_answer('The answer is 42')")

    assert result.error is None
    assert result.final_answer == "The answer is 42"


@pytest.mark.asyncio
async def test_local_executor_final_answer_with_computation():
    """Test final_answer() with actual computation."""
    executor = LocalExecutor()

    # Execute multi-line code with final_answer
    code = """
result = 5 + 3
final_answer(result)
"""
    result = await executor.execute(code)

    assert result.error is None
    assert result.final_answer == 8


@pytest.mark.asyncio
async def test_local_executor_send_variables():
    """Test injecting variables into namespace."""
    executor = LocalExecutor()

    # Inject variables
    await executor.send_variables({"user_data": {"name": "Alice", "age": 30}, "count": 5})

    # Use injected variables
    result = await executor.execute('print(f\'{user_data["name"]} is {user_data["age"]} years old\')')

    assert result.error is None
    assert "Alice is 30 years old" in result.output


@pytest.mark.asyncio
async def test_local_executor_send_variables_persist():
    """Test that injected variables persist."""
    executor = LocalExecutor()

    # Inject variable
    await executor.send_variables({"base_value": 100})

    # First execution using injected variable
    result1 = await executor.execute("result = base_value + 50")
    assert result1.error is None

    # Second execution using both injected and computed variables
    result2 = await executor.execute("print(result)")
    assert result2.error is None
    assert "150" in result2.output


@pytest.mark.asyncio
async def test_local_executor_multiple_executions():
    """Test multiple sequential executions."""
    executor = LocalExecutor()

    # Execute multiple code blocks
    await executor.execute("a = 1")
    await executor.execute("b = 2")
    await executor.execute("c = a + b")
    result = await executor.execute("print(c)")

    assert result.error is None
    assert "3" in result.output


@pytest.mark.asyncio
async def test_local_executor_import_json():
    """Test that json module is available (needed for final_answer serialization)."""
    executor = LocalExecutor()

    result = await executor.execute("import json; print(json.dumps({'key': 'value'}))")

    assert result.error is None
    assert '{"key": "value"}' in result.output


@pytest.mark.asyncio
async def test_local_executor_stdout_capture():
    """Test that stdout is properly captured."""
    executor = LocalExecutor()

    result = await executor.execute("print('Line 1'); print('Line 2')")

    assert result.error is None
    assert "Line 1" in result.stdout
    assert "Line 2" in result.stdout


@pytest.mark.asyncio
async def test_local_executor_stderr_capture():
    """Test that stderr is properly captured on errors."""
    executor = LocalExecutor()

    # This should produce an error
    result = await executor.execute("import sys; sys.stderr.write('Error message\\n'); 1/0")

    assert result.error is not None
    assert "ZeroDivisionError" in result.error


@pytest.mark.asyncio
async def test_local_executor_complex_data_types():
    """Test with complex data types (lists, dicts, etc.)."""
    executor = LocalExecutor()

    # Inject complex data
    data = {"items": [1, 2, 3, 4, 5], "metadata": {"source": "test", "count": 5}}

    await executor.send_variables({"data": data})

    # Use the data
    result = await executor.execute("total = sum(data['items']); print(total)")

    assert result.error is None
    assert "15" in result.output
