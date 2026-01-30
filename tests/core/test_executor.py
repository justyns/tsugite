"""Tests for code execution backends."""

import pytest

from tsugite.core.executor import ExecutionResult, LocalExecutor, _summarize_variable


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


@pytest.mark.asyncio
async def test_local_executor_last_expression_simple_value():
    """Test that last expression value is automatically displayed (REPL-like)."""
    executor = LocalExecutor()

    # Code ending with a simple expression
    result = await executor.execute("x = 42\nx")

    assert result.error is None
    assert "42" in result.output


@pytest.mark.asyncio
async def test_local_executor_last_expression_dict():
    """Test that last expression dict is pretty-printed."""
    executor = LocalExecutor()

    # Code ending with a dict expression
    result = await executor.execute("data = {'name': 'test', 'value': 123}\ndata")

    assert result.error is None
    assert "'name': 'test'" in result.output
    assert "'value': 123" in result.output


@pytest.mark.asyncio
async def test_local_executor_last_expression_list():
    """Test that last expression list is pretty-printed."""
    executor = LocalExecutor()

    # Code ending with a list expression
    result = await executor.execute("items = [1, 2, 3, 4, 5]\nitems")

    assert result.error is None
    assert "[1, 2, 3, 4, 5]" in result.output


@pytest.mark.asyncio
async def test_local_executor_last_expression_none_not_displayed():
    """Test that None values are not displayed."""
    executor = LocalExecutor()

    # Code ending with None expression
    result = await executor.execute("x = None\nx")

    assert result.error is None
    # Should be empty - None should not be displayed
    assert result.output == ""


@pytest.mark.asyncio
async def test_local_executor_no_last_expression():
    """Test that code ending with statement (not expression) works normally."""
    executor = LocalExecutor()

    # Code ending with a statement (assignment), not expression
    result = await executor.execute("x = 5\ny = 10")

    assert result.error is None
    # Should be empty - no expression to display
    assert result.output == ""


@pytest.mark.asyncio
async def test_local_executor_last_expression_with_print():
    """Test that both print and last expression work together."""
    executor = LocalExecutor()

    # Code with print statement and ending with expression
    result = await executor.execute("print('Debug: calculating')\nx = 5 + 3\nx")

    assert result.error is None
    assert "Debug: calculating" in result.output
    assert "8" in result.output


@pytest.mark.asyncio
async def test_local_executor_last_expression_function_call():
    """Test that function call as last expression displays result."""
    executor = LocalExecutor()

    # Define a function and call it as last expression
    code = """
def get_data():
    return {'status': 'success', 'value': 42}

get_data()
"""
    result = await executor.execute(code)

    assert result.error is None
    assert "'status': 'success'" in result.output
    assert "'value': 42" in result.output


@pytest.mark.asyncio
async def test_local_executor_single_expression():
    """Test that a single expression is displayed."""
    executor = LocalExecutor()

    # Just a single expression, no setup
    result = await executor.execute("2 + 2")

    assert result.error is None
    assert "4" in result.output


# ============================================================================
# XML Execution Result Tests
# ============================================================================


class TestSummarizeVariable:
    """Tests for the _summarize_variable helper function."""

    def test_dict(self):
        assert _summarize_variable({"a": 1, "b": 2, "c": 3}) == "dict(3 keys)"

    def test_list(self):
        assert _summarize_variable([1, 2, 3, 4, 5]) == "list(5 items)"

    def test_tuple(self):
        assert _summarize_variable((1, 2, 3)) == "tuple(3 items)"

    def test_set(self):
        assert _summarize_variable({1, 2, 3, 4}) == "set(4 items)"

    def test_str(self):
        assert _summarize_variable("hello world") == "str(11 chars)"

    def test_bytes(self):
        assert _summarize_variable(b"test") == "bytes(4 bytes)"

    def test_int(self):
        assert _summarize_variable(42) == "int"

    def test_float(self):
        assert _summarize_variable(3.14) == "float"

    def test_bool(self):
        assert _summarize_variable(True) == "bool"

    def test_none(self):
        assert _summarize_variable(None) == "NoneType"

    def test_custom_class(self):
        class MyClass:
            pass

        assert _summarize_variable(MyClass()) == "MyClass"


class TestExecutionResultToXml:
    """Tests for ExecutionResult.to_xml() method."""

    def test_success_basic(self):
        result = ExecutionResult(
            output="Hello, world!",
            error=None,
            stdout="Hello, world!",
            stderr="",
        )
        xml = result.to_xml()
        assert '<tsugite_execution_result status="success">' in xml
        assert "<output>Hello, world!</output>" in xml
        assert "</tsugite_execution_result>" in xml
        assert "<error>" not in xml

    def test_success_with_duration(self):
        result = ExecutionResult(
            output="test",
            error=None,
            stdout="test",
            stderr="",
        )
        xml = result.to_xml(duration_ms=142)
        assert 'duration_ms="142"' in xml

    def test_error_with_traceback(self):
        result = ExecutionResult(
            output="partial output",
            error="FileNotFoundError: /tmp/missing.txt",
            stdout="partial output",
            stderr='Traceback:\n  File "test.py", line 1\nFileNotFoundError: /tmp/missing.txt',
        )
        xml = result.to_xml()
        assert '<tsugite_execution_result status="error">' in xml
        assert "<error>FileNotFoundError: /tmp/missing.txt</error>" in xml
        assert "<traceback>" in xml
        assert "</traceback>" in xml

    def test_xml_escaping(self):
        result = ExecutionResult(
            output="<script>alert('xss')</script>",
            error=None,
            stdout="<script>alert('xss')</script>",
            stderr="",
        )
        xml = result.to_xml()
        assert "&lt;script&gt;" in xml
        assert "<script>" not in xml.replace("&lt;script&gt;", "")

    def test_variables_set(self):
        result = ExecutionResult(
            output="",
            error=None,
            stdout="",
            stderr="",
            variables_set={"config": "dict(3 keys)", "files": "list(5 items)"},
        )
        xml = result.to_xml()
        assert "<variables_set>" in xml
        assert "config=dict(3 keys)" in xml
        assert "files=list(5 items)" in xml

    def test_final_answer(self):
        result = ExecutionResult(
            output="",
            error=None,
            stdout="",
            stderr="",
            final_answer="The answer is 42",
        )
        xml = result.to_xml()
        assert "<final_answer>The answer is 42</final_answer>" in xml

    def test_truncation(self):
        large_output = "x" * (100 * 1024)  # 100KB
        result = ExecutionResult(
            output=large_output,
            error=None,
            stdout=large_output,
            stderr="",
        )
        xml = result.to_xml(max_output_kb=50)
        assert 'truncated="true"' in xml
        assert result.truncated is True
        # Output should be truncated to ~50KB
        assert len(xml) < len(large_output)

    def test_empty_output_still_present(self):
        result = ExecutionResult(
            output="",
            error=None,
            stdout="",
            stderr="",
        )
        xml = result.to_xml()
        assert "<output></output>" in xml

    def test_unicode_in_output(self):
        result = ExecutionResult(
            output="Hello 世界 🌍",
            error=None,
            stdout="Hello 世界 🌍",
            stderr="",
        )
        xml = result.to_xml()
        assert "Hello 世界 🌍" in xml

    def test_variable_name_escaping(self):
        result = ExecutionResult(
            output="",
            error=None,
            stdout="",
            stderr="",
            variables_set={"my<var>": "str(10 chars)"},
        )
        xml = result.to_xml()
        assert "my&lt;var&gt;=" in xml


@pytest.mark.asyncio
async def test_local_executor_variable_tracking():
    """Test that new variables are tracked in variables_set."""
    executor = LocalExecutor()

    result = await executor.execute("x = 42\ny = 'hello'\nz = [1, 2, 3]")

    assert result.error is None
    assert "x" in result.variables_set
    assert "y" in result.variables_set
    assert "z" in result.variables_set
    assert result.variables_set["x"] == "int"
    assert result.variables_set["y"] == "str(5 chars)"
    assert result.variables_set["z"] == "list(3 items)"


@pytest.mark.asyncio
async def test_local_executor_private_variables_excluded():
    """Test that private variables (starting with _) are not tracked."""
    executor = LocalExecutor()

    result = await executor.execute("x = 1\n_private = 2\n__dunder = 3")

    assert result.error is None
    assert "x" in result.variables_set
    assert "_private" not in result.variables_set
    assert "__dunder" not in result.variables_set


@pytest.mark.asyncio
async def test_local_executor_variables_tracked_on_error():
    """Test that variables set before an error are still captured."""
    executor = LocalExecutor()

    result = await executor.execute("x = 42\ny = 'set before error'\n1/0")

    assert result.error is not None
    assert "ZeroDivisionError" in result.error
    # Variables set before the error should still be captured
    assert "x" in result.variables_set
    assert "y" in result.variables_set


@pytest.mark.asyncio
async def test_local_executor_xml_observation_success():
    """Test that to_xml() produces valid XML on success."""
    executor = LocalExecutor()

    result = await executor.execute("x = {'a': 1, 'b': 2}\nprint('hello')")
    xml = result.to_xml(duration_ms=50)

    assert '<tsugite_execution_result status="success"' in xml
    assert "<output>hello" in xml
    assert "x=dict(2 keys)" in xml
    assert "</tsugite_execution_result>" in xml


@pytest.mark.asyncio
async def test_local_executor_xml_observation_error():
    """Test that to_xml() produces valid XML on error."""
    executor = LocalExecutor()

    result = await executor.execute("print('before error')\nundefined_var")
    xml = result.to_xml(duration_ms=25)

    assert '<tsugite_execution_result status="error"' in xml
    assert "<output>before error" in xml
    assert "<error>NameError:" in xml
    assert "</tsugite_execution_result>" in xml


@pytest.mark.asyncio
async def test_local_executor_xml_with_final_answer():
    """Test that final_answer appears in XML."""
    executor = LocalExecutor()

    result = await executor.execute("final_answer('completed!')")
    xml = result.to_xml()

    assert "<final_answer>completed!</final_answer>" in xml
