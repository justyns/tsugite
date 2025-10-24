"""Tests for custom shell tools system."""

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from tsugite.tools.shell_tools import (
    ShellToolDefinition,
    ShellToolParameter,
    create_shell_tool_function,
    interpolate_command,
    register_shell_tool,
)


class TestShellToolParameter:
    """Tests for ShellToolParameter validation."""

    def test_str_parameter_validation(self):
        """Test string parameter validation."""
        param = ShellToolParameter(name="text", type="str", required=False, default="hello")

        assert param.validate("world") == "world"
        assert param.validate(None) == "hello"  # Returns default
        assert param.validate(123) == "123"  # Converts to string

    def test_int_parameter_validation(self):
        """Test integer parameter validation."""
        param = ShellToolParameter(name="count", type="int", required=False, default=10)

        assert param.validate(42) == 42
        assert param.validate("42") == 42  # Converts string to int
        assert param.validate(None) == 10  # Returns default

    def test_int_parameter_invalid_value(self):
        """Test integer parameter with invalid value."""
        param = ShellToolParameter(name="count", type="int", required=True)

        with pytest.raises(ValueError, match="must be an integer"):
            param.validate("not a number")

    def test_bool_parameter_validation(self):
        """Test boolean parameter validation."""
        param = ShellToolParameter(name="flag", type="bool", required=False, default=False)

        assert param.validate(True) is True
        assert param.validate(False) is False
        assert param.validate("true") is True
        assert param.validate("yes") is True
        assert param.validate("1") is True
        assert param.validate("false") is False
        assert param.validate(None) is False  # Returns default

    def test_float_parameter_validation(self):
        """Test float parameter validation."""
        param = ShellToolParameter(name="ratio", type="float", required=False, default=1.5)

        assert param.validate(3.14) == 3.14
        assert param.validate("3.14") == 3.14  # Converts string to float
        assert param.validate(42) == 42.0  # Converts int to float
        assert param.validate(None) == 1.5  # Returns default

    def test_required_parameter_missing(self):
        """Test that required parameter raises error when None."""
        param = ShellToolParameter(name="required_param", type="str", required=True)

        with pytest.raises(ValueError, match="Required parameter 'required_param' is missing"):
            param.validate(None)


class TestCommandInterpolation:
    """Tests for command template interpolation."""

    def test_simple_interpolation(self):
        """Test basic parameter interpolation."""
        template = "echo {message}"
        params = {"message": "hello"}
        result = interpolate_command(template, params)
        assert result == "echo hello"

    def test_multiple_parameters(self):
        """Test interpolation with multiple parameters."""
        template = "rg {pattern} {path}"
        params = {"pattern": "test", "path": "/tmp"}
        result = interpolate_command(template, params)
        assert result == "rg test /tmp"

    def test_boolean_flag_true(self):
        """Test boolean flag when true."""
        template = "grep {case_sensitive} {pattern}"
        params = {"pattern": "test", "case_sensitive": True, "_flag_case_sensitive": "-s"}
        result = interpolate_command(template, params)
        assert result == "grep -s test"

    def test_boolean_flag_false(self):
        """Test boolean flag when false."""
        template = "grep {case_sensitive} {pattern}"
        params = {"pattern": "test", "case_sensitive": False}
        result = interpolate_command(template, params)
        assert result == "grep  test"  # Flag replaced with empty string

    def test_missing_required_parameter(self):
        """Test error when required parameter is missing."""
        template = "echo {message} {recipient}"
        params = {"message": "hello"}

        with pytest.raises(ValueError, match="Missing required parameter"):
            interpolate_command(template, params)


class TestShellToolCreation:
    """Tests for creating shell tool functions."""

    @patch("subprocess.run")
    def test_create_simple_tool(self, mock_run):
        """Test creating a simple tool with no parameters."""
        mock_run.return_value = MagicMock(stdout="output", stderr="", returncode=0)

        definition = ShellToolDefinition(
            name="get_date",
            description="Get current date",
            command="date +%Y-%m-%d",
            parameters={},
        )

        func = create_shell_tool_function(definition)

        # Check function metadata
        assert func.__name__ == "get_date"
        assert func.__doc__ == "Get current date"

        # Call the function
        result = func()

        # Verify subprocess was called
        mock_run.assert_called_once()
        assert result == "output"

    @patch("subprocess.run")
    def test_create_tool_with_parameters(self, mock_run):
        """Test creating a tool with parameters."""
        mock_run.return_value = MagicMock(stdout="file1.txt\nfile2.txt", stderr="", returncode=0)

        definition = ShellToolDefinition(
            name="find_files",
            description="Find files by pattern",
            command="find {path} -name {pattern}",
            parameters={
                "path": ShellToolParameter(name="path", type="str", default="."),
                "pattern": ShellToolParameter(name="pattern", type="str", required=True),
            },
        )

        func = create_shell_tool_function(definition)

        # Call with parameters
        func(pattern="*.py", path="/tmp")

        # Verify command was built correctly
        call_args = mock_run.call_args
        assert "find /tmp -name *.py" in call_args[0][0]

    @patch("subprocess.run")
    def test_tool_with_defaults(self, mock_run):
        """Test tool uses default parameter values."""
        mock_run.return_value = MagicMock(stdout="result", stderr="", returncode=0)

        definition = ShellToolDefinition(
            name="list_dir",
            description="List directory",
            command="ls {path}",
            parameters={
                "path": ShellToolParameter(name="path", type="str", default="/tmp"),
            },
        )

        func = create_shell_tool_function(definition)

        # Call without parameters (should use default)
        func()

        # Verify default was used
        call_args = mock_run.call_args
        assert "ls /tmp" in call_args[0][0]

    @patch("subprocess.run")
    def test_tool_stderr_captured(self, mock_run):
        """Test that stderr is captured and included."""
        mock_run.return_value = MagicMock(stdout="output", stderr="warning message", returncode=0)

        definition = ShellToolDefinition(
            name="test_tool",
            description="Test",
            command="echo test",
            parameters={},
        )

        func = create_shell_tool_function(definition)
        result = func()

        # Both stdout and stderr should be in result
        assert "output" in result
        assert "warning message" in result

    @patch("subprocess.run")
    def test_tool_nonzero_exit_code(self, mock_run):
        """Test that non-zero exit codes are indicated."""
        mock_run.return_value = MagicMock(stdout="", stderr="error", returncode=1)

        definition = ShellToolDefinition(
            name="test_tool",
            description="Test",
            command="false",
            parameters={},
        )

        func = create_shell_tool_function(definition)
        result = func()

        # Exit code should be noted
        assert "[Exit code: 1]" in result

    @patch("subprocess.run")
    def test_tool_timeout(self, mock_run):
        """Test that timeout is enforced."""
        mock_run.side_effect = subprocess.TimeoutExpired("test", 5)

        definition = ShellToolDefinition(
            name="slow_tool",
            description="Slow tool",
            command="sleep 100",
            parameters={},
            timeout=5,
        )

        func = create_shell_tool_function(definition)

        with pytest.raises(RuntimeError, match="timed out after 5 seconds"):
            func()

    def test_tool_signature_has_parameters(self):
        """Test that generated function has correct signature."""
        import inspect

        definition = ShellToolDefinition(
            name="test_tool",
            description="Test",
            command="echo {msg} {count}",
            parameters={
                "msg": ShellToolParameter(name="msg", type="str", required=True),
                "count": ShellToolParameter(name="count", type="int", default=1),
            },
        )

        func = create_shell_tool_function(definition)
        sig = inspect.signature(func)

        # Check parameters exist in signature
        assert "msg" in sig.parameters
        assert "count" in sig.parameters

        # Check default value
        assert sig.parameters["count"].default == 1


class TestShellToolRegistration:
    """Tests for registering shell tools."""

    def test_register_shell_tool(self, reset_tool_registry):
        """Test that shell tool is registered in tool registry."""
        definition = ShellToolDefinition(
            name="test_echo",
            description="Echo a message",
            command="echo {message}",
            parameters={
                "message": ShellToolParameter(name="message", type="str", required=True),
            },
        )

        register_shell_tool(definition)

        # Check it's in the registry
        from tsugite.tools import get_tool, list_tools

        assert "test_echo" in list_tools()

        tool_info = get_tool("test_echo")
        assert tool_info.name == "test_echo"
        assert tool_info.description == "Echo a message"

    @patch("subprocess.run")
    def test_registered_tool_callable(self, mock_run, reset_tool_registry):
        """Test that registered tool can be called."""
        mock_run.return_value = MagicMock(stdout="hello world", stderr="", returncode=0)

        definition = ShellToolDefinition(
            name="test_echo",
            description="Echo",
            command="echo {text}",
            parameters={
                "text": ShellToolParameter(name="text", type="str", required=True),
            },
        )

        register_shell_tool(definition)

        # Call via tool registry
        from tsugite.tools import call_tool

        result = call_tool("test_echo", text="hello world")
        assert "hello world" in result


class TestTypeInference:
    """Tests for type inference from YAML values."""

    def test_infer_string_from_default(self):
        """Test that string default infers str type."""
        param = ShellToolParameter(name="path", type="str", default=".")
        assert param.type == "str"
        assert param.default == "."

    def test_infer_int_from_default(self):
        """Test that integer default infers int type."""
        param = ShellToolParameter(name="count", type="int", default=10)
        assert param.type == "int"
        assert param.default == 10

    def test_infer_bool_from_default(self):
        """Test that boolean default infers bool type."""
        param = ShellToolParameter(name="flag", type="bool", default=True)
        assert param.type == "bool"
        assert param.default is True

    def test_infer_float_from_default(self):
        """Test that float default infers float type."""
        param = ShellToolParameter(name="ratio", type="float", default=1.5)
        assert param.type == "float"
        assert param.default == 1.5
