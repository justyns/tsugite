"""Tests for the tool registry system."""

import pytest
from tsugite.tools import tool, get_tool, call_tool, list_tools, describe_tool, ToolInfo


def test_tool_decorator(reset_tool_registry):
    """Test that the @tool decorator registers functions correctly."""

    @tool
    def test_function(arg1: str, arg2: int = 42) -> str:
        """A test function."""
        return f"Result: {arg1}, {arg2}"

    # Check tool was registered
    assert "test_function" in list_tools()

    # Check tool info
    tool_info = get_tool("test_function")
    assert isinstance(tool_info, ToolInfo)
    assert tool_info.name == "test_function"
    assert tool_info.description == "A test function."

    # Check parameters
    params = tool_info.parameters
    assert "arg1" in params
    assert "arg2" in params
    assert params["arg1"]["required"] is True
    assert params["arg2"]["required"] is False
    assert params["arg2"]["default"] == 42


def test_tool_registration_without_annotations(reset_tool_registry):
    """Test registering a tool without type annotations."""

    @tool
    def simple_tool(name):
        """Simple tool without annotations."""
        return f"Hello {name}"

    tool_info = get_tool("simple_tool")
    assert tool_info.parameters["name"]["type"] == str  # Default type


def test_tool_registration_without_docstring(reset_tool_registry):
    """Test registering a tool without docstring."""

    @tool
    def undocumented_tool():
        return "result"

    tool_info = get_tool("undocumented_tool")
    assert tool_info.description == "No description available"


def test_get_nonexistent_tool(reset_tool_registry):
    """Test getting a tool that doesn't exist."""
    with pytest.raises(ValueError, match="Tool 'nonexistent' not found"):
        get_tool("nonexistent")


def test_call_tool_success(reset_tool_registry):
    """Test successfully calling a tool."""

    @tool
    def add_numbers(a: int, b: int = 10) -> int:
        """Add two numbers."""
        return a + b

    # Test with all args
    result = call_tool("add_numbers", a=5, b=3)
    assert result == 8

    # Test with default
    result = call_tool("add_numbers", a=5)
    assert result == 15


def test_call_tool_missing_required_param(reset_tool_registry):
    """Test calling a tool with missing required parameter."""

    @tool
    def requires_param(required_arg: str) -> str:
        """Tool that requires a parameter."""
        return required_arg.upper()

    with pytest.raises(ValueError, match="Missing required parameter 'required_arg'"):
        call_tool("requires_param")


def test_call_tool_runtime_error(reset_tool_registry):
    """Test tool that raises an exception during execution."""

    @tool
    def failing_tool() -> str:
        """A tool that always fails."""
        raise RuntimeError("Something went wrong")

    with pytest.raises(RuntimeError, match="Tool 'failing_tool' failed: Something went wrong"):
        call_tool("failing_tool")


def test_list_tools_empty(reset_tool_registry):
    """Test listing tools when registry is empty."""
    assert list_tools() == []


def test_list_tools_multiple(reset_tool_registry):
    """Test listing multiple tools."""

    @tool
    def tool_a():
        return "a"

    @tool
    def tool_b():
        return "b"

    tools = list_tools()
    assert "tool_a" in tools
    assert "tool_b" in tools
    assert len(tools) == 2


def test_describe_tool(reset_tool_registry):
    """Test tool description generation."""

    @tool
    def example_tool(name: str, count: int = 5, active: bool = True) -> str:
        """An example tool for testing."""
        return f"Processed {name} {count} times, active={active}"

    description = describe_tool("example_tool")

    assert "example_tool: An example tool for testing." in description
    assert "Parameters:" in description
    assert "name: str (required)" in description
    assert "count: int (default: 5)" in description
    assert "active: bool (default: True)" in description


def test_tool_with_complex_types(reset_tool_registry):
    """Test tool with complex parameter types."""

    @tool
    def complex_tool(items: list, metadata: dict = None) -> dict:
        """Tool with complex types."""
        return {"items": len(items), "metadata": metadata}

    tool_info = get_tool("complex_tool")
    assert tool_info.parameters["items"]["type"] == list
    assert tool_info.parameters["metadata"]["type"] == dict
    assert tool_info.parameters["metadata"]["default"] is None


def test_multiple_tool_registrations(reset_tool_registry):
    """Test that multiple tools can be registered and work independently."""

    @tool
    def math_add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    @tool
    def string_upper(text: str) -> str:
        """Convert text to uppercase."""
        return text.upper()

    @tool
    def bool_toggle(value: bool = False) -> bool:
        """Toggle a boolean value."""
        return not value

    # Test all tools work
    assert call_tool("math_add", a=3, b=7) == 10
    assert call_tool("string_upper", text="hello") == "HELLO"
    assert call_tool("bool_toggle") is True
    assert call_tool("bool_toggle", value=True) is False

    # Test tool listing
    tools = list_tools()
    assert len(tools) == 3
    assert all(name in tools for name in ["math_add", "string_upper", "bool_toggle"])
