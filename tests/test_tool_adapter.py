"""Tests for the smolagents tool adapter."""

import pytest
from tsugite.tool_adapter import create_smolagents_tool_from_tsugite, get_smolagents_tools
from tsugite.tools import tool


@pytest.fixture
def sample_tool(reset_tool_registry):
    """Create a sample tool for testing."""

    @tool
    def sample_function(arg1: str, arg2: int = 5) -> str:
        """Sample function for testing

        Args:
            arg1: First argument
            arg2: Second argument with default

        Returns:
            str: Formatted result
        """
        return f"arg1={arg1}, arg2={arg2}"

    return "sample_function"


def test_create_smolagents_tool_positional_args(sample_tool):
    """Test that smolagents tools handle positional arguments correctly."""
    smol_tool = create_smolagents_tool_from_tsugite(sample_tool)

    # Test with positional args (how smolagents typically calls tools)
    result = smol_tool("hello", 10)
    assert result == "arg1=hello, arg2=10"


def test_create_smolagents_tool_keyword_args(sample_tool):
    """Test that smolagents tools handle keyword arguments correctly."""
    smol_tool = create_smolagents_tool_from_tsugite(sample_tool)

    # Test with keyword args
    result = smol_tool(arg1="world", arg2=20)
    assert result == "arg1=world, arg2=20"


def test_create_smolagents_tool_mixed_args(sample_tool):
    """Test that smolagents tools handle mixed positional and keyword arguments."""
    smol_tool = create_smolagents_tool_from_tsugite(sample_tool)

    # Test with mixed args
    result = smol_tool("mixed", arg2=30)
    assert result == "arg1=mixed, arg2=30"


def test_create_smolagents_tool_default_args(sample_tool):
    """Test that smolagents tools handle default arguments correctly."""
    smol_tool = create_smolagents_tool_from_tsugite(sample_tool)

    # Test with only required arg (using default for arg2)
    result = smol_tool("test")
    assert result == "arg1=test, arg2=5"


def test_create_smolagents_tool_error_handling(sample_tool):
    """Test that smolagents tools handle errors correctly."""

    @tool
    def error_tool(value: str) -> str:
        """Tool that raises an error

        Args:
            value: Input value

        Returns:
            str: Output
        """
        raise ValueError(f"Test error: {value}")

    smol_tool = create_smolagents_tool_from_tsugite("error_tool")

    # Tool should catch the error and return error message
    result = smol_tool("test")
    assert "failed" in result.lower()
    assert "test error" in result.lower()


def test_get_smolagents_tools(reset_tool_registry):
    """Test getting multiple smolagents tools at once."""

    @tool
    def tool1(arg: str) -> str:
        """First tool

        Args:
            arg: Input argument

        Returns:
            str: Output
        """
        return f"tool1: {arg}"

    @tool
    def tool2(arg: str) -> str:
        """Second tool

        Args:
            arg: Input argument

        Returns:
            str: Output
        """
        return f"tool2: {arg}"

    # Get both tools
    smol_tools = get_smolagents_tools(["tool1", "tool2"])

    assert len(smol_tools) == 2

    # Test that they work
    assert smol_tools[0]("test") == "tool1: test"
    assert smol_tools[1]("test") == "tool2: test"


def test_read_file_tool_positional(file_tools):
    """Test read_file tool with positional argument (real-world usage)."""
    smol_read_file = create_smolagents_tool_from_tsugite("read_file")

    # Should work with positional arg
    result = smol_read_file("README.md")
    assert isinstance(result, str)
    assert len(result) > 0
    assert "tsugite" in result.lower()
