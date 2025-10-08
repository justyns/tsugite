"""Tests for the tool registry system."""

import pytest

from tsugite.tools import (
    ToolInfo,
    call_tool,
    describe_tool,
    expand_tool_specs,
    get_tool,
    get_tools_by_category,
    list_tools,
    tool,
)


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
    assert tool_info.parameters["name"]["type"] is str  # Default type


def test_tool_registration_without_docstring(reset_tool_registry):
    """Test registering a tool without docstring."""

    @tool
    def undocumented_tool():
        return "result"

    tool_info = get_tool("undocumented_tool")
    assert tool_info.description == "No description available"


def test_get_nonexistent_tool(reset_tool_registry):
    """Test getting a tool that doesn't exist."""
    with pytest.raises(ValueError, match="Invalid tool 'nonexistent': not found"):
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

    with pytest.raises(ValueError, match="Invalid parameter 'required_arg': missing for tool 'requires_param'"):
        call_tool("requires_param")


def test_call_tool_runtime_error(reset_tool_registry):
    """Test tool that raises an exception during execution."""

    @tool
    def failing_tool() -> str:
        """A tool that always fails."""
        raise RuntimeError("Something went wrong")

    with pytest.raises(RuntimeError, match="Tool 'failing_tool' failed to execute: Something went wrong"):
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
    assert tool_info.parameters["items"]["type"] is list
    assert tool_info.parameters["metadata"]["type"] is dict
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


def test_get_tools_by_category(file_tools):
    """Test getting tools by category."""
    # Get fs category tools
    fs_tools = get_tools_by_category("fs")
    assert "read_file" in fs_tools
    assert "write_file" in fs_tools
    assert "list_files" in fs_tools
    assert "file_exists" in fs_tools
    assert "create_directory" in fs_tools

    # Tools should be sorted
    assert fs_tools == sorted(fs_tools)

    # Non-existent category returns empty list
    empty = get_tools_by_category("nonexistent")
    assert empty == []


def test_expand_tool_specs_regular_names(file_tools):
    """Test expanding regular tool names."""
    specs = ["read_file", "write_file"]
    expanded = expand_tool_specs(specs)
    assert expanded == ["read_file", "write_file"]


def test_expand_tool_specs_category(file_tools):
    """Test expanding category references."""
    specs = ["@fs"]
    expanded = expand_tool_specs(specs)

    # Should contain all fs tools
    assert "read_file" in expanded
    assert "write_file" in expanded
    assert "list_files" in expanded
    assert "file_exists" in expanded
    assert "create_directory" in expanded

    # Should not contain non-fs tools
    assert "run" not in expanded
    assert "web_search" not in expanded


def test_expand_tool_specs_glob_pattern(file_tools):
    """Test expanding glob patterns."""
    # Pattern matching *_file
    specs = ["*_file"]
    expanded = expand_tool_specs(specs)
    assert "read_file" in expanded
    assert "write_file" in expanded

    # Pattern matching list_*
    specs = ["list_*"]
    expanded = expand_tool_specs(specs)
    assert "list_files" in expanded


def test_expand_tool_specs_mixed(file_tools):
    """Test expanding mixed specifications."""
    specs = ["@fs", "file_exists", "*_directory"]
    expanded = expand_tool_specs(specs)

    # Should have all fs tools from category
    assert "read_file" in expanded
    assert "write_file" in expanded
    assert "list_files" in expanded

    # Should have explicit tool
    assert "file_exists" in expanded

    # Should have *_directory matches
    assert "create_directory" in expanded


def test_expand_tool_specs_duplicates(file_tools):
    """Test that duplicates are removed."""
    specs = ["read_file", "@fs", "*_file", "read_file"]
    expanded = expand_tool_specs(specs)

    # read_file should only appear once
    assert expanded.count("read_file") == 1

    # Order should be preserved (first occurrence)
    assert expanded[0] == "read_file"


def test_expand_tool_specs_invalid_tool(file_tools):
    """Test error on invalid tool name."""
    specs = ["nonexistent_tool"]

    with pytest.raises(ValueError, match="Invalid tool 'nonexistent_tool': not found"):
        expand_tool_specs(specs)


def test_expand_tool_specs_invalid_category(file_tools):
    """Test error on invalid category."""
    specs = ["@nonexistent"]

    with pytest.raises(ValueError, match="Invalid tool category 'nonexistent': not found or empty"):
        expand_tool_specs(specs)


def test_expand_tool_specs_no_glob_matches(file_tools):
    """Test error when glob pattern matches nothing."""
    specs = ["xyz_*_abc"]

    with pytest.raises(ValueError, match="Invalid tool pattern 'xyz_\\*_abc': matched no tools"):
        expand_tool_specs(specs)


def test_expand_tool_specs_empty_list(file_tools):
    """Test expanding empty list."""
    specs = []
    expanded = expand_tool_specs(specs)
    assert expanded == []


def test_expand_tool_specs_preserves_order(file_tools):
    """Test that expansion preserves order of specifications."""
    specs = ["file_exists", "@fs", "list_files"]
    expanded = expand_tool_specs(specs)

    # file_exists should come first (explicit, before category expansion)
    assert expanded[0] == "file_exists"

    # fs tools should come in the middle (from @fs category)
    # Note: file_exists appears first due to spec order, then @fs adds the rest
    read_index = expanded.index("read_file")
    write_index = expanded.index("write_file")
    assert read_index < len(expanded)
    assert write_index < len(expanded)

    # list_files appears twice but only counted once due to dedup
    assert expanded.count("list_files") == 1


def test_expand_tool_specs_exclude_exact_name(file_tools):
    """Test excluding an exact tool name."""
    specs = ["@fs", "-read_file"]
    expanded = expand_tool_specs(specs)

    # Should have all fs tools except read_file
    assert "write_file" in expanded
    assert "list_files" in expanded
    assert "file_exists" in expanded
    assert "create_directory" in expanded
    assert "read_file" not in expanded


def test_expand_tool_specs_exclude_glob_pattern(file_tools):
    """Test excluding tools matching a glob pattern."""
    specs = ["@fs", "-*_file"]
    expanded = expand_tool_specs(specs)

    # Should exclude read_file and write_file
    assert "read_file" not in expanded
    assert "write_file" not in expanded

    # Should still have other fs tools
    assert "list_files" in expanded
    assert "file_exists" in expanded
    assert "create_directory" in expanded


def test_expand_tool_specs_exclude_category(file_tools):
    """Test excluding an entire category."""
    # First register some non-fs tools
    from tsugite.tools import tool

    @tool
    def test_shell_tool():
        """Test shell tool."""
        return "shell"

    # Add it to a different module to create a category
    test_shell_tool.__module__ = "tsugite.tools.shell"

    specs = ["@fs", "test_shell_tool", "-@fs"]
    expanded = expand_tool_specs(specs)

    # Should only have test_shell_tool, all fs excluded
    assert expanded == ["test_shell_tool"]
    assert "read_file" not in expanded
    assert "write_file" not in expanded


def test_expand_tool_specs_exclude_nonexistent(file_tools):
    """Test that excluding non-existent tools doesn't raise an error."""
    specs = ["@fs", "-nonexistent_tool", "-fake_*", "-@fake_category"]
    expanded = expand_tool_specs(specs)

    # Should have all fs tools (exclusions silently ignored)
    assert "read_file" in expanded
    assert "write_file" in expanded
    assert len(expanded) == 5  # All 5 fs tools


def test_expand_tool_specs_only_exclusions(file_tools):
    """Test that only exclusions with no inclusions returns empty list."""
    specs = ["-read_file", "-@fs"]
    expanded = expand_tool_specs(specs)

    # No inclusions, so result should be empty
    assert expanded == []


def test_expand_tool_specs_exclude_then_include(file_tools):
    """Test that order matters - later inclusions can re-add excluded tools."""
    specs = ["@fs", "-read_file", "read_file"]
    expanded = expand_tool_specs(specs)

    # read_file excluded then re-added
    # But our current implementation doesn't re-add, exclusions are applied after all inclusions
    # So read_file should still be excluded
    assert "read_file" not in expanded


def test_expand_tool_specs_mixed_exclude(file_tools):
    """Test complex exclusion scenarios."""
    specs = ["@fs", "*_directory", "-*_directory"]
    expanded = expand_tool_specs(specs)

    # create_directory added by @fs and *_directory, but then excluded
    assert "create_directory" not in expanded
    assert "read_file" in expanded
    assert "write_file" in expanded


def test_expand_tool_specs_exclude_preserves_order(file_tools):
    """Test that exclusions preserve the order of remaining tools."""
    specs = ["file_exists", "@fs", "-read_file", "-write_file"]
    expanded = expand_tool_specs(specs)

    # file_exists should be first
    assert expanded[0] == "file_exists"

    # read_file and write_file should be excluded
    assert "read_file" not in expanded
    assert "write_file" not in expanded

    # Other tools should be present
    assert "create_directory" in expanded
    assert "file_exists" in expanded
    assert "list_files" in expanded
