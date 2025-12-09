"""Tests for MCP server functionality."""

import pytest
from typer.testing import CliRunner

from tsugite.cli import app
from tsugite.mcp_server import (
    EXPOSED_TOOLS,
    _python_type_to_json_schema,
    execute_tool,
    get_mcp_tools,
)


class TestTypeConversion:
    """Tests for Python type to JSON schema conversion."""

    def test_string_type(self):
        """Test string type conversion."""
        result = _python_type_to_json_schema(str)
        assert result == {"type": "string"}

    def test_int_type(self):
        """Test int type conversion."""
        result = _python_type_to_json_schema(int)
        assert result == {"type": "integer"}

    def test_float_type(self):
        """Test float type conversion."""
        result = _python_type_to_json_schema(float)
        assert result == {"type": "number"}

    def test_bool_type(self):
        """Test bool type conversion."""
        result = _python_type_to_json_schema(bool)
        assert result == {"type": "boolean"}

    def test_list_type(self):
        """Test list type conversion."""
        result = _python_type_to_json_schema(list)
        assert result == {"type": "array"}

    def test_dict_type(self):
        """Test dict type conversion."""
        result = _python_type_to_json_schema(dict)
        assert result == {"type": "object"}


class TestGetMCPTools:
    """Tests for MCP tool generation."""

    def test_get_mcp_tools_returns_list(self, memory_tools, agents_tools):
        """Test that get_mcp_tools returns a list of Tools."""
        tools = get_mcp_tools()
        assert isinstance(tools, list)
        assert len(tools) > 0

    def test_exposed_tools_are_present(self, memory_tools, agents_tools):
        """Test that all exposed tools are in the list."""
        tools = get_mcp_tools()
        tool_names = [t.name for t in tools]

        for expected in EXPOSED_TOOLS:
            assert expected in tool_names, f"Expected tool '{expected}' not found"

    def test_tools_have_required_fields(self, memory_tools, agents_tools):
        """Test that each tool has name, description, and inputSchema."""
        tools = get_mcp_tools()

        for tool in tools:
            assert tool.name, "Tool missing name"
            assert tool.description, f"Tool {tool.name} missing description"
            assert tool.inputSchema, f"Tool {tool.name} missing inputSchema"
            assert "type" in tool.inputSchema, f"Tool {tool.name} schema missing type"

    def test_input_schema_structure(self, memory_tools, agents_tools):
        """Test that inputSchema has proper structure."""
        tools = get_mcp_tools()

        for tool in tools:
            schema = tool.inputSchema
            assert schema.get("type") == "object"
            assert "properties" in schema
            assert "required" in schema


class TestExecuteTool:
    """Tests for tool execution."""

    @pytest.mark.asyncio
    async def test_execute_unknown_tool(self):
        """Test executing a tool not in EXPOSED_TOOLS."""
        result = await execute_tool("unknown_tool", {})
        assert len(result) == 1
        assert "error" in result[0].text

    @pytest.mark.asyncio
    async def test_execute_list_agents(self, memory_tools, agents_tools, tmp_path, monkeypatch):
        """Test executing list_agents tool."""
        monkeypatch.chdir(tmp_path)
        result = await execute_tool("list_agents", {})
        assert len(result) == 1
        # Should not error
        assert "error" not in result[0].text.lower() or "not found" not in result[0].text.lower()


class TestCLIServeCommand:
    """Tests for serve CLI command."""

    def setup_method(self):
        """Set up test runner."""
        self.runner = CliRunner()

    def test_serve_help(self):
        """Test serve command help."""
        result = self.runner.invoke(app, ["serve", "--help"])
        assert result.exit_code == 0
        assert "mcp" in result.stdout

    def test_serve_mcp_help(self):
        """Test serve mcp command help."""
        result = self.runner.invoke(app, ["serve", "mcp", "--help"])
        assert result.exit_code == 0
        assert "--info" in result.stdout

    def test_serve_mcp_info(self):
        """Test serve mcp --info command."""
        result = self.runner.invoke(app, ["serve", "mcp", "--info"])
        assert result.exit_code == 0
        assert "Tsugite MCP Server" in result.stdout
        assert "memory_store" in result.stdout


class TestExposedTools:
    """Tests for EXPOSED_TOOLS configuration."""

    def test_exposed_tools_not_empty(self):
        """Test that EXPOSED_TOOLS is not empty."""
        assert len(EXPOSED_TOOLS) > 0

    def test_exposed_tools_contains_memory_tools(self):
        """Test that memory tools are exposed."""
        memory_tools = [
            "memory_store",
            "memory_search",
            "memory_list",
            "memory_get",
            "memory_update",
            "memory_delete",
            "memory_count",
        ]
        for tool in memory_tools:
            assert tool in EXPOSED_TOOLS

    def test_exposed_tools_contains_agent_tools(self):
        """Test that agent tools are exposed."""
        assert "spawn_agent" in EXPOSED_TOOLS
        assert "list_agents" in EXPOSED_TOOLS
