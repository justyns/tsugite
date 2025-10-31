"""Tests for MCP client."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tsugite.core.tools import Tool
from tsugite.mcp_client import MCPClient, load_mcp_tools
from tsugite.mcp_config import MCPServerConfig


@pytest.fixture
def stdio_server_config():
    """Create stdio server config for testing."""
    return MCPServerConfig(
        name="test-server",
        command="test-command",
        args=["--arg1", "value1"],
        env={"ENV_VAR": "test"},
    )


@pytest.fixture
def http_server_config():
    """Create HTTP server config for testing."""
    return MCPServerConfig(name="test-http-server", url="http://localhost:8000/mcp")


@pytest.fixture
def mock_mcp_session():
    """Create mock MCP session."""
    session = AsyncMock()

    # Mock list_tools response
    tool1 = MagicMock()
    tool1.name = "test_tool_1"
    tool1.description = "Test tool 1"
    tool1.inputSchema = {
        "type": "object",
        "properties": {"arg1": {"type": "string", "description": "First argument"}},
        "required": ["arg1"],
    }

    tool2 = MagicMock()
    tool2.name = "test_tool_2"
    tool2.description = "Test tool 2"
    tool2.inputSchema = {
        "type": "object",
        "properties": {"arg2": {"type": "integer", "description": "Second argument"}},
        "required": [],
    }

    response = MagicMock()
    response.tools = [tool1, tool2]
    session.list_tools = AsyncMock(return_value=response)

    # Mock call_tool response
    def mock_call_tool(tool_name, arguments):
        result = MagicMock()
        content_item = MagicMock()
        content_item.type = "text"
        content_item.text = f"Result from {tool_name}"
        result.content = [content_item]
        return AsyncMock(return_value=result)()

    session.call_tool = mock_call_tool

    return session


@pytest.mark.asyncio
async def test_mcp_client_creation(stdio_server_config):
    """Test creating an MCP client."""
    client = MCPClient(stdio_server_config)

    assert client.config == stdio_server_config
    assert client.session is None
    assert client.mcp_tools == []


@pytest.mark.asyncio
async def test_mcp_client_connect_stdio(stdio_server_config, mock_mcp_session):
    """Test connecting to stdio MCP server."""
    with (
        patch("tsugite.mcp_client.stdio_client") as mock_stdio_client,
        patch("tsugite.mcp_client.ClientSession") as mock_session_class,
    ):
        # Mock stdio transport
        mock_transport = AsyncMock()
        mock_transport.__aenter__ = AsyncMock(return_value=(MagicMock(), MagicMock()))
        mock_stdio_client.return_value = mock_transport

        # Mock session
        mock_session_class.return_value.__aenter__ = AsyncMock(return_value=mock_mcp_session)

        client = MCPClient(stdio_server_config)
        await client.connect()

        # Verify stdio_client was called with correct params
        mock_stdio_client.assert_called_once()
        call_args = mock_stdio_client.call_args[0][0]
        assert call_args.command == "test-command"
        assert call_args.args == ["--arg1", "value1"]

        # Verify session was initialized
        mock_mcp_session.initialize.assert_called_once()

        # Verify tools were loaded
        assert len(client.mcp_tools) == 2
        assert client.mcp_tools[0].name == "test_tool_1"


@pytest.mark.asyncio
async def test_mcp_client_connect_http(http_server_config, mock_mcp_session):
    """Test connecting to HTTP MCP server."""
    with (
        patch("tsugite.mcp_client.streamablehttp_client") as mock_http_client,
        patch("tsugite.mcp_client.ClientSession") as mock_session_class,
    ):
        # Mock HTTP transport
        mock_transport = AsyncMock()
        mock_transport.__aenter__ = AsyncMock(return_value=(MagicMock(), MagicMock(), MagicMock()))
        mock_http_client.return_value = mock_transport

        # Mock session
        mock_session_class.return_value.__aenter__ = AsyncMock(return_value=mock_mcp_session)

        client = MCPClient(http_server_config)
        await client.connect()

        # Verify streamablehttp_client was called with correct URL
        mock_http_client.assert_called_once_with("http://localhost:8000/mcp")

        # Verify session was initialized
        mock_mcp_session.initialize.assert_called_once()


@pytest.mark.asyncio
async def test_mcp_client_get_tools(stdio_server_config, mock_mcp_session):
    """Test getting tools from MCP client."""
    with (
        patch("tsugite.mcp_client.stdio_client") as mock_stdio_client,
        patch("tsugite.mcp_client.ClientSession") as mock_session_class,
    ):
        # Setup mocks
        mock_transport = AsyncMock()
        mock_transport.__aenter__ = AsyncMock(return_value=(MagicMock(), MagicMock()))
        mock_stdio_client.return_value = mock_transport
        mock_session_class.return_value.__aenter__ = AsyncMock(return_value=mock_mcp_session)

        client = MCPClient(stdio_server_config)
        await client.connect()

        tools = await client.get_tools()

        # Should return Tool objects
        assert len(tools) == 2
        assert all(isinstance(tool, Tool) for tool in tools)

        # Check first tool
        assert tools[0].name == "test_tool_1"
        assert tools[0].description == "Test tool 1"
        assert "arg1" in tools[0].parameters["properties"]


@pytest.mark.asyncio
async def test_mcp_client_get_tools_filtered(stdio_server_config, mock_mcp_session):
    """Test getting filtered tools from MCP client."""
    with (
        patch("tsugite.mcp_client.stdio_client") as mock_stdio_client,
        patch("tsugite.mcp_client.ClientSession") as mock_session_class,
    ):
        # Setup mocks
        mock_transport = AsyncMock()
        mock_transport.__aenter__ = AsyncMock(return_value=(MagicMock(), MagicMock()))
        mock_stdio_client.return_value = mock_transport
        mock_session_class.return_value.__aenter__ = AsyncMock(return_value=mock_mcp_session)

        client = MCPClient(stdio_server_config)
        await client.connect()

        # Only get test_tool_1
        tools = await client.get_tools(allowed_tools=["test_tool_1"])

        assert len(tools) == 1
        assert tools[0].name == "test_tool_1"


@pytest.mark.asyncio
async def test_mcp_client_tool_execution(stdio_server_config, mock_mcp_session):
    """Test executing tools via MCP client."""
    with (
        patch("tsugite.mcp_client.stdio_client") as mock_stdio_client,
        patch("tsugite.mcp_client.ClientSession") as mock_session_class,
    ):
        # Setup mocks
        mock_transport = AsyncMock()
        mock_transport.__aenter__ = AsyncMock(return_value=(MagicMock(), MagicMock()))
        mock_stdio_client.return_value = mock_transport
        mock_session_class.return_value.__aenter__ = AsyncMock(return_value=mock_mcp_session)

        client = MCPClient(stdio_server_config)
        await client.connect()

        tools = await client.get_tools()

        # Execute first tool
        result = await tools[0].execute(arg1="test_value")

        # Should get result from MCP server
        assert "Result from test_tool_1" in result


@pytest.mark.asyncio
async def test_load_mcp_tools_convenience(stdio_server_config, mock_mcp_session):
    """Test load_mcp_tools convenience function."""
    with (
        patch("tsugite.mcp_client.stdio_client") as mock_stdio_client,
        patch("tsugite.mcp_client.ClientSession") as mock_session_class,
    ):
        # Setup mocks
        mock_transport = AsyncMock()
        mock_transport.__aenter__ = AsyncMock(return_value=(MagicMock(), MagicMock()))
        mock_stdio_client.return_value = mock_transport
        mock_session_class.return_value.__aenter__ = AsyncMock(return_value=mock_mcp_session)

        # Load tools using convenience function
        client, tools = await load_mcp_tools(stdio_server_config)

        # Should return client and Tool objects
        assert isinstance(client, MCPClient)
        assert len(tools) == 2
        assert all(isinstance(tool, Tool) for tool in tools)

        # Cleanup
        await client.disconnect()


@pytest.mark.asyncio
async def test_load_mcp_tools_filtered(stdio_server_config, mock_mcp_session):
    """Test load_mcp_tools with filtering."""
    with (
        patch("tsugite.mcp_client.stdio_client") as mock_stdio_client,
        patch("tsugite.mcp_client.ClientSession") as mock_session_class,
    ):
        # Setup mocks
        mock_transport = AsyncMock()
        mock_transport.__aenter__ = AsyncMock(return_value=(MagicMock(), MagicMock()))
        mock_stdio_client.return_value = mock_transport
        mock_session_class.return_value.__aenter__ = AsyncMock(return_value=mock_mcp_session)

        # Load only specific tool
        client, tools = await load_mcp_tools(stdio_server_config, allowed_tools=["test_tool_2"])

        assert len(tools) == 1
        assert tools[0].name == "test_tool_2"

        # Cleanup
        await client.disconnect()


@pytest.mark.asyncio
async def test_mcp_client_invalid_transport():
    """Test that invalid transport type raises error."""
    # Create config with invalid type
    config = MCPServerConfig(name="invalid", command="test")
    config.type = "invalid_type"

    client = MCPClient(config)

    with pytest.raises(ValueError, match="Unknown transport type"):
        await client.connect()


@pytest.mark.asyncio
async def test_mcp_client_disconnect(stdio_server_config, mock_mcp_session):
    """Test that disconnect() properly cleans up session and transport."""
    with (
        patch("tsugite.mcp_client.stdio_client") as mock_stdio_client,
        patch("tsugite.mcp_client.ClientSession") as mock_session_class,
    ):
        # Setup mocks
        mock_transport = AsyncMock()
        mock_transport.__aenter__ = AsyncMock(return_value=(MagicMock(), MagicMock()))
        mock_transport.__aexit__ = AsyncMock(return_value=None)
        mock_stdio_client.return_value = mock_transport

        mock_session_ctx = AsyncMock()
        mock_session_ctx.__aenter__ = AsyncMock(return_value=mock_mcp_session)
        mock_session_ctx.__aexit__ = AsyncMock(return_value=None)
        mock_session_class.return_value = mock_session_ctx

        client = MCPClient(stdio_server_config)
        await client.connect()

        # Verify connected
        assert client.session is not None
        assert client.transport is not None

        # Disconnect
        await client.disconnect()

        # Verify __aexit__ was called on both context managers
        mock_session_ctx.__aexit__.assert_called_once_with(None, None, None)
        mock_transport.__aexit__.assert_called_once_with(None, None, None)

        # Verify client state is cleaned up
        assert client.session is None
        assert client.session_ctx is None
        assert client.transport is None


@pytest.mark.asyncio
async def test_mcp_client_disconnect_handles_errors(stdio_server_config, mock_mcp_session):
    """Test that disconnect() handles errors gracefully during cleanup."""
    with (
        patch("tsugite.mcp_client.stdio_client") as mock_stdio_client,
        patch("tsugite.mcp_client.ClientSession") as mock_session_class,
    ):
        # Setup mocks
        mock_transport = AsyncMock()
        mock_transport.__aenter__ = AsyncMock(return_value=(MagicMock(), MagicMock()))
        # Make __aexit__ raise an error
        mock_transport.__aexit__ = AsyncMock(side_effect=RuntimeError("Transport cleanup failed"))
        mock_stdio_client.return_value = mock_transport

        mock_session_ctx = AsyncMock()
        mock_session_ctx.__aenter__ = AsyncMock(return_value=mock_mcp_session)
        # Make __aexit__ raise an error
        mock_session_ctx.__aexit__ = AsyncMock(side_effect=RuntimeError("Session cleanup failed"))
        mock_session_class.return_value = mock_session_ctx

        client = MCPClient(stdio_server_config)
        await client.connect()

        # Disconnect should not raise errors, even if cleanup fails
        await client.disconnect()

        # Verify both __aexit__ methods were called despite errors
        mock_session_ctx.__aexit__.assert_called_once()
        mock_transport.__aexit__.assert_called_once()

        # Verify client state is cleaned up
        assert client.session is None
        assert client.session_ctx is None
        assert client.transport is None


@pytest.mark.asyncio
async def test_tools_work_before_disconnect(stdio_server_config, mock_mcp_session):
    """Test that tools work correctly before client is disconnected."""
    with (
        patch("tsugite.mcp_client.stdio_client") as mock_stdio_client,
        patch("tsugite.mcp_client.ClientSession") as mock_session_class,
    ):
        # Setup mocks
        mock_transport = AsyncMock()
        mock_transport.__aenter__ = AsyncMock(return_value=(MagicMock(), MagicMock()))
        mock_transport.__aexit__ = AsyncMock(return_value=None)
        mock_stdio_client.return_value = mock_transport

        mock_session_ctx = AsyncMock()
        mock_session_ctx.__aenter__ = AsyncMock(return_value=mock_mcp_session)
        mock_session_ctx.__aexit__ = AsyncMock(return_value=None)
        mock_session_class.return_value = mock_session_ctx

        client, tools = await load_mcp_tools(stdio_server_config)

        # Tools should work before disconnect
        result = await tools[0].execute(arg1="test")
        assert "Result from test_tool_1" in result

        # Cleanup
        await client.disconnect()

        # Verify cleanup happened
        mock_session_ctx.__aexit__.assert_called_once()
        mock_transport.__aexit__.assert_called_once()
