"""Tests for MCP integration."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tsugite.mcp_config import MCPServerConfig, load_mcp_config
from tsugite.mcp_integration import convert_to_server_params, load_all_mcp_tools, load_mcp_tools


class TestMCPServerConfig:
    """Tests for MCPServerConfig dataclass."""

    def test_stdio_server_config(self):
        """Test creating a stdio server configuration."""
        config = MCPServerConfig(
            name="test-server",
            command="npx",
            args=["-y", "test-mcp-server"],
            env={"API_KEY": "test-key"},
        )

        assert config.name == "test-server"
        assert config.command == "npx"
        assert config.args == ["-y", "test-mcp-server"]
        assert config.env == {"API_KEY": "test-key"}
        assert config.type == "stdio"
        assert config.is_stdio()
        assert not config.is_http()

    def test_http_server_config(self):
        """Test creating an HTTP server configuration."""
        config = MCPServerConfig(name="test-server", url="http://localhost:8000/mcp")

        assert config.name == "test-server"
        assert config.url == "http://localhost:8000/mcp"
        assert config.type == "http"
        assert config.is_http()
        assert not config.is_stdio()

    def test_server_config_with_explicit_type(self):
        """Test creating a server with explicit type."""
        config = MCPServerConfig(name="test-server", command="npx", args=["test"], type="stdio")

        assert config.type == "stdio"

    def test_invalid_server_config_no_command_or_url(self):
        """Test that server config requires command or url."""
        with pytest.raises(ValueError, match="must have either 'command' or 'url'"):
            MCPServerConfig(name="test-server")

    def test_invalid_stdio_server_without_command(self):
        """Test that stdio server requires command."""
        with pytest.raises(ValueError, match="Stdio server.*must have 'command'"):
            MCPServerConfig(name="test-server", type="stdio", url="http://test.com")

    def test_invalid_http_server_without_url(self):
        """Test that HTTP server requires url."""
        with pytest.raises(ValueError, match="HTTP server.*must have 'url'"):
            MCPServerConfig(name="test-server", type="http", command="npx")


class TestLoadMCPConfig:
    """Tests for loading MCP configuration from file."""

    def test_load_valid_config(self, tmp_path):
        """Test loading a valid MCP configuration."""
        config_data = {
            "mcpServers": {
                "test-stdio": {"command": "npx", "args": ["-y", "test-server"], "env": {"API_KEY": "test"}},
                "test-http": {"url": "http://localhost:8000/mcp"},
            }
        }

        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps(config_data))

        servers = load_mcp_config(config_path)

        assert len(servers) == 2
        assert "test-stdio" in servers
        assert "test-http" in servers

        assert servers["test-stdio"].command == "npx"
        assert servers["test-stdio"].is_stdio()

        assert servers["test-http"].url == "http://localhost:8000/mcp"
        assert servers["test-http"].is_http()

    def test_load_missing_config(self):
        """Test loading config from non-existent file."""
        servers = load_mcp_config(Path("/nonexistent/config.json"))
        assert servers == {}

    def test_load_invalid_json(self, tmp_path):
        """Test loading invalid JSON file."""
        config_path = tmp_path / "invalid.json"
        config_path.write_text("{ invalid json")

        servers = load_mcp_config(config_path)
        assert servers == {}

    def test_load_config_missing_mcpServers_key(self, tmp_path):
        """Test loading config without mcpServers key."""
        config_data = {"someOtherKey": {}}
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps(config_data))

        servers = load_mcp_config(config_path)
        assert servers == {}

    def test_load_config_with_invalid_server(self, tmp_path):
        """Test loading config with one invalid server."""
        config_data = {
            "mcpServers": {
                "valid-server": {"command": "npx", "args": ["test"]},
                "invalid-server": {},  # Missing command/url
            }
        }
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps(config_data))

        servers = load_mcp_config(config_path)
        assert len(servers) == 1
        assert "valid-server" in servers
        assert "invalid-server" not in servers


class TestConvertToServerParams:
    """Tests for converting MCPServerConfig to server parameters."""

    def test_convert_stdio_server(self):
        """Test converting stdio server config."""
        config = MCPServerConfig(name="test", command="npx", args=["-y", "test"], env={"KEY": "value"})

        params = convert_to_server_params(config)

        assert hasattr(params, "command")
        assert params.command == "npx"
        assert params.args == ["-y", "test"]
        assert "KEY" in params.env
        assert params.env["KEY"] == "value"

    def test_convert_http_server(self):
        """Test converting HTTP server config."""
        config = MCPServerConfig(name="test", url="http://localhost:8000/mcp")

        params = convert_to_server_params(config)

        assert isinstance(params, dict)
        assert params["url"] == "http://localhost:8000/mcp"
        assert params["transport"] == "streamable-http"

    def test_convert_unknown_server_type(self):
        """Test converting server with unknown type raises error."""
        config = MCPServerConfig(name="test", command="npx")
        config.type = "unknown"

        with pytest.raises(ValueError, match="Unknown server type"):
            convert_to_server_params(config)


class TestLoadMCPTools:
    """Tests for loading MCP tools from a server."""

    @patch("tsugite.mcp_integration.ToolCollection")
    def test_load_all_tools(self, mock_tool_collection):
        """Test loading all tools from a server."""
        # Setup mock tools
        mock_tool1 = MagicMock()
        mock_tool1.name = "tool1"
        mock_tool2 = MagicMock()
        mock_tool2.name = "tool2"

        mock_context = MagicMock()
        mock_context.__enter__ = MagicMock(return_value=MagicMock(tools=[mock_tool1, mock_tool2]))
        mock_context.__exit__ = MagicMock(return_value=False)

        mock_tool_collection.from_mcp = MagicMock(return_value=mock_context)

        config = MCPServerConfig(name="test", command="npx", args=["test"])

        tools = load_mcp_tools("test", config, allowed_tools=None, trust_remote_code=True)

        assert len(tools) == 2
        assert tools[0].name == "tool1"
        assert tools[1].name == "tool2"

    @patch("tsugite.mcp_integration.ToolCollection")
    def test_load_filtered_tools(self, mock_tool_collection):
        """Test loading specific tools from a server."""
        mock_tool1 = MagicMock()
        mock_tool1.name = "tool1"
        mock_tool2 = MagicMock()
        mock_tool2.name = "tool2"
        mock_tool3 = MagicMock()
        mock_tool3.name = "tool3"

        mock_context = MagicMock()
        mock_context.__enter__ = MagicMock(return_value=MagicMock(tools=[mock_tool1, mock_tool2, mock_tool3]))
        mock_context.__exit__ = MagicMock(return_value=False)

        mock_tool_collection.from_mcp = MagicMock(return_value=mock_context)

        config = MCPServerConfig(name="test", command="npx", args=["test"])

        tools = load_mcp_tools("test", config, allowed_tools=["tool1", "tool3"], trust_remote_code=True)

        assert len(tools) == 2
        assert tools[0].name == "tool1"
        assert tools[1].name == "tool3"

    @patch("tsugite.mcp_integration.ToolCollection")
    def test_load_tools_with_unknown_tool_name(self, mock_tool_collection, capsys):
        """Test loading tools with unknown tool name prints warning."""
        mock_tool1 = MagicMock()
        mock_tool1.name = "tool1"

        mock_context = MagicMock()
        mock_context.__enter__ = MagicMock(return_value=MagicMock(tools=[mock_tool1]))
        mock_context.__exit__ = MagicMock(return_value=False)

        mock_tool_collection.from_mcp = MagicMock(return_value=mock_context)

        config = MCPServerConfig(name="test", command="npx", args=["test"])

        tools = load_mcp_tools("test", config, allowed_tools=["tool1", "unknown_tool"], trust_remote_code=True)

        assert len(tools) == 1
        assert tools[0].name == "tool1"

        captured = capsys.readouterr()
        assert "unknown_tool" in captured.out
        assert "not found" in captured.out

    @patch("tsugite.mcp_integration.ToolCollection")
    def test_load_tools_connection_failure(self, mock_tool_collection):
        """Test that connection failures raise RuntimeError."""
        mock_tool_collection.from_mcp = MagicMock(side_effect=Exception("Connection failed"))

        config = MCPServerConfig(name="test", command="npx", args=["test"])

        with pytest.raises(RuntimeError, match="Failed to load tools.*Connection failed"):
            load_mcp_tools("test", config, allowed_tools=None, trust_remote_code=True)


class TestLoadAllMCPTools:
    """Tests for loading tools from multiple MCP servers."""

    @patch("tsugite.mcp_integration.load_mcp_tools")
    def test_load_from_multiple_servers(self, mock_load_mcp_tools):
        """Test loading tools from multiple servers."""
        mock_tool1 = MagicMock()
        mock_tool1.name = "tool1"
        mock_tool2 = MagicMock()
        mock_tool2.name = "tool2"

        mock_load_mcp_tools.side_effect = [[mock_tool1], [mock_tool2]]

        mcp_servers_config = {"server1": None, "server2": ["tool2"]}

        global_mcp_config = {
            "server1": MCPServerConfig(name="server1", command="npx", args=["server1"]),
            "server2": MCPServerConfig(name="server2", url="http://localhost:8000/mcp"),
        }

        tools = load_all_mcp_tools(mcp_servers_config, global_mcp_config, trust_remote_code=True)

        assert len(tools) == 2
        assert mock_load_mcp_tools.call_count == 2

    @patch("tsugite.mcp_integration.load_mcp_tools")
    def test_load_with_unknown_server(self, mock_load_mcp_tools, capsys):
        """Test loading with unknown server prints warning."""
        mcp_servers_config = {"unknown_server": None, "known_server": None}

        mock_tool = MagicMock()
        mock_tool.name = "tool1"
        mock_load_mcp_tools.return_value = [mock_tool]

        global_mcp_config = {"known_server": MCPServerConfig(name="known_server", command="npx", args=["test"])}

        tools = load_all_mcp_tools(mcp_servers_config, global_mcp_config, trust_remote_code=True)

        assert len(tools) == 1
        captured = capsys.readouterr()
        assert "unknown_server" in captured.out
        assert "not found" in captured.out

    @patch("tsugite.mcp_integration.load_mcp_tools")
    def test_load_with_server_failure(self, mock_load_mcp_tools, capsys):
        """Test that server failures are handled gracefully."""
        mock_tool = MagicMock()
        mock_tool.name = "tool1"

        mock_load_mcp_tools.side_effect = [RuntimeError("Connection failed"), [mock_tool]]

        mcp_servers_config = {"failing_server": None, "working_server": None}

        global_mcp_config = {
            "failing_server": MCPServerConfig(name="failing_server", command="npx", args=["fail"]),
            "working_server": MCPServerConfig(name="working_server", command="npx", args=["work"]),
        }

        tools = load_all_mcp_tools(mcp_servers_config, global_mcp_config, trust_remote_code=True)

        assert len(tools) == 1
        captured = capsys.readouterr()
        assert "Warning" in captured.out
        assert "Connection failed" in captured.out


class TestSaveMCPConfig:
    """Tests for saving MCP configuration to file."""

    def test_save_empty_config(self, tmp_path):
        """Test saving empty MCP configuration."""
        from tsugite.mcp_config import save_mcp_config

        config_path = tmp_path / "config.json"
        save_mcp_config({}, config_path)

        assert config_path.exists()
        data = json.loads(config_path.read_text())
        assert "mcpServers" in data
        assert data["mcpServers"] == {}

    def test_save_stdio_server(self, tmp_path):
        """Test saving stdio server configuration."""
        from tsugite.mcp_config import save_mcp_config

        config_path = tmp_path / "mcp.json"
        servers = {
            "test-server": MCPServerConfig(
                name="test-server", command="npx", args=["-y", "test"], env={"API_KEY": "test"}
            )
        }

        save_mcp_config(servers, config_path)

        data = json.loads(config_path.read_text())
        assert "mcpServers" in data
        assert "test-server" in data["mcpServers"]

        server_data = data["mcpServers"]["test-server"]
        assert server_data["command"] == "npx"
        assert server_data["args"] == ["-y", "test"]
        assert server_data["env"] == {"API_KEY": "test"}

    def test_save_http_server(self, tmp_path):
        """Test saving HTTP server configuration."""
        from tsugite.mcp_config import save_mcp_config

        config_path = tmp_path / "mcp.json"
        servers = {"test-server": MCPServerConfig(name="test-server", url="http://localhost:8000/mcp")}

        save_mcp_config(servers, config_path)

        data = json.loads(config_path.read_text())
        assert "mcpServers" in data
        assert "test-server" in data["mcpServers"]

        server_data = data["mcpServers"]["test-server"]
        assert server_data["type"] == "http"
        assert server_data["url"] == "http://localhost:8000/mcp"

    def test_save_creates_directory(self, tmp_path):
        """Test that save_mcp_config creates parent directory if needed."""
        from tsugite.mcp_config import save_mcp_config

        config_path = tmp_path / "subdir" / "mcp.json"
        assert not config_path.parent.exists()

        save_mcp_config({}, config_path)

        assert config_path.parent.exists()
        assert config_path.exists()


class TestAddServerToConfig:
    """Tests for adding servers to configuration."""

    def test_add_new_server(self, tmp_path):
        """Test adding a new server to empty config."""
        from tsugite.mcp_config import add_server_to_config

        config_path = tmp_path / "mcp.json"
        server = MCPServerConfig(name="new-server", url="http://localhost:8000/mcp")

        result = add_server_to_config(server, config_path)

        assert result is True
        assert config_path.exists()
        data = json.loads(config_path.read_text())
        assert "new-server" in data["mcpServers"]

    def test_add_server_to_existing_config(self, tmp_path):
        """Test adding a server to existing config."""
        from tsugite.mcp_config import add_server_to_config, save_mcp_config

        config_path = tmp_path / "mcp.json"

        # Create initial config
        initial_servers = {"server1": MCPServerConfig(name="server1", url="http://localhost:8000/mcp")}
        save_mcp_config(initial_servers, config_path)

        # Add second server
        server2 = MCPServerConfig(name="server2", command="npx", args=["test"])
        add_server_to_config(server2, config_path)

        # Verify both servers exist
        data = json.loads(config_path.read_text())
        assert len(data["mcpServers"]) == 2
        assert "server1" in data["mcpServers"]
        assert "server2" in data["mcpServers"]

    def test_add_duplicate_server_without_force(self, tmp_path):
        """Test adding duplicate server fails without force."""
        from tsugite.mcp_config import add_server_to_config

        config_path = tmp_path / "mcp.json"
        server1 = MCPServerConfig(name="test", url="http://localhost:8000/mcp")
        add_server_to_config(server1, config_path)

        # Try to add same server again
        server2 = MCPServerConfig(name="test", url="http://localhost:9000/mcp")

        with pytest.raises(ValueError, match="already exists"):
            add_server_to_config(server2, config_path, overwrite=False)

    def test_add_duplicate_server_with_force(self, tmp_path):
        """Test adding duplicate server with force overwrites."""
        from tsugite.mcp_config import add_server_to_config

        config_path = tmp_path / "mcp.json"
        server1 = MCPServerConfig(name="test", url="http://localhost:8000/mcp")
        add_server_to_config(server1, config_path)

        # Overwrite with different config
        server2 = MCPServerConfig(name="test", url="http://localhost:9000/mcp")
        result = add_server_to_config(server2, config_path, overwrite=True)

        assert result is True
        data = json.loads(config_path.read_text())
        assert data["mcpServers"]["test"]["url"] == "http://localhost:9000/mcp"


class TestXDGConfigPaths:
    """Tests for XDG Base Directory support."""

    def test_home_tsugite_path_takes_precedence(self, monkeypatch):
        """Test that ~/.tsugite/mcp.json takes precedence over XDG."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create both ~/.tsugite and XDG paths
            home_tsugite_dir = Path(tmpdir) / ".tsugite"
            home_tsugite_dir.mkdir()
            home_tsugite_path = home_tsugite_dir / "mcp.json"

            xdg_dir = Path(tmpdir) / ".config" / "tsugite"
            xdg_dir.mkdir(parents=True)
            xdg_path = xdg_dir / "mcp.json"

            # Write different content to each
            home_tsugite_path.write_text('{"mcpServers": {"home": {}}}')
            xdg_path.write_text('{"mcpServers": {"xdg": {}}}')

            # Mock home directory
            monkeypatch.setattr(Path, "home", lambda: Path(tmpdir))

            from tsugite.mcp_config import get_default_config_path

            result_path = get_default_config_path()

            # Should return ~/.tsugite path
            assert result_path == home_tsugite_path

    def test_xdg_config_home_used_when_set(self, monkeypatch):
        """Test that XDG_CONFIG_HOME is used when environment variable is set."""
        with tempfile.TemporaryDirectory() as tmpdir:
            xdg_config = Path(tmpdir) / "custom_config"
            xdg_config.mkdir()

            xdg_path = xdg_config / "tsugite" / "mcp.json"
            xdg_path.parent.mkdir()
            xdg_path.write_text('{"mcpServers": {}}')

            # Set XDG_CONFIG_HOME
            monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg_config))
            monkeypatch.setattr(Path, "home", lambda: Path(tmpdir))

            from tsugite.mcp_config import get_default_config_path

            result_path = get_default_config_path()

            # Should return XDG path
            assert result_path == xdg_path

    def test_default_xdg_path_when_no_env(self, monkeypatch):
        """Test that ~/.config/tsugite/mcp.json is used as default."""
        with tempfile.TemporaryDirectory() as tmpdir:
            default_path = Path(tmpdir) / ".config" / "tsugite" / "mcp.json"
            default_path.parent.mkdir(parents=True)
            default_path.write_text('{"mcpServers": {}}')

            # No XDG_CONFIG_HOME set
            monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
            monkeypatch.setattr(Path, "home", lambda: Path(tmpdir))

            from tsugite.mcp_config import get_default_config_path

            result_path = get_default_config_path()

            # Should return default XDG path
            assert result_path == default_path

    def test_new_install_uses_xdg(self, monkeypatch):
        """Test that new installations use XDG location."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # No config files exist
            monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
            monkeypatch.setattr(Path, "home", lambda: Path(tmpdir))

            from tsugite.mcp_config import get_default_config_path

            result_path = get_default_config_path()

            # Should return default XDG path (even though file doesn't exist)
            expected = Path(tmpdir) / ".config" / "tsugite" / "mcp.json"
            assert result_path == expected

    def test_write_path_respects_existing_home_tsugite(self, monkeypatch):
        """Test that write operations respect existing ~/.tsugite config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            home_tsugite_dir = Path(tmpdir) / ".tsugite"
            home_tsugite_dir.mkdir()
            home_tsugite_path = home_tsugite_dir / "mcp.json"
            home_tsugite_path.write_text('{"mcpServers": {}}')

            monkeypatch.setattr(Path, "home", lambda: Path(tmpdir))

            from tsugite.mcp_config import get_config_path_for_write

            result_path = get_config_path_for_write()

            # Should return ~/.tsugite path since it exists
            assert result_path == home_tsugite_path

    def test_write_path_uses_xdg_for_new_config(self, monkeypatch):
        """Test that new configs are written to XDG location."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # No existing config
            xdg_config = Path(tmpdir) / "custom_xdg"

            monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg_config))
            monkeypatch.setattr(Path, "home", lambda: Path(tmpdir))

            from tsugite.mcp_config import get_config_path_for_write

            result_path = get_config_path_for_write()

            # Should return XDG path
            expected = xdg_config / "tsugite" / "mcp.json"
            assert result_path == expected

    def test_write_path_uses_default_xdg_when_no_env(self, monkeypatch):
        """Test that default XDG path is used for new configs when XDG_CONFIG_HOME not set."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # No existing config, no XDG_CONFIG_HOME
            monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
            monkeypatch.setattr(Path, "home", lambda: Path(tmpdir))

            from tsugite.mcp_config import get_config_path_for_write

            result_path = get_config_path_for_write()

            # Should return default XDG path
            expected = Path(tmpdir) / ".config" / "tsugite" / "mcp.json"
            assert result_path == expected
