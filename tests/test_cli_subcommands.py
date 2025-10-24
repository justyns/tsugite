"""Basic smoke tests for CLI subcommands."""

import tempfile
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from tsugite.cli import app

runner = CliRunner()


class TestMcpSubcommands:
    """Test MCP CLI subcommands."""

    def test_mcp_help(self):
        """Test mcp --help works."""
        result = runner.invoke(app, ["mcp", "--help"])
        assert result.exit_code == 0
        assert "mcp" in result.stdout.lower()

    def test_mcp_list_empty(self):
        """Test mcp list with no servers."""
        with patch("tsugite.mcp_config.load_mcp_config") as mock_load:
            mock_load.return_value = {}
            result = runner.invoke(app, ["mcp", "list"])
            assert result.exit_code == 0
            assert "No MCP servers configured" in result.stdout

    def test_mcp_list_with_servers(self):
        """Test mcp list with servers."""
        from tsugite.mcp_config import MCPServerConfig

        with patch("tsugite.mcp_config.load_mcp_config") as mock_load:
            mock_load.return_value = {
                "test-server": MCPServerConfig(name="test-server", url="http://localhost:8000/mcp")
            }
            result = runner.invoke(app, ["mcp", "list"])
            assert result.exit_code == 0
            assert "test-server" in result.stdout


class TestAgentsSubcommands:
    """Test agents CLI subcommands."""

    def test_agents_help(self):
        """Test agents --help works."""
        result = runner.invoke(app, ["agents", "--help"])
        assert result.exit_code == 0
        assert "agents" in result.stdout.lower()

    def test_agents_list_empty(self):
        """Test agents list with no agents."""
        with (
            patch("tsugite.agent_inheritance.get_global_agents_paths") as mock_global,
            patch("tsugite.agent_utils.list_local_agents") as mock_local,
        ):
            mock_global.return_value = []
            mock_local.return_value = {}
            result = runner.invoke(app, ["agents", "list"])
            assert result.exit_code == 0
            assert "agents" in result.stdout.lower()

    def test_agents_show_builtin(self):
        """Test agents show for builtin agent."""
        with (
            patch("tsugite.builtin_agents.is_builtin_agent") as mock_builtin,
            patch("tsugite.builtin_agents.get_builtin_default_agent") as mock_get,
        ):
            mock_builtin.return_value = True
            mock_agent = MagicMock()
            mock_agent.config.name = "builtin-default"
            mock_agent.config.description = None
            mock_agent.config.model = "ollama:qwen2.5-coder:7b"
            mock_agent.config.max_turns = 5
            mock_agent.config.tools = []
            mock_agent.config.extends = None
            mock_get.return_value = mock_agent

            result = runner.invoke(app, ["agents", "show", "builtin-default"])
            assert result.exit_code == 0
            assert "builtin-default" in result.stdout


class TestConfigSubcommands:
    """Test config CLI subcommands."""

    def test_config_help(self):
        """Test config --help works."""
        result = runner.invoke(app, ["config", "--help"])
        assert result.exit_code == 0
        assert "config" in result.stdout.lower()

    def test_config_show(self):
        """Test config show."""
        with patch("tsugite.config.load_config") as mock_load:
            mock_config = MagicMock()
            mock_config.default_model = "ollama:qwen2.5-coder:7b"
            mock_config.default_base_agent = None
            mock_config.model_aliases = {}
            mock_load.return_value = mock_config

            result = runner.invoke(app, ["config", "show"])
            assert result.exit_code == 0
            assert "Configuration" in result.stdout

    def test_config_list_aliases_empty(self):
        """Test config list-aliases with no aliases."""
        with patch("tsugite.config.load_config") as mock_load:
            mock_config = MagicMock()
            mock_config.model_aliases = {}
            mock_load.return_value = mock_config

            result = runner.invoke(app, ["config", "list-aliases"])
            assert result.exit_code == 0
            assert "No model aliases" in result.stdout


class TestAttachmentsSubcommands:
    """Test attachments CLI subcommands."""

    def test_attachments_help(self):
        """Test attachments --help works."""
        result = runner.invoke(app, ["attachments", "--help"])
        assert result.exit_code == 0
        assert "attachment" in result.stdout.lower()

    def test_attachments_list_empty(self):
        """Test attachments list with no attachments."""
        with patch("tsugite.attachments.list_attachments") as mock_list:
            mock_list.return_value = {}
            result = runner.invoke(app, ["attachments", "list"])
            assert result.exit_code == 0
            assert "No attachments found" in result.stdout

    def test_attachments_list_with_items(self):
        """Test attachments list with attachments."""
        with patch("tsugite.attachments.list_attachments") as mock_list:
            mock_list.return_value = {
                "test-attachment": {
                    "source": "inline",
                    "content": "test content",
                    "updated": "2024-01-01T00:00:00",
                }
            }
            result = runner.invoke(app, ["attachments", "list"])
            assert result.exit_code == 0
            assert "test-attachment" in result.stdout

    def test_attachments_add_file_not_found(self):
        """Test attachments add with missing file."""
        result = runner.invoke(app, ["attachments", "add", "test", "/nonexistent/file.txt"])
        assert result.exit_code == 1
        assert "not found" in result.stdout.lower()


class TestCacheSubcommands:
    """Test cache CLI subcommands."""

    def test_cache_help(self):
        """Test cache --help works."""
        result = runner.invoke(app, ["cache", "--help"])
        assert result.exit_code == 0
        assert "cache" in result.stdout.lower()

    def test_cache_list_empty(self):
        """Test cache list with no entries."""
        with patch("tsugite.cache.list_cache") as mock_list:
            mock_list.return_value = {}
            result = runner.invoke(app, ["cache", "list"])
            assert result.exit_code == 0
            assert "No cached entries" in result.stdout

    def test_cache_clear_all(self):
        """Test cache clear all."""
        with patch("tsugite.cache.clear_cache") as mock_clear:
            mock_clear.return_value = 5
            result = runner.invoke(app, ["cache", "clear"])
            assert result.exit_code == 0
            assert "5" in result.stdout
            mock_clear.assert_called_once_with()


class TestBenchmarkCommand:
    """Test benchmark CLI command."""

    def test_benchmark_help(self):
        """Test benchmark --help works."""
        result = runner.invoke(app, ["benchmark", "--help"])
        assert result.exit_code == 0
        assert "benchmark" in result.stdout.lower()

    def test_benchmark_run_no_models(self):
        """Test benchmark run without --models flag."""
        result = runner.invoke(app, ["benchmark", "run"])
        assert result.exit_code == 1
        assert "--models is required" in result.stdout


class TestChatCommand:
    """Test chat CLI command."""

    def test_chat_help(self):
        """Test chat --help works."""
        result = runner.invoke(app, ["chat", "--help"])
        assert result.exit_code == 0
        assert "chat" in result.stdout.lower()

    def test_chat_no_default_agent(self):
        """Test chat without default agent."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("tsugite.ui.textual_chat.run_textual_chat") as mock_run:
                result = runner.invoke(app, ["chat", "--root", tmpdir])
                # The chat command should succeed but run_textual_chat should be called
                # Built-in chat assistant exists by default
                assert result.exit_code == 0
                mock_run.assert_called_once()
