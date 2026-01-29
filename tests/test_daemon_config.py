"""Tests for daemon configuration."""

import os
from pathlib import Path

import pytest
import yaml

from tsugite.daemon.config import AgentConfig, DaemonConfig, DiscordBotConfig, load_daemon_config


def test_agent_config():
    """Test AgentConfig model."""
    config = AgentConfig(
        workspace_dir=Path("/tmp/workspace"),
        agent_file="assistant.md",
        memory_enabled=True,
        memory_inject_days=2,
        context_limit=128000,
        memory_extraction_interval=10,
    )
    assert config.workspace_dir == Path("/tmp/workspace")
    assert config.agent_file == "assistant.md"
    assert config.memory_enabled is True
    assert config.memory_inject_days == 2
    assert config.context_limit == 128000
    assert config.memory_extraction_interval == 10


def test_discord_bot_config():
    """Test DiscordBotConfig model."""
    config = DiscordBotConfig(
        name="test-bot",
        token="test-token",
        agent="test-agent",
        command_prefix="!",
        dm_policy="allowlist",
        allow_from=["123"],
    )
    assert config.name == "test-bot"
    assert config.token == "test-token"
    assert config.agent == "test-agent"
    assert config.command_prefix == "!"
    assert config.dm_policy == "allowlist"
    assert config.allow_from == ["123"]


def test_daemon_config():
    """Test DaemonConfig model."""
    config = DaemonConfig(
        state_dir=Path("/tmp/daemon"),
        log_level="info",
        agents={
            "test": AgentConfig(
                workspace_dir=Path("/tmp/workspace"),
                agent_file="assistant.md",
                memory_enabled=True,
                memory_inject_days=2,
            )
        },
        discord_bots=[DiscordBotConfig(name="bot", token="token", agent="test")],
    )
    assert config.state_dir == Path("/tmp/daemon")
    assert config.log_level == "info"
    assert "test" in config.agents
    assert len(config.discord_bots) == 1


def test_load_daemon_config(tmp_path):
    """Test loading daemon config from YAML."""
    config_file = tmp_path / "daemon.yaml"

    # Create test config
    config_data = {
        "state_dir": str(tmp_path / "daemon"),
        "log_level": "debug",
        "agents": {
            "test": {
                "workspace_dir": str(tmp_path / "workspace"),
                "agent_file": "assistant.md",
                "memory_enabled": True,
                "memory_inject_days": 2,
                "context_limit": 100000,
                "memory_extraction_interval": 5,
            }
        },
        "discord_bots": [{"name": "test-bot", "token": "test-token", "agent": "test", "command_prefix": "!"}],
    }

    with open(config_file, "w") as f:
        yaml.dump(config_data, f)

    # Load config
    config = load_daemon_config(config_file)

    assert config.log_level == "debug"
    assert "test" in config.agents
    assert config.agents["test"].workspace_dir == tmp_path / "workspace"
    assert config.agents["test"].agent_file == "assistant.md"
    assert len(config.discord_bots) == 1
    assert config.discord_bots[0].name == "test-bot"


def test_load_daemon_config_env_expansion(tmp_path):
    """Test environment variable expansion in config."""
    config_file = tmp_path / "daemon.yaml"

    # Set test env var
    os.environ["TEST_TOKEN"] = "secret-token"

    config_data = {
        "agents": {"test": {"workspace_dir": str(tmp_path / "workspace"), "agent_file": "assistant.md"}},
        "discord_bots": [{"name": "test-bot", "token": "${TEST_TOKEN}", "agent": "test"}],
    }

    with open(config_file, "w") as f:
        yaml.dump(config_data, f)

    config = load_daemon_config(config_file)

    assert config.discord_bots[0].token == "secret-token"

    # Cleanup
    del os.environ["TEST_TOKEN"]


def test_load_daemon_config_not_found():
    """Test loading non-existent config file."""
    with pytest.raises(ValueError, match="Daemon config not found"):
        load_daemon_config(Path("/nonexistent/daemon.yaml"))


def test_agent_config_defaults():
    """Test AgentConfig default values."""
    config = AgentConfig(workspace_dir=Path("/tmp/workspace"), agent_file="assistant.md")
    assert config.memory_enabled is True  # default
    assert config.memory_inject_days == 2  # default
    assert config.context_limit == 128000  # default
    assert config.memory_extraction_interval == 10  # default


def test_discord_bot_config_defaults():
    """Test DiscordBotConfig default values."""
    config = DiscordBotConfig(name="bot", token="token", agent="agent")
    assert config.command_prefix == "!"  # default
    assert config.dm_policy == "allowlist"  # default
    assert config.allow_from == []  # default
