"""Tests for daemon configuration."""

import os
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from tsugite.daemon.config import AgentConfig, DaemonConfig, DiscordBotConfig, load_daemon_config


def test_agent_config():
    """Test AgentConfig model."""
    config = AgentConfig(
        workspace_dir=Path("/tmp/workspace"),
        agent_file="default",
        context_limit=128000,
    )
    assert config.workspace_dir == Path("/tmp/workspace")
    assert config.agent_file == "default"
    assert config.context_limit == 128000


def test_discord_bot_config():
    """Test DiscordBotConfig model with token_secret."""
    config = DiscordBotConfig(
        name="test-bot",
        token_secret="test-secret",
        agent="test-agent",
        command_prefix="!",
        dm_policy="allowlist",
        allow_from=["123"],
    )
    assert config.name == "test-bot"
    assert config.token_secret == "test-secret"
    assert config.token_file is None
    assert config.agent == "test-agent"
    assert config.command_prefix == "!"
    assert config.dm_policy == "allowlist"
    assert config.allow_from == ["123"]


def test_discord_bot_config_requires_token_source():
    """Test that DiscordBotConfig requires either token_secret or token_file."""
    with pytest.raises(ValidationError, match="must set exactly one of token_secret, token_file"):
        DiscordBotConfig(name="bot", agent="agent")


def test_discord_bot_config_rejects_both_token_sources():
    """Test that DiscordBotConfig rejects setting both token_secret and token_file."""
    with pytest.raises(ValidationError, match="must set exactly one of token_secret, token_file"):
        DiscordBotConfig(
            name="bot",
            agent="agent",
            token_secret="x",
            token_file=Path("/tmp/x.token"),
        )


def test_daemon_config():
    """Test DaemonConfig model."""
    config = DaemonConfig(
        state_dir=Path("/tmp/daemon"),
        log_level="info",
        agents={
            "test": AgentConfig(
                workspace_dir=Path("/tmp/workspace"),
                agent_file="default",
            )
        },
        discord_bots=[DiscordBotConfig(name="bot", token_secret="bot-secret", agent="test")],
    )
    assert config.state_dir == Path("/tmp/daemon")
    assert config.log_level == "info"
    assert "test" in config.agents
    assert len(config.discord_bots) == 1


def test_load_daemon_config(tmp_path):
    """Test loading daemon config from YAML."""
    config_file = tmp_path / "daemon.yaml"

    config_data = {
        "state_dir": str(tmp_path / "daemon"),
        "log_level": "debug",
        "agents": {
            "test": {
                "workspace_dir": str(tmp_path / "workspace"),
                "agent_file": "default",
                "context_limit": 100000,
            }
        },
        "discord_bots": [
            {"name": "test-bot", "token_secret": "test-bot-secret", "agent": "test", "command_prefix": "!"}
        ],
    }

    with open(config_file, "w") as f:
        yaml.dump(config_data, f)

    config = load_daemon_config(config_file)

    assert config.log_level == "debug"
    assert "test" in config.agents
    assert config.agents["test"].workspace_dir == tmp_path / "workspace"
    assert config.agents["test"].agent_file == "default"
    assert len(config.discord_bots) == 1
    assert config.discord_bots[0].name == "test-bot"
    assert config.discord_bots[0].token_secret == "test-bot-secret"


def test_load_daemon_config_rejects_legacy_plain_token(tmp_path):
    """Test that legacy plaintext token: field is rejected with a migration hint."""
    config_file = tmp_path / "daemon.yaml"

    config_data = {
        "agents": {"test": {"workspace_dir": str(tmp_path / "workspace"), "agent_file": "default"}},
        "discord_bots": [{"name": "legacy-bot", "token": "PLAINTEXT-DISCORD-TOKEN", "agent": "test"}],
    }

    with open(config_file, "w") as f:
        yaml.dump(config_data, f)

    with pytest.raises(ValueError, match="plaintext 'token:' is no longer supported"):
        load_daemon_config(config_file)


def test_load_daemon_config_token_file_path_expanded(tmp_path, monkeypatch):
    """Test that ~ in token_file path is expanded."""
    config_file = tmp_path / "daemon.yaml"
    monkeypatch.setenv("HOME", str(tmp_path))

    config_data = {
        "agents": {"test": {"workspace_dir": str(tmp_path / "workspace"), "agent_file": "default"}},
        "discord_bots": [{"name": "test-bot", "token_file": "~/discord.token", "agent": "test"}],
    }

    with open(config_file, "w") as f:
        yaml.dump(config_data, f)

    config = load_daemon_config(config_file)

    assert config.discord_bots[0].token_file == tmp_path / "discord.token"


def test_load_daemon_config_token_file_env_var_expanded(tmp_path, monkeypatch):
    """Test that ${VAR} in token_file path is expanded."""
    config_file = tmp_path / "daemon.yaml"
    monkeypatch.setenv("TSUGITE_TEST_TOKEN_DIR", str(tmp_path / "secrets-dir"))

    config_data = {
        "agents": {"test": {"workspace_dir": str(tmp_path / "workspace"), "agent_file": "default"}},
        "discord_bots": [
            {"name": "test-bot", "token_file": "${TSUGITE_TEST_TOKEN_DIR}/discord.token", "agent": "test"}
        ],
    }

    with open(config_file, "w") as f:
        yaml.dump(config_data, f)

    config = load_daemon_config(config_file)

    assert config.discord_bots[0].token_file == tmp_path / "secrets-dir" / "discord.token"


def test_resolve_token_from_secret(secret_backend):
    """Test that resolve_token() returns the secret value from the backend."""
    secret_backend.set("my-discord-secret", "actual-token-value")

    config = DiscordBotConfig(name="bot", token_secret="my-discord-secret", agent="test")
    assert config.resolve_token() == "actual-token-value"


def test_resolve_token_secret_missing_raises(secret_backend):
    """Test that a missing secret raises RuntimeError."""
    config = DiscordBotConfig(name="bot", token_secret="does-not-exist", agent="test")
    with pytest.raises(RuntimeError, match="secret 'does-not-exist' not found"):
        config.resolve_token()


def test_resolve_token_from_file(tmp_path):
    """Test that resolve_token() reads the token from a file."""
    token_file = tmp_path / "discord.token"
    token_file.write_text("file-token-value\n", encoding="utf-8")

    config = DiscordBotConfig(name="bot", token_file=token_file, agent="test")
    assert config.resolve_token() == "file-token-value"


def test_load_daemon_config_not_found():
    """Test loading non-existent config file."""
    with pytest.raises(ValueError, match="Daemon config not found"):
        load_daemon_config(Path("/nonexistent/daemon.yaml"))


def test_agent_config_defaults():
    """Test AgentConfig default values."""
    config = AgentConfig(workspace_dir=Path("/tmp/workspace"), agent_file="default")
    assert config.context_limit is None  # auto-detected at startup


def test_discord_bot_config_defaults():
    """Test DiscordBotConfig default values."""
    config = DiscordBotConfig(name="bot", token_secret="x", agent="agent")
    assert config.command_prefix == "!"  # default
    assert config.dm_policy == "allowlist"  # default
    assert config.allow_from == []  # default
