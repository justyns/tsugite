"""Tests for daemon configuration."""

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError
from tsugite_daemon.config import DaemonConfig, DiscordBotConfig, RuntimeDefaults, SandboxSettings, load_daemon_config


def _write_config(path: Path, data: dict) -> Path:
    config_file = path / "daemon.yaml"
    with open(config_file, "w") as f:
        yaml.dump(data, f)
    return config_file


def _defaults(tmp_path: Path, **extra) -> dict:
    base = {"default_workspace_dir": str(tmp_path / "workspace"), "default_agent_file": "default"}
    base.update(extra)
    return base


def test_runtime_defaults():
    """Test RuntimeDefaults model."""
    config = RuntimeDefaults(
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
        command_prefix="!",
        dm_policy="allowlist",
        allow_from=["123"],
    )
    assert config.name == "test-bot"
    assert config.token_secret == "test-secret"
    assert config.token_file is None
    assert config.command_prefix == "!"
    assert config.dm_policy == "allowlist"
    assert config.allow_from == ["123"]


def test_discord_bot_config_requires_token_source():
    """Test that DiscordBotConfig requires either token_secret or token_file."""
    with pytest.raises(ValidationError, match="must set exactly one of token_secret, token_file"):
        DiscordBotConfig(name="bot")


def test_discord_bot_config_rejects_both_token_sources():
    """Test that DiscordBotConfig rejects setting both token_secret and token_file."""
    with pytest.raises(ValidationError, match="must set exactly one of token_secret, token_file"):
        DiscordBotConfig(
            name="bot",
            token_secret="x",
            token_file=Path("/tmp/x.token"),
        )


def test_daemon_config():
    """Test DaemonConfig model."""
    config = DaemonConfig(
        state_dir=Path("/tmp/daemon"),
        log_level="info",
        default_workspace_dir=Path("/tmp/workspace"),
        default_agent_file="default",
        discord_bots=[DiscordBotConfig(name="bot", token_secret="bot-secret")],
    )
    assert config.state_dir == Path("/tmp/daemon")
    assert config.log_level == "info"
    assert config.default_workspace_dir == Path("/tmp/workspace")
    assert len(config.discord_bots) == 1


def test_runtime_property_assembles_defaults():
    """DaemonConfig.runtime collects the default_* keys into one object."""
    config = DaemonConfig(
        default_workspace_dir=Path("/tmp/workspace"),
        default_agent_file="odyn",
        default_model="openai:gpt-4o",
        default_compaction_model="openai:gpt-4o-mini",
        default_context_limit=200000,
        default_max_turns=25,
        timezone="America/Chicago",
        sandbox=SandboxSettings(enabled=True, allow_domains=["github.com"]),
    )
    runtime = config.runtime
    assert runtime.workspace_dir == Path("/tmp/workspace")
    assert runtime.agent_file == "odyn"
    assert runtime.model == "openai:gpt-4o"
    assert runtime.compaction_model == "openai:gpt-4o-mini"
    assert runtime.context_limit == 200000
    assert runtime.max_turns == 25
    assert runtime.timezone == "America/Chicago"
    assert runtime.sandbox.allow_domains == ["github.com"]


def test_load_daemon_config(tmp_path):
    """Test loading daemon config from YAML."""
    config_file = _write_config(
        tmp_path,
        {
            "state_dir": str(tmp_path / "daemon"),
            "log_level": "debug",
            "default_workspace_dir": str(tmp_path / "workspace"),
            "default_agent_file": "default",
            "default_context_limit": 100000,
            "discord_bots": [{"name": "test-bot", "token_secret": "test-bot-secret", "command_prefix": "!"}],
        },
    )

    config = load_daemon_config(config_file)

    assert config.log_level == "debug"
    assert config.default_workspace_dir == tmp_path / "workspace"
    assert config.default_agent_file == "default"
    assert config.default_context_limit == 100000
    assert len(config.discord_bots) == 1
    assert config.discord_bots[0].name == "test-bot"
    assert config.discord_bots[0].token_secret == "test-bot-secret"


def test_load_daemon_config_rejects_legacy_agents_block(tmp_path):
    """A stale `agents:` block must fail loudly with a migration hint.

    Pydantic ignores unknown keys, so without this guard the daemon would boot
    with no default workspace instead of reporting the stale config.
    """
    config_file = _write_config(
        tmp_path,
        {"agents": {"test": {"workspace_dir": str(tmp_path / "workspace"), "agent_file": "default"}}},
    )

    with pytest.raises(ValueError, match="'agents:' block is no longer supported"):
        load_daemon_config(config_file)


def test_load_daemon_config_rejects_legacy_plain_token(tmp_path):
    """Test that legacy plaintext token: field is rejected with a migration hint."""
    config_file = _write_config(
        tmp_path,
        _defaults(tmp_path, discord_bots=[{"name": "legacy-bot", "token": "PLAINTEXT-DISCORD-TOKEN"}]),
    )

    with pytest.raises(ValueError, match="plaintext 'token:' is no longer supported"):
        load_daemon_config(config_file)


def test_load_daemon_config_token_file_path_expanded(tmp_path, monkeypatch):
    """Test that ~ in token_file path is expanded."""
    monkeypatch.setenv("HOME", str(tmp_path))
    config_file = _write_config(
        tmp_path,
        _defaults(tmp_path, discord_bots=[{"name": "test-bot", "token_file": "~/discord.token"}]),
    )

    config = load_daemon_config(config_file)

    assert config.discord_bots[0].token_file == tmp_path / "discord.token"


def test_load_daemon_config_token_file_env_var_expanded(tmp_path, monkeypatch):
    """Test that ${VAR} in token_file path is expanded."""
    monkeypatch.setenv("TSUGITE_TEST_TOKEN_DIR", str(tmp_path / "secrets-dir"))
    config_file = _write_config(
        tmp_path,
        _defaults(
            tmp_path,
            discord_bots=[{"name": "test-bot", "token_file": "${TSUGITE_TEST_TOKEN_DIR}/discord.token"}],
        ),
    )

    config = load_daemon_config(config_file)

    assert config.discord_bots[0].token_file == tmp_path / "secrets-dir" / "discord.token"


def test_load_daemon_config_workspace_dir_expanded(tmp_path, monkeypatch):
    """~ in default_workspace_dir is expanded."""
    monkeypatch.setenv("HOME", str(tmp_path))
    config_file = _write_config(tmp_path, {"default_workspace_dir": "~/ws", "default_agent_file": "default"})

    config = load_daemon_config(config_file)

    assert config.default_workspace_dir == tmp_path / "ws"


def test_resolve_token_from_secret(secret_backend):
    """Test that resolve_token() returns the secret value from the backend."""
    secret_backend.set("my-discord-secret", "actual-token-value")

    config = DiscordBotConfig(name="bot", token_secret="my-discord-secret")
    assert config.resolve_token() == "actual-token-value"


def test_resolve_token_secret_missing_raises(secret_backend):
    """Test that a missing secret raises RuntimeError."""
    config = DiscordBotConfig(name="bot", token_secret="does-not-exist")
    with pytest.raises(RuntimeError, match="secret 'does-not-exist' not found"):
        config.resolve_token()


def test_resolve_token_from_file(tmp_path):
    """Test that resolve_token() reads the token from a file."""
    token_file = tmp_path / "discord.token"
    token_file.write_text("file-token-value\n", encoding="utf-8")

    config = DiscordBotConfig(name="bot", token_file=token_file)
    assert config.resolve_token() == "file-token-value"


def test_load_daemon_config_not_found():
    """Test loading non-existent config file."""
    with pytest.raises(ValueError, match="Daemon config not found"):
        load_daemon_config(Path("/nonexistent/daemon.yaml"))


def test_runtime_defaults_context_limit_unset():
    """Test RuntimeDefaults default values."""
    config = RuntimeDefaults(workspace_dir=Path("/tmp/workspace"), agent_file="default")
    assert config.context_limit is None  # auto-detected at startup


def test_discord_bot_config_defaults():
    """Test DiscordBotConfig default values."""
    config = DiscordBotConfig(name="bot", token_secret="x")
    assert config.command_prefix == "!"  # default
    assert config.dm_policy == "allowlist"  # default
    assert config.allow_from == []  # default


def test_sandbox_settings_defaults():
    """SandboxSettings is disabled with no network restrictions by default."""
    sb = SandboxSettings()
    assert sb.enabled is False
    assert sb.no_network is False
    assert sb.allow_domains == []
    assert sb.extra_ro_binds == []
    assert sb.extra_rw_binds == []


def test_sandbox_defaults_none():
    """No sandbox block resolves to None (treated as disabled)."""
    config = RuntimeDefaults(workspace_dir=Path("/tmp/workspace"), agent_file="default")
    assert config.sandbox is None


def test_sandbox_block_reaches_the_runtime(tmp_path):
    """The daemon-wide sandbox block is what the runtime resolves to."""
    config_file = _write_config(
        tmp_path,
        _defaults(tmp_path, sandbox={"enabled": True, "allow_domains": ["github.com"]}),
    )
    config = load_daemon_config(config_file)
    assert config.runtime.sandbox.enabled is True
    assert config.runtime.sandbox.allow_domains == ["github.com"]


def test_sandbox_absent_leaves_runtime_unsandboxed(tmp_path):
    config_file = _write_config(tmp_path, _defaults(tmp_path))
    config = load_daemon_config(config_file)
    assert config.runtime.sandbox is None


def test_sandbox_extra_binds_path_expansion(tmp_path, monkeypatch):
    """~ in sandbox bind paths is expanded to absolute paths."""
    monkeypatch.setenv("HOME", str(tmp_path))
    config_file = _write_config(
        tmp_path,
        _defaults(
            tmp_path,
            sandbox={"enabled": True, "extra_ro_binds": ["~/creds"], "extra_rw_binds": ["~/scratch"]},
        ),
    )
    config = load_daemon_config(config_file)
    assert config.sandbox.extra_ro_binds == [tmp_path / "creds"]
    assert config.sandbox.extra_rw_binds == [tmp_path / "scratch"]


def test_http_config_image_defaults():
    from tsugite_daemon.config import HTTPConfig

    cfg = HTTPConfig()
    assert cfg.image_max_edge == 1568
    assert cfg.image_quality == 0.85


def test_http_config_image_overrides_from_yaml(tmp_path):
    config_file = _write_config(
        tmp_path,
        _defaults(tmp_path, http={"enabled": True, "image_max_edge": 1024, "image_quality": 0.7}),
    )
    config = load_daemon_config(config_file)
    assert config.http.image_max_edge == 1024
    assert config.http.image_quality == 0.7


def test_discord_session_name_must_be_a_valid_alias():
    """It routes DMs by alias, so a value the alias contract rejects has to fail at
    config load rather than on the first DM."""
    with pytest.raises(ValidationError):
        DiscordBotConfig(name="bot", token_secret="x", session_name="not a slug")


def test_discord_session_name_may_be_empty():
    config = DiscordBotConfig(name="bot", token_secret="x", session_name="")
    assert config.session_name == ""
