"""Config resolution for the ACP command/env."""

from __future__ import annotations


def test_default_command_is_npx_claude_agent_acp():
    from tsugite_acp.config import DEFAULT_COMMAND, resolve_command

    cfg = resolve_command(env_override="")  # empty string disables env override
    assert cfg.argv() == DEFAULT_COMMAND


def test_env_override_wins(monkeypatch):
    from tsugite_acp.config import resolve_command

    monkeypatch.delenv("TSUGITE_ACP_COMMAND", raising=False)
    cfg = resolve_command(env_override="/usr/local/bin/my-acp --flag")
    assert cfg.command == "/usr/local/bin/my-acp"
    assert cfg.args == ["--flag"]


def test_environment_variable_picked_up(monkeypatch):
    from tsugite_acp.config import resolve_command

    monkeypatch.setenv("TSUGITE_ACP_COMMAND", "node /opt/acp.js --port 0")
    cfg = resolve_command()
    assert cfg.command == "node"
    assert cfg.args == ["/opt/acp.js", "--port", "0"]


def test_config_dict_used_when_no_env_override(monkeypatch):
    from tsugite_acp.config import resolve_command

    monkeypatch.delenv("TSUGITE_ACP_COMMAND", raising=False)
    cfg = resolve_command(config={"command": "claude-agent-acp"})
    assert cfg.argv() == ["claude-agent-acp"]


def test_config_env_layered_over_inherit(monkeypatch):
    from tsugite_acp.config import resolve_command

    monkeypatch.delenv("TSUGITE_ACP_COMMAND", raising=False)
    monkeypatch.setenv("PATH", "/inherit")
    cfg = resolve_command(config={"env": {"ANTHROPIC_API_KEY": "sk-x"}})
    assert cfg.env["PATH"] == "/inherit"
    assert cfg.env["ANTHROPIC_API_KEY"] == "sk-x"


def test_empty_command_raises(monkeypatch):
    import pytest
    from tsugite_acp.config import resolve_command

    monkeypatch.delenv("TSUGITE_ACP_COMMAND", raising=False)
    with pytest.raises(ValueError):
        resolve_command(env_override="   ")  # whitespace only
