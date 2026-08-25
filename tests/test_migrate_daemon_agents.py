"""Round-trips for the one-off daemon `agents:` migration script."""

import importlib.util
from pathlib import Path

import pytest
import yaml

_spec = importlib.util.spec_from_file_location(
    "migrate_daemon_agents", Path(__file__).parent.parent / "scripts" / "migrate_daemon_agents.py"
)
migrate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(migrate)


def test_hoists_the_single_agent_and_keeps_comments():
    out = migrate.migrate_text(
        "http:\n"
        "  port: 8375\n"
        "agents:\n"
        "  main:\n"
        "    # the workspace everything runs in\n"
        "    workspace_dir: /srv/agent\n"
        "    agent_file: default\n"
        "    model: anthropic:claude-opus-4\n"
        "    timezone: UTC\n"
    )
    assert "# the workspace everything runs in" in out
    assert "agents:" not in out
    data = yaml.safe_load(out)
    assert data["default_workspace_dir"] == "/srv/agent"
    assert data["default_agent_file"] == "default"
    assert data["default_model"] == "anthropic:claude-opus-4"
    assert data["timezone"] == "UTC"
    assert data["http"]["port"] == 8375


def test_drops_an_agent_sandbox_identical_to_the_global_one():
    out = migrate.migrate_text(
        "sandbox:\n  enabled: true\nagents:\n  main:\n    workspace_dir: /srv/agent\n    sandbox:\n      enabled: true\n"
    )
    assert out.count("sandbox:") == 1
    data = yaml.safe_load(out)
    assert data["sandbox"] == {"enabled": True}


def test_hoists_an_agent_only_sandbox():
    out = migrate.migrate_text("agents:\n  main:\n    workspace_dir: /srv/agent\n    sandbox:\n      enabled: true\n")
    assert yaml.safe_load(out)["sandbox"] == {"enabled": True}


def test_strips_agent_bindings_from_discord_bots_only():
    out = migrate.migrate_text(
        "agents:\n"
        "  main:\n"
        "    workspace_dir: /srv/agent\n"
        "discord_bots:\n"
        "  - name: helper\n"
        "    agent: main\n"
        "    token: abc\n"
        "plugins:\n"
        "  tsugite-pty:\n"
        "    agent: keep-me\n"
    )
    data = yaml.safe_load(out)
    assert data["discord_bots"] == [{"name": "helper", "token": "abc"}]
    assert data["plugins"]["tsugite-pty"]["agent"] == "keep-me"


def test_refuses_more_than_one_agent():
    with pytest.raises(SystemExit, match="The daemon now runs exactly one"):
        migrate.migrate_text("agents:\n  a:\n    workspace_dir: /a\n  b:\n    workspace_dir: /b\n")


def test_refuses_a_diverging_sandbox():
    with pytest.raises(SystemExit, match="differs from the global one"):
        migrate.migrate_text(
            "sandbox:\n  enabled: true\nagents:\n  main:\n    workspace_dir: /a\n    sandbox:\n      enabled: false\n"
        )


def test_refuses_keys_it_cannot_hoist():
    with pytest.raises(SystemExit, match="cannot hoist: mystery"):
        migrate.migrate_text("agents:\n  main:\n    workspace_dir: /a\n    mystery: 1\n")


def test_refuses_a_config_with_no_agents_block():
    with pytest.raises(SystemExit, match="nothing to migrate"):
        migrate.migrate_text("http:\n  port: 8375\n")
