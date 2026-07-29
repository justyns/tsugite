"""Gateway.reload_config: re-reads daemon.yaml and hot-reconciles the HTTP
agent set (add/remove/update in place), while boot-only sections are reported
as restart_required rather than silently ignored."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from tsugite_daemon.config import load_daemon_config
from tsugite_daemon.gateway import Gateway


def _write_config(path, agents: dict[str, str], port: int = 8321, extra: str = "") -> None:
    agent_blocks = "\n".join(
        f"  {name}:\n    agent_file: {agent_file}\n    workspace_dir: {path.parent / 'ws'}"
        for name, agent_file in agents.items()
    )
    path.write_text(
        f"state_dir: {path.parent / 'state'}\n"
        f"http:\n  enabled: true\n  host: 127.0.0.1\n  port: {port}\n"
        f"agents:\n{agent_blocks}\n" + extra
    )


@pytest.fixture
def env(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    for name in ("alpha", "beta", "gamma"):
        (ws / f"{name}.md").write_text(f"---\nname: {name}\n---\nYou are {name}.\n")
    cfg_path = tmp_path / "daemon.yaml"
    _write_config(cfg_path, {"alpha": "alpha.md", "beta": "beta.md"})
    config = load_daemon_config(cfg_path)
    # start() enriches every agent config with a resolved context_limit; the
    # reload diff compares against that enriched state, so mirror it here.
    for cfg in config.agents.values():
        cfg.context_limit = 128000
    gateway = Gateway(config, config_path=cfg_path)
    # Stand in for the started state: a server stub holding the live adapter
    # dict + event bus, and the session store the reload hands to new adapters.
    alpha = SimpleNamespace(agent_config=config.agents["alpha"], event_bus=None)
    beta = SimpleNamespace(agent_config=config.agents["beta"], event_bus=None)
    gateway._http_server = SimpleNamespace(adapters={"alpha": alpha, "beta": beta}, event_bus=object())
    gateway._session_store = SimpleNamespace(update_context_limit=lambda *a, **k: None)
    return gateway, cfg_path, ws


@pytest.mark.asyncio
async def test_reload_adds_removes_and_updates_agents(env, monkeypatch):
    gateway, cfg_path, ws = env

    built = []

    class FakeAdapter:
        def __init__(self, name, cfg, session_store, identity_map=None):
            built.append(name)
            self.agent_name = name
            self.agent_config = cfg
            self.event_bus = None

    import tsugite_daemon.adapters.http as http_mod

    monkeypatch.setattr(http_mod, "HTTPAgentAdapter", FakeAdapter)

    # beta gone, gamma new, alpha's model changed.
    _write_config(cfg_path, {"alpha": "alpha.md", "gamma": "gamma.md"})
    text = cfg_path.read_text().replace("  alpha:\n", "  alpha:\n    model: openai:gpt-5.4-mini\n", 1)
    cfg_path.write_text(text)

    result = await gateway.reload_config()

    assert result["added"] == ["gamma"]
    assert result["removed"] == ["beta"]
    assert result["updated"] == ["alpha"]
    assert built == ["gamma"]
    adapters = gateway._http_server.adapters
    assert set(adapters) == {"alpha", "gamma"}
    assert adapters["gamma"].event_bus is gateway._http_server.event_bus
    # The updated agent got the NEW config object hot-swapped in.
    assert adapters["alpha"].agent_config.model == "openai:gpt-5.4-mini"
    assert set(gateway.config.agents) == {"alpha", "gamma"}


def test_every_config_section_is_classified_hot_or_boot_only():
    """reload_config must classify every DaemonConfig section as either
    hot-reconciled or boot-only. A newly added section that lands in neither would
    silently escape restart_required detection, so fail until it's classified."""
    from tsugite_daemon.config import DaemonConfig
    from tsugite_daemon.gateway import BOOT_ONLY_SECTIONS

    hot_reconciled = {"agents", "notification_channels", "identity_links"}
    unclassified = set(DaemonConfig.model_fields) - hot_reconciled - set(BOOT_ONLY_SECTIONS)
    assert not unclassified, f"Unclassified daemon config sections: {unclassified}"


@pytest.mark.asyncio
async def test_reload_reports_boot_only_sections(env):
    gateway, cfg_path, ws = env
    _write_config(cfg_path, {"alpha": "alpha.md", "beta": "beta.md"}, port=9999)

    result = await gateway.reload_config()

    assert result["restart_required"] == ["http"]
    assert result["added"] == [] and result["removed"] == [] and result["updated"] == []


@pytest.mark.asyncio
async def test_reload_skips_unresolvable_new_agent(env, monkeypatch):
    gateway, cfg_path, ws = env
    _write_config(cfg_path, {"alpha": "alpha.md", "beta": "beta.md", "ghost": "missing.md"})

    result = await gateway.reload_config()

    assert result["skipped"] == ["ghost"]
    assert "ghost" not in gateway._http_server.adapters


@pytest.mark.asyncio
async def test_reload_rebuilds_identity_map_in_place(env):
    gateway, cfg_path, ws = env
    shared_ref = gateway._identity_map
    _write_config(
        cfg_path,
        {"alpha": "alpha.md", "beta": "beta.md"},
        extra='identity_links:\n  alice: ["discord:42"]\n',
    )

    await gateway.reload_config()

    assert gateway._identity_map is shared_ref
    assert shared_ref == {"discord:42": "alice"}
