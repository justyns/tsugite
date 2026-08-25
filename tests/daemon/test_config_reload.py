"""Gateway.reload_config: re-reads daemon.yaml and hot-swaps the runtime defaults
onto the live adapter, while boot-only sections are reported as restart_required
rather than silently ignored."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from tsugite_daemon.config import load_daemon_config
from tsugite_daemon.gateway import Gateway


def _write_config(path, agent_file: str = "alpha.md", port: int = 8321, extra: str = "") -> None:
    path.write_text(
        f"state_dir: {path.parent / 'state'}\n"
        f"http:\n  enabled: true\n  host: 127.0.0.1\n  port: {port}\n"
        f"default_workspace_dir: {path.parent / 'ws'}\n"
        f"default_agent_file: {agent_file}\n" + extra
    )


@pytest.fixture
def env(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    for name in ("alpha", "beta"):
        (ws / f"{name}.md").write_text(f"---\nname: {name}\n---\nYou are {name}.\n")
    cfg_path = tmp_path / "daemon.yaml"
    _write_config(cfg_path)
    config = load_daemon_config(cfg_path)
    gateway = Gateway(config, config_path=cfg_path)
    # Stand in for the started state: a server stub holding the live adapter and
    # event bus, plus the session store reload updates the context limit on.
    # start() enriches the runtime with a resolved context_limit; the reload diff
    # compares against that enriched state, so mirror it here.
    runtime = config.runtime
    runtime.context_limit = Gateway._resolve_context_limit(runtime)
    adapter = SimpleNamespace(runtime=runtime, event_bus=None)
    gateway._http_server = SimpleNamespace(adapter=adapter, event_bus=object())
    gateway._session_store = SimpleNamespace(update_context_limit=lambda *a, **k: None)
    return gateway, cfg_path, ws


@pytest.mark.asyncio
async def test_reload_hot_swaps_changed_runtime_defaults(env):
    gateway, cfg_path, ws = env

    _write_config(cfg_path, agent_file="beta.md", extra="default_model: openai:gpt-5.4-mini\n")

    result = await gateway.reload_config()

    assert result["updated"] == ["runtime"]
    runtime = gateway._http_server.adapter.runtime
    assert runtime.agent_file == "beta.md"
    assert runtime.model == "openai:gpt-5.4-mini"
    assert gateway.config.default_agent_file == "beta.md"


@pytest.mark.asyncio
async def test_reload_reports_no_change_when_config_is_identical(env):
    gateway, cfg_path, ws = env

    result = await gateway.reload_config()

    assert result["updated"] == []
    assert result["restart_required"] == []


def test_every_config_section_is_classified_hot_or_boot_only():
    """reload_config must classify every DaemonConfig section as either
    hot-reconciled or boot-only. A newly added section that lands in neither would
    silently escape restart_required detection, so fail until it's classified."""
    from tsugite_daemon.config import DaemonConfig
    from tsugite_daemon.gateway import BOOT_ONLY_SECTIONS, RUNTIME_DEFAULT_FIELDS

    hot_reconciled = {"notification_channels", "identity_links", *RUNTIME_DEFAULT_FIELDS}
    unclassified = set(DaemonConfig.model_fields) - hot_reconciled - set(BOOT_ONLY_SECTIONS)
    assert not unclassified, f"Unclassified daemon config sections: {unclassified}"


@pytest.mark.asyncio
async def test_reload_reports_boot_only_sections(env):
    gateway, cfg_path, ws = env
    _write_config(cfg_path, port=9999)

    result = await gateway.reload_config()

    assert result["restart_required"] == ["http"]
    assert result["updated"] == []


@pytest.mark.asyncio
async def test_reload_rebuilds_identity_map_in_place(env):
    gateway, cfg_path, ws = env
    shared_ref = gateway._identity_map
    _write_config(cfg_path, extra='identity_links:\n  alice: ["discord:42"]\n')

    await gateway.reload_config()

    assert gateway._identity_map is shared_ref
    assert shared_ref == {"discord:42": "alice"}
