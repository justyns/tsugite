"""cc-driver adapter factory smoke test (tsugite.adapters entry-point contract)."""

from unittest.mock import MagicMock

from tsugite_cc_driver.adapter import CCDriverConfig, create_adapter


def test_create_adapter_exposes_plugin_surface():
    adapter = create_adapter(
        config={"enabled": True, "base_url": "http://127.0.0.1:9999"},
        agents_config={},
        session_store=MagicMock(),
        identity_map={},
    )
    # Adapter lifecycle + plugin duck-typed surface.
    assert hasattr(adapter, "start") and hasattr(adapter, "stop")
    routes = adapter.get_public_http_routes()
    assert len(routes) == 1
    assert "/hook/" in routes[0].path
    assert set(adapter.get_job_executors()) == {"cc"}


def test_create_adapter_backrefs_orchestrator_to_executor():
    adapter = create_adapter(config={"enabled": True}, agents_config={}, session_store=MagicMock(), identity_map={})
    orch = object()
    adapter.set_jobs_orchestrator(orch)
    assert adapter._executor.orchestrator is orch


def test_create_adapter_disabled_by_default_returns_none():
    """cc-driver spawns claude with skip-permissions, so it must be explicit opt-in:
    installing the plugin must NOT activate it unless daemon.yaml enables it."""
    assert create_adapter(config={}, agents_config={}, session_store=MagicMock(), identity_map={}) is None
    assert (
        create_adapter(config={"enabled": False}, agents_config={}, session_store=MagicMock(), identity_map={}) is None
    )


def test_create_adapter_enabled_returns_adapter():
    adapter = create_adapter(config={"enabled": True}, agents_config={}, session_store=MagicMock(), identity_map={})
    assert adapter is not None
    assert set(adapter.get_job_executors()) == {"cc"}


def test_config_defaults():
    cfg = CCDriverConfig()
    assert cfg.permission_mode == "bypassPermissions"
    assert cfg.completion_marker == "CCDRIVER_GOAL_COMPLETE"
    assert cfg.max_consecutive_continues == 5
    # Autonomous default is sandboxed: the trust workaround pairs with fs isolation.
    assert cfg.sandbox is True
