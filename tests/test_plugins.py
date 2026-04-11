"""Tests for plugin discovery and loading."""

from unittest.mock import MagicMock, patch

from tsugite.hooks import HookRule
from tsugite.plugins import discover_plugins, load_adapter_plugins, load_hook_plugins, load_tool_plugins


def _make_entry_point(name, value, group):
    ep = MagicMock()
    ep.name = name
    ep.value = value
    ep.group = group
    return ep


def _mock_entry_points(eps):
    """Return a side_effect function that filters by group kwarg."""

    def _entry_points(group=None):
        return [ep for ep in eps if ep.group == group]

    return _entry_points


class TestDiscoverPlugins:
    def test_empty_when_no_plugins(self):
        with patch("tsugite.plugins.importlib.metadata.entry_points", return_value=[]):
            result = discover_plugins()
        assert result == []

    def test_discovers_tool_plugin(self):
        ep = _make_entry_point("weather", "tsugite_weather:register_tools", "tsugite.tools")
        with patch("tsugite.plugins.importlib.metadata.entry_points", side_effect=_mock_entry_points([ep])):
            result = discover_plugins()
        assert len(result) == 1
        assert result[0].name == "weather"
        assert result[0].group == "tsugite.tools"
        assert result[0].enabled is True

    def test_discovers_adapter_plugin(self):
        ep = _make_entry_point("slack", "tsugite_slack:create_adapter", "tsugite.adapters")
        with patch("tsugite.plugins.importlib.metadata.entry_points", side_effect=_mock_entry_points([ep])):
            result = discover_plugins()
        assert len(result) == 1
        assert result[0].group == "tsugite.adapters"

    def test_respects_enabled_false(self):
        ep = _make_entry_point("weather", "tsugite_weather:register_tools", "tsugite.tools")
        config = {"weather": {"enabled": False}}
        with patch("tsugite.plugins.importlib.metadata.entry_points", side_effect=_mock_entry_points([ep])):
            result = discover_plugins(plugin_config=config)
        assert result[0].enabled is False

    def test_enabled_by_default(self):
        ep = _make_entry_point("weather", "tsugite_weather:register_tools", "tsugite.tools")
        with patch("tsugite.plugins.importlib.metadata.entry_points", side_effect=_mock_entry_points([ep])):
            result = discover_plugins(plugin_config={})
        assert result[0].enabled is True


class TestLoadToolPlugins:
    def test_registers_tools(self):
        def my_tool(x: str) -> str:
            """A test tool."""
            return x

        register_fn = MagicMock(return_value=[my_tool])
        ep = _make_entry_point("test-plugin", "test_plugin:register", "tsugite.tools")
        ep.load.return_value = register_fn

        with patch("tsugite.plugins.importlib.metadata.entry_points", return_value=[ep]):
            with patch("tsugite.tools._register_tool") as mock_register:
                results = load_tool_plugins()

        register_fn.assert_called_once_with({})
        mock_register.assert_called_once_with(my_tool)
        assert len(results) == 1
        assert results[0].loaded is True

    def test_skips_disabled(self):
        ep = _make_entry_point("test-plugin", "test_plugin:register", "tsugite.tools")

        with patch("tsugite.plugins.importlib.metadata.entry_points", return_value=[ep]):
            results = load_tool_plugins(plugin_config={"test-plugin": {"enabled": False}})

        ep.load.assert_not_called()
        assert results[0].enabled is False
        assert results[0].loaded is False

    def test_graceful_failure(self):
        ep = _make_entry_point("bad-plugin", "bad_plugin:register", "tsugite.tools")
        ep.load.side_effect = ImportError("no such module")

        with patch("tsugite.plugins.importlib.metadata.entry_points", return_value=[ep]):
            results = load_tool_plugins()

        assert len(results) == 1
        assert results[0].loaded is False
        assert "no such module" in results[0].error

    def test_passes_config_to_register_fn(self):
        register_fn = MagicMock(return_value=[])
        ep = _make_entry_point("weather", "tsugite_weather:register", "tsugite.tools")
        ep.load.return_value = register_fn

        config = {"weather": {"api_key": "abc123"}}
        with patch("tsugite.plugins.importlib.metadata.entry_points", return_value=[ep]):
            with patch("tsugite.tools._register_tool"):
                load_tool_plugins(plugin_config=config)

        register_fn.assert_called_once_with({"api_key": "abc123"})


class TestLoadAdapterPlugins:
    def test_instantiates_adapter(self):
        mock_adapter = MagicMock()
        factory = MagicMock(return_value=mock_adapter)
        ep = _make_entry_point("slack", "tsugite_slack:create", "tsugite.adapters")
        ep.load.return_value = factory

        session_store = MagicMock()
        identity_map = {"user1": "agent1"}
        agents_config = {"assistant": {}}

        with patch("tsugite.plugins.importlib.metadata.entry_points", return_value=[ep]):
            results = load_adapter_plugins(
                plugin_config={"slack": {"token": "xoxb-123"}},
                session_store=session_store,
                identity_map=identity_map,
                agents_config=agents_config,
            )

        assert len(results) == 1
        info, adapter = results[0]
        assert info.loaded is True
        assert adapter is mock_adapter
        factory.assert_called_once_with(
            config={"token": "xoxb-123"},
            agents_config=agents_config,
            session_store=session_store,
            identity_map=identity_map,
        )

    def test_skips_disabled_adapter(self):
        ep = _make_entry_point("slack", "tsugite_slack:create", "tsugite.adapters")

        with patch("tsugite.plugins.importlib.metadata.entry_points", return_value=[ep]):
            results = load_adapter_plugins(
                plugin_config={"slack": {"enabled": False}},
                session_store=MagicMock(),
                identity_map={},
                agents_config={},
            )

        ep.load.assert_not_called()
        info, adapter = results[0]
        assert info.enabled is False
        assert adapter is None

    def test_graceful_adapter_failure(self):
        ep = _make_entry_point("bad", "bad_adapter:create", "tsugite.adapters")
        ep.load.side_effect = Exception("connection failed")

        with patch("tsugite.plugins.importlib.metadata.entry_points", return_value=[ep]):
            results = load_adapter_plugins(
                plugin_config={},
                session_store=MagicMock(),
                identity_map={},
                agents_config={},
            )

        info, adapter = results[0]
        assert adapter is None
        assert "connection failed" in info.error


class TestLoadHookPlugins:
    def setup_method(self):
        """Reset plugin hooks state between tests."""
        import tsugite.plugins

        tsugite.plugins._plugin_hooks = {}

    def test_registers_hooks(self):
        async def my_hook(ctx):
            return "hello"

        rules = {"pre_context_build": [HookRule(type="python", hook_callable=my_hook, name="test")]}
        register_fn = MagicMock(return_value=rules)
        ep = _make_entry_point("test-hooks", "test_hooks:register", "tsugite.hooks")
        ep.load.return_value = register_fn

        with patch("tsugite.plugins.importlib.metadata.entry_points", return_value=[ep]):
            results = load_hook_plugins()

        register_fn.assert_called_once_with({})
        assert len(results) == 1
        assert results[0].loaded is True

        from tsugite.plugins import get_plugin_hooks

        hooks = get_plugin_hooks()
        assert "pre_context_build" in hooks
        assert len(hooks["pre_context_build"]) == 1

    def test_skips_disabled(self):
        ep = _make_entry_point("test-hooks", "test_hooks:register", "tsugite.hooks")

        with patch("tsugite.plugins.importlib.metadata.entry_points", return_value=[ep]):
            results = load_hook_plugins(plugin_config={"test-hooks": {"enabled": False}})

        ep.load.assert_not_called()
        assert results[0].enabled is False

    def test_graceful_failure(self):
        ep = _make_entry_point("bad-hooks", "bad_hooks:register", "tsugite.hooks")
        ep.load.side_effect = ImportError("no such module")

        with patch("tsugite.plugins.importlib.metadata.entry_points", return_value=[ep]):
            results = load_hook_plugins()

        assert results[0].loaded is False
        assert "no such module" in results[0].error

    def test_passes_config(self):
        register_fn = MagicMock(return_value={})
        ep = _make_entry_point("uridx", "uridx:register_hooks", "tsugite.hooks")
        ep.load.return_value = register_fn

        config = {"uridx": {"api_url": "http://localhost:8080"}}
        with patch("tsugite.plugins.importlib.metadata.entry_points", return_value=[ep]):
            load_hook_plugins(plugin_config=config)

        register_fn.assert_called_once_with({"api_url": "http://localhost:8080"})

    def test_multiple_plugins_merged(self):
        async def hook_a(ctx):
            pass

        async def hook_b(ctx):
            pass

        register_a = MagicMock(return_value={
            "pre_context_build": [HookRule(type="python", hook_callable=hook_a, name="a")],
        })
        register_b = MagicMock(return_value={
            "pre_context_build": [HookRule(type="python", hook_callable=hook_b, name="b")],
            "session_end": [HookRule(type="python", hook_callable=hook_b, name="b-end")],
        })

        ep_a = _make_entry_point("plugin-a", "a:register", "tsugite.hooks")
        ep_a.load.return_value = register_a
        ep_b = _make_entry_point("plugin-b", "b:register", "tsugite.hooks")
        ep_b.load.return_value = register_b

        with patch("tsugite.plugins.importlib.metadata.entry_points", return_value=[ep_a, ep_b]):
            load_hook_plugins()

        from tsugite.plugins import get_plugin_hooks

        hooks = get_plugin_hooks()
        assert len(hooks["pre_context_build"]) == 2
        assert len(hooks["session_end"]) == 1

    def test_discovers_hook_plugins(self):
        ep = _make_entry_point("uridx", "uridx:register_hooks", "tsugite.hooks")
        with patch("tsugite.plugins.importlib.metadata.entry_points", side_effect=_mock_entry_points([ep])):
            result = discover_plugins()
        assert any(p.group == "tsugite.hooks" for p in result)
