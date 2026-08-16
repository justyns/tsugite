"""Tests for plugin discovery and loading."""

from unittest.mock import MagicMock, patch

import pytest

from tsugite.plugins import (
    GROUP_PLUGINS,
    discover_plugins,
    load_adapter_plugins,
    load_module_only_plugins,
)
from tsugite.ui_surfaces import registered_ui_surfaces


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

    def test_discovers_decorator_plugin(self):
        ep = _make_entry_point("weather", "tsugite_weather", "tsugite.plugins")
        with patch("tsugite.plugins.importlib.metadata.entry_points", side_effect=_mock_entry_points([ep])):
            result = discover_plugins()
        assert len(result) == 1
        assert result[0].name == "weather"
        assert result[0].group == "tsugite.plugins"
        assert result[0].enabled is True

    def test_discovers_adapter_plugin(self):
        ep = _make_entry_point("slack", "tsugite_slack:create_adapter", "tsugite.adapters")
        with patch("tsugite.plugins.importlib.metadata.entry_points", side_effect=_mock_entry_points([ep])):
            result = discover_plugins()
        assert len(result) == 1
        assert result[0].group == "tsugite.adapters"

    def test_respects_enabled_false(self):
        ep = _make_entry_point("weather", "tsugite_weather", "tsugite.plugins")
        config = {"weather": {"enabled": False}}
        with patch("tsugite.plugins.importlib.metadata.entry_points", side_effect=_mock_entry_points([ep])):
            result = discover_plugins(plugin_config=config)
        assert result[0].enabled is False

    def test_enabled_by_default(self):
        ep = _make_entry_point("weather", "tsugite_weather", "tsugite.plugins")
        with patch("tsugite.plugins.importlib.metadata.entry_points", side_effect=_mock_entry_points([ep])):
            result = discover_plugins(plugin_config={})
        assert result[0].enabled is True


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


class TestUnifiedPluginsGroup:
    """Plugins can declare a single tsugite.plugins entry point and rely on
    @tool / @hook / @subscribe decorators to register themselves at import."""

    def test_imports_module_via_unified_entry_point(self):
        import types

        from tsugite.plugins import GROUP_PLUGINS, load_module_only_plugins

        fake_module = types.ModuleType("fake_unified_plugin")
        ep = _make_entry_point("kitchen-sink", "fake_unified_plugin", "tsugite.plugins")
        ep.load.return_value = fake_module

        with patch("tsugite.plugins.importlib.metadata.entry_points", return_value=[ep]):
            results = load_module_only_plugins(GROUP_PLUGINS)

        assert results[0].loaded is True
        ep.load.assert_called_once()

    def test_skips_disabled_unified_plugin(self):
        from tsugite.plugins import GROUP_PLUGINS, load_module_only_plugins

        ep = _make_entry_point("kitchen-sink", "fake_unified_plugin", "tsugite.plugins")
        with patch("tsugite.plugins.importlib.metadata.entry_points", return_value=[ep]):
            results = load_module_only_plugins(GROUP_PLUGINS, plugin_config={"kitchen-sink": {"enabled": False}})

        ep.load.assert_not_called()
        assert results[0].enabled is False

    def test_unified_group_in_discover(self):
        ep = _make_entry_point("kitchen-sink", "fake_unified_plugin", "tsugite.plugins")
        with patch("tsugite.plugins.importlib.metadata.entry_points", side_effect=_mock_entry_points([ep])):
            result = discover_plugins()
        assert any(p.group == "tsugite.plugins" for p in result)


class TestLoadCommandPlugins:
    """Plugins contribute daemon slash commands via a module-only
    tsugite.commands entry point whose import runs @adapter_command."""

    def test_command_group_in_discover(self):
        ep = _make_entry_point("fake-cmds", "fake_command_plugin", "tsugite.commands")
        with patch("tsugite.plugins.importlib.metadata.entry_points", side_effect=_mock_entry_points([ep])):
            result = discover_plugins()
        assert any(p.group == "tsugite.commands" for p in result)

    def test_module_only_command_entry_point(self):
        import types

        from tsugite.plugins import GROUP_COMMANDS, load_module_only_plugins

        fake_module = types.ModuleType("fake_command_plugin")
        ep = _make_entry_point("cmds-plugin", "fake_command_plugin", "tsugite.commands")
        ep.load.return_value = fake_module

        with patch("tsugite.plugins.importlib.metadata.entry_points", return_value=[ep]):
            results = load_module_only_plugins(GROUP_COMMANDS)

        assert results[0].loaded is True
        ep.load.assert_called_once()

    def test_skips_disabled_command_plugin(self):
        from tsugite.plugins import GROUP_COMMANDS, load_module_only_plugins

        ep = _make_entry_point("cmds-plugin", "fake_command_plugin", "tsugite.commands")
        with patch("tsugite.plugins.importlib.metadata.entry_points", return_value=[ep]):
            results = load_module_only_plugins(GROUP_COMMANDS, plugin_config={"cmds-plugin": {"enabled": False}})

        ep.load.assert_not_called()
        assert results[0].enabled is False


def test_all_group_constants_are_enumerated():
    """Every GROUP_* entry-point group must appear in PLUGIN_GROUPS, or
    discover_plugins() (and the /api/plugins endpoint built on it) silently
    omits that group's plugins."""
    import tsugite.plugins as plugins_mod

    group_consts = {v for k, v in vars(plugins_mod).items() if k.startswith("GROUP_") and isinstance(v, str)}
    missing = group_consts - set(plugins_mod.PLUGIN_GROUPS)
    assert not missing, f"GROUP_* constants missing from PLUGIN_GROUPS: {sorted(missing)}"


class TestLocalFilePlugins:
    """A `path` entry in the plugin config points at a single trusted .py file that
    registers the same way an installed decorator plugin does."""

    @pytest.fixture(autouse=True)
    def _no_installed_plugins(self):
        """Local files are the only plugins in play unless a test says otherwise."""
        with patch("tsugite.plugins.importlib.metadata.entry_points", return_value=[]):
            yield

    def _write_plugin(self, tmp_path, body="MARKER = 'loaded'\n"):
        plugin_file = tmp_path / "dashboard.py"
        plugin_file.write_text(body)
        return plugin_file

    def test_discovered_with_its_path_as_the_entry_point(self, tmp_path):
        plugin_file = self._write_plugin(tmp_path)
        result = discover_plugins({"dashboard": {"path": str(plugin_file)}})

        assert [(p.name, p.group, p.entry_point) for p in result] == [
            ("dashboard", "tsugite.plugins", str(plugin_file))
        ]

    def test_loading_imports_the_file(self, tmp_path):
        ran = tmp_path / "ran"
        plugin_file = self._write_plugin(tmp_path, f"open({str(ran)!r}, 'w').close()\n")

        results = load_module_only_plugins(GROUP_PLUGINS, {"dashboard": {"path": str(plugin_file)}})

        assert ran.exists()
        assert results[0].loaded is True

    def test_disabled_file_is_not_imported(self, tmp_path):
        plugin_file = self._write_plugin(tmp_path, "raise AssertionError('must not be imported')\n")
        config = {"dashboard": {"path": str(plugin_file), "enabled": False}}

        results = load_module_only_plugins(GROUP_PLUGINS, config)

        assert [(r.name, r.enabled, r.loaded) for r in results] == [("dashboard", False, False)]

    def test_import_error_is_isolated_and_reported(self, tmp_path):
        broken = tmp_path / "broken.py"
        broken.write_text("raise RuntimeError('boom')\n")
        working = self._write_plugin(tmp_path)
        config = {"broken": {"path": str(broken)}, "dashboard": {"path": str(working)}}

        results = load_module_only_plugins(GROUP_PLUGINS, config)

        assert [(r.name, r.loaded, "boom" in (r.error or "")) for r in results] == [
            ("broken", False, True),
            ("dashboard", True, False),
        ]

    def test_relative_path_resolves_against_the_config_dir(self, tmp_path, monkeypatch):
        config_dir = tmp_path / "cfg"
        config_dir.mkdir()
        plugin_file = config_dir / "dashboard.py"
        plugin_file.write_text("MARKER = 'loaded'\n")
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr("tsugite.config.get_config_path", lambda: config_dir / "config.json")

        result = discover_plugins({"dashboard": {"path": "dashboard.py"}})

        assert [p.entry_point for p in result] == [str(plugin_file)]

    def test_installed_plugin_of_the_same_name_wins(self, tmp_path):
        plugin_file = self._write_plugin(tmp_path)
        ep = _make_entry_point("dashboard", "tsugite_dashboard", "tsugite.plugins")

        with patch("tsugite.plugins.importlib.metadata.entry_points", side_effect=_mock_entry_points([ep])):
            result = discover_plugins({"dashboard": {"path": str(plugin_file)}})

        assert [p.entry_point for p in result] == ["tsugite_dashboard"]


_REGISTERS_A_PAGE = """\
from tsugite.ui_surfaces import ui_surface

@ui_surface(kind="dash", label="Homelab", nav=True)
def page():
    return "<h1>ok</h1>"
"""


class TestUISurfaceAttribution:
    """A surface registers at import time and cannot know its own entry point, so the
    loader stamps the plugin name the config gave it."""

    @pytest.fixture(autouse=True)
    def _no_installed_plugins(self):
        with patch("tsugite.plugins.importlib.metadata.entry_points", return_value=[]):
            yield

    def _write_plugin(self, tmp_path, name, body=_REGISTERS_A_PAGE):
        plugin_file = tmp_path / f"{name}.py"
        plugin_file.write_text(body)
        return plugin_file

    def test_stamped_with_the_configured_plugin_name(self, tmp_path):
        plugin_file = self._write_plugin(tmp_path, "dashboard")

        load_module_only_plugins(GROUP_PLUGINS, {"dashboard": {"path": str(plugin_file)}})

        assert {name: [d["kind"] for d in ds] for name, ds in registered_ui_surfaces().items()} == {
            "dashboard": ["dash"]
        }

    def test_two_plugins_do_not_cross_attribute(self, tmp_path):
        first = self._write_plugin(tmp_path, "alpha")
        second = self._write_plugin(tmp_path, "beta", _REGISTERS_A_PAGE.replace("dash", "beta_dash"))
        config = {"alpha": {"path": str(first)}, "beta": {"path": str(second)}}

        load_module_only_plugins(GROUP_PLUGINS, config)

        assert {name: [d["kind"] for d in ds] for name, ds in registered_ui_surfaces().items()} == {
            "alpha": ["dash"],
            "beta": ["beta_dash"],
        }

    def test_a_plugin_that_raises_after_registering_gets_no_page(self, tmp_path):
        broken = self._write_plugin(tmp_path, "broken", _REGISTERS_A_PAGE + "\nraise RuntimeError('boom')\n")
        working = self._write_plugin(tmp_path, "dashboard")
        config = {"broken": {"path": str(broken)}, "dashboard": {"path": str(working)}}

        results = load_module_only_plugins(GROUP_PLUGINS, config)

        assert [(r.name, r.loaded) for r in results] == [("broken", False), ("dashboard", True)]
        assert list(registered_ui_surfaces()) == ["dashboard"]
