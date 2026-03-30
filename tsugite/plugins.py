"""Plugin discovery and loading via Python entry points."""

import importlib.metadata
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

GROUP_TOOLS = "tsugite.tools"
GROUP_ADAPTERS = "tsugite.adapters"
GROUP_KVSTORE = "tsugite.kvstore"
GROUP_PROVIDERS = "tsugite.providers"
GROUP_SECRETS = "tsugite.secrets"
PLUGIN_GROUPS = (GROUP_TOOLS, GROUP_ADAPTERS, GROUP_KVSTORE, GROUP_PROVIDERS, GROUP_SECRETS)


@dataclass
class PluginInfo:
    """Metadata about a discovered plugin."""

    name: str
    group: str
    entry_point: str
    enabled: bool = True
    loaded: bool = False
    error: str | None = None

    @classmethod
    def from_entry_point(cls, ep, group: str, **kwargs) -> "PluginInfo":
        return cls(name=ep.name, group=group, entry_point=ep.value, **kwargs)


def _iter_plugins(group: str, plugin_config: dict | None = None):
    """Yield (entry_point, config_dict, enabled) for each plugin in a group."""
    plugin_config = plugin_config or {}
    for ep in importlib.metadata.entry_points(group=group):
        cfg = plugin_config.get(ep.name, {})
        enabled = cfg.get("enabled", True)
        yield ep, cfg, enabled


def discover_plugins(plugin_config: dict | None = None) -> list[PluginInfo]:
    """Scan entry points for all plugin groups."""
    plugins = []
    for group in PLUGIN_GROUPS:
        for ep, _cfg, enabled in _iter_plugins(group, plugin_config):
            plugins.append(PluginInfo.from_entry_point(ep, group, enabled=enabled))
    return plugins


def load_tool_plugins(plugin_config: dict | None = None) -> list[PluginInfo]:
    """Discover and register tool plugins.

    Each entry point should resolve to a callable that returns a list of tool functions.
    """
    from tsugite.tools import _register_tool

    results = []
    for ep, cfg, enabled in _iter_plugins(GROUP_TOOLS, plugin_config):
        if not enabled:
            results.append(PluginInfo.from_entry_point(ep, GROUP_TOOLS, enabled=False))
            logger.debug("Plugin '%s' disabled, skipping", ep.name)
            continue
        try:
            register_fn = ep.load()
            tools = register_fn(cfg)
            for func in tools:
                _register_tool(func)
            results.append(PluginInfo.from_entry_point(ep, GROUP_TOOLS, loaded=True))
            logger.info("Loaded tool plugin '%s' (%d tools)", ep.name, len(tools))
        except Exception as e:
            logger.warning("Failed to load tool plugin '%s': %s", ep.name, e)
            results.append(PluginInfo.from_entry_point(ep, GROUP_TOOLS, error=str(e)))
    return results


def load_adapter_plugins(
    plugin_config, session_store, identity_map, agents_config
) -> list[tuple[PluginInfo, object | None]]:
    """Discover and instantiate adapter plugins.

    Each entry point should resolve to a factory callable that accepts
    (config, agents_config, session_store, identity_map) and returns a BaseAdapter instance.
    """
    results = []
    for ep, cfg, enabled in _iter_plugins(GROUP_ADAPTERS, plugin_config):
        if not enabled:
            results.append((PluginInfo.from_entry_point(ep, GROUP_ADAPTERS, enabled=False), None))
            logger.debug("Adapter plugin '%s' disabled, skipping", ep.name)
            continue
        try:
            factory = ep.load()
            adapter = factory(
                config=cfg,
                agents_config=agents_config,
                session_store=session_store,
                identity_map=identity_map,
            )
            results.append((PluginInfo.from_entry_point(ep, GROUP_ADAPTERS, loaded=True), adapter))
            logger.info("Loaded adapter plugin '%s'", ep.name)
        except Exception as e:
            logger.warning("Failed to load adapter plugin '%s': %s", ep.name, e)
            results.append((PluginInfo.from_entry_point(ep, GROUP_ADAPTERS, error=str(e)), None))
    return results
