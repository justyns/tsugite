"""Plugin discovery and loading via Python entry points."""

import importlib.metadata
import inspect
import logging
from dataclasses import dataclass

from tsugite.events.bus import Subscription

logger = logging.getLogger(__name__)

GROUP_PLUGINS = "tsugite.plugins"
GROUP_TOOLS = "tsugite.tools"
GROUP_ADAPTERS = "tsugite.adapters"
GROUP_PROVIDERS = "tsugite.providers"
GROUP_SECRETS = "tsugite.secrets"
GROUP_HOOKS = "tsugite.hooks"
GROUP_EVENT_SUBSCRIBERS = "tsugite.event_subscribers"
GROUP_HISTORY = "tsugite.history"
GROUP_ATTACHMENTS = "tsugite.attachments"
GROUP_SANDBOX = "tsugite.sandbox"
GROUP_EXECUTORS = "tsugite.executors"
PLUGIN_GROUPS = (
    GROUP_PLUGINS,
    GROUP_TOOLS,
    GROUP_ADAPTERS,
    GROUP_PROVIDERS,
    GROUP_SECRETS,
    GROUP_HOOKS,
    GROUP_EVENT_SUBSCRIBERS,
    GROUP_HISTORY,
    GROUP_ATTACHMENTS,
    GROUP_SANDBOX,
    GROUP_EXECUTORS,
)

_plugin_hooks: dict[str, list] = {}
_plugin_subscriptions: list[Subscription] = []
_plugin_attachment_handlers: list | None = None


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


def load_backend_entry_point(group: str, name: str):
    """Return the object registered under `name` in entry-point `group`, or None.

    Shared by the backend resolvers (executor, history, sandbox, ...) to look up a
    swappable battery by name. Built-in backends are handled by the caller before
    falling through to this plugin lookup.
    """
    for ep in importlib.metadata.entry_points(group=group):
        if ep.name == name:
            return ep.load()
    return None


def discover_plugins(plugin_config: dict | None = None) -> list[PluginInfo]:
    """Scan entry points for all plugin groups."""
    plugins = []
    for group in PLUGIN_GROUPS:
        for ep, _cfg, enabled in _iter_plugins(group, plugin_config):
            plugins.append(PluginInfo.from_entry_point(ep, group, enabled=enabled))
    return plugins


def _load_plugin_group(group, plugin_config, on_loaded, summarize=None) -> list[PluginInfo]:
    """Iterate a plugin group: skip disabled, ep.load(), invoke register_fn(cfg),
    pass result to on_loaded(payload), accumulate PluginInfo, isolate errors.
    `summarize(payload) -> str` adds detail to the success log line.
    """
    results = []
    for ep, cfg, enabled in _iter_plugins(group, plugin_config):
        if not enabled:
            results.append(PluginInfo.from_entry_point(ep, group, enabled=False))
            logger.debug("Plugin '%s' (%s) disabled, skipping", ep.name, group)
            continue
        try:
            target = ep.load()
            if inspect.ismodule(target):
                # Module-only entry point: import did the registration via decorators.
                payload = None
                extra = " (module-only)"
            else:
                payload = target(cfg)
                on_loaded(payload)
                extra = f": {summarize(payload)}" if summarize else ""
            results.append(PluginInfo.from_entry_point(ep, group, loaded=True))
            logger.info("Loaded %s plugin '%s'%s", group, ep.name, extra)
        except Exception as e:
            logger.warning("Failed to load %s plugin '%s': %s", group, ep.name, e)
            results.append(PluginInfo.from_entry_point(ep, group, error=str(e)))
    return results


def load_tool_plugins(plugin_config: dict | None = None) -> list[PluginInfo]:
    """Discover and register tool plugins.

    Each entry point should resolve to a callable that returns a list of tool functions.
    """
    from tsugite.tools import _register_tool

    def consume(tools):
        for func in tools:
            _register_tool(func)

    return _load_plugin_group(GROUP_TOOLS, plugin_config, consume, summarize=lambda t: f"{len(t)} tools")


def load_hook_plugins(plugin_config: dict | None = None) -> list[PluginInfo]:
    """Discover and register hook plugins.

    Each entry point should resolve to a callable that returns
    dict[str, list[HookRule]] mapping phase names to hook rules.
    """

    def consume(hooks):
        for phase, rules in hooks.items():
            _plugin_hooks.setdefault(phase, []).extend(rules)

    return _load_plugin_group(
        GROUP_HOOKS,
        plugin_config,
        consume,
        summarize=lambda h: ", ".join(f"{phase}({len(rules)})" for phase, rules in h.items()),
    )


def get_plugin_hooks() -> dict[str, list]:
    """Return all registered plugin hooks, keyed by phase name."""
    return _plugin_hooks


def load_event_subscriber_plugins(plugin_config: dict | None = None) -> list[PluginInfo]:
    """Discover and register event subscriber plugins.

    Each entry point should resolve to a callable that returns
    list[Subscription]. Subscriptions are picked up by every EventBus
    instance constructed afterwards.
    """
    return _load_plugin_group(
        GROUP_EVENT_SUBSCRIBERS,
        plugin_config,
        _plugin_subscriptions.extend,
        summarize=lambda subs: f"{len(subs)} subscriptions",
    )


def get_plugin_subscriptions() -> list[Subscription]:
    """Return all registered plugin event subscriptions."""
    return _plugin_subscriptions


def load_attachment_plugins(plugin_config: dict | None = None) -> list[PluginInfo]:
    """Discover attachment handler plugins.

    Each entry point resolves to a factory callable that accepts a config dict and
    returns an AttachmentHandler instance.
    """
    global _plugin_attachment_handlers
    handlers: list = []
    results = _load_plugin_group(
        GROUP_ATTACHMENTS,
        plugin_config,
        handlers.append,
        summarize=lambda h: type(h).__name__,
    )
    _plugin_attachment_handlers = handlers
    return results


def get_attachment_handlers() -> list:
    """Return plugin-contributed attachment handlers (loaded once, cached)."""
    if _plugin_attachment_handlers is None:
        load_attachment_plugins()
    return _plugin_attachment_handlers


def reset_attachment_handlers() -> None:
    """Clear the cached plugin attachment handlers (used by tests)."""
    global _plugin_attachment_handlers
    _plugin_attachment_handlers = None


def load_decorator_plugins(plugin_config: dict | None = None) -> list[PluginInfo]:
    """Discover and import unified-group plugins.

    Plugins under tsugite.plugins are expected to be module-only entry points
    whose import triggers @tool / @hook / @subscribe decorators. The loader
    has nothing to do beyond importing the module - registration happens via
    the decorators' side effects.
    """
    return _load_plugin_group(GROUP_PLUGINS, plugin_config, on_loaded=lambda _: None)


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
