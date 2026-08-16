"""Plugin discovery and loading via Python entry points."""

import importlib.metadata
import importlib.util
import inspect
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

from tsugite.events.bus import Subscription
from tsugite.ui_surfaces import attributing_to

logger = logging.getLogger(__name__)

GROUP_PLUGINS = "tsugite.plugins"
GROUP_ADAPTERS = "tsugite.adapters"
GROUP_PROVIDERS = "tsugite.providers"
GROUP_SECRETS = "tsugite.secrets"
GROUP_HISTORY = "tsugite.history"
GROUP_ATTACHMENTS = "tsugite.attachments"
GROUP_SANDBOX = "tsugite.sandbox"
GROUP_EXECUTORS = "tsugite.executors"
GROUP_COMMANDS = "tsugite.commands"
GROUP_CONTEXT_PROVIDERS = "tsugite.context_providers"
PLUGIN_GROUPS = (
    GROUP_PLUGINS,
    GROUP_ADAPTERS,
    GROUP_PROVIDERS,
    GROUP_SECRETS,
    GROUP_HISTORY,
    GROUP_ATTACHMENTS,
    GROUP_SANDBOX,
    GROUP_EXECUTORS,
    GROUP_COMMANDS,
    GROUP_CONTEXT_PROVIDERS,
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


@dataclass(frozen=True)
class LocalPluginEntryPoint:
    """A configured .py file standing in for an entry point. Importing it registers,
    so the loaders treat it as module-only."""

    name: str
    value: str

    def load(self):
        spec = importlib.util.spec_from_file_location(f"tsugite_local_plugins.{self.name}", self.value)
        if spec is None or spec.loader is None:
            raise ValueError(f"Not an importable Python file: {self.value}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module


def _resolve_plugin_config(plugin_config: dict | None) -> dict:
    """Merge a caller's plugin map over the core config's, so every path sees the
    same plugins. The daemon passes its own daemon.yaml map for adapters."""
    from tsugite.config import load_config

    return {**load_config().plugins, **(plugin_config or {})}


def _local_entry_points(group: str, plugin_config: dict, installed: set[str]):
    """Yield a stand-in entry point per configured local plugin file. Only the
    decorator group takes them, since a local file registers by being imported."""
    if group != GROUP_PLUGINS:
        return
    from tsugite.config import get_config_path

    # Config-relative, so the same config names the same file from any cwd.
    config_dir = get_config_path().parent
    for name, cfg in plugin_config.items():
        if not cfg.get("path"):
            continue
        if name in installed:
            logger.warning("Local plugin '%s' has the name of an installed plugin, skipping the local file", name)
            continue
        yield LocalPluginEntryPoint(name, str(config_dir / Path(cfg["path"]).expanduser()))


def _iter_plugins(group: str, plugin_config: dict):
    """Yield (entry_point, config_dict, enabled) for each plugin in a group."""
    installed = list(importlib.metadata.entry_points(group=group))
    names = {ep.name for ep in installed}
    for ep in [*installed, *_local_entry_points(group, plugin_config, names)]:
        cfg = plugin_config.get(ep.name, {})
        yield ep, cfg, cfg.get("enabled", True)


def local_plugin_files() -> list[LocalPluginEntryPoint]:
    """The enabled single-file plugins the next start would import."""
    plugin_config = _resolve_plugin_config(None)
    return [
        ep
        for ep, _cfg, enabled in _iter_plugins(GROUP_PLUGINS, plugin_config)
        if enabled and isinstance(ep, LocalPluginEntryPoint)
    ]


def check_plugin_config() -> list[str]:
    """Report what would stop the configured plugins loading on a fresh start.

    The JSON is re-read here because load_config() swallows a parse error and
    returns defaults, which drops every plugin without saying so.
    """
    from tsugite.config import get_config_path

    config_path = get_config_path()
    if config_path.exists():
        try:
            json.loads(config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            return [f"{config_path} is not valid JSON: {e}"]

    problems = []
    for ep in local_plugin_files():
        path = Path(ep.value)
        if not path.is_file():
            problems.append(f"Plugin '{ep.name}': no such file: {path}")
            continue
        try:
            compile(path.read_text(encoding="utf-8"), str(path), "exec")
        except SyntaxError as e:
            problems.append(f"Plugin '{ep.name}': syntax error at {path}:{e.lineno}: {e.msg}")
    return problems


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
    plugin_config = _resolve_plugin_config(plugin_config)
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
    for ep, cfg, enabled in _iter_plugins(group, _resolve_plugin_config(plugin_config)):
        if not enabled:
            results.append(PluginInfo.from_entry_point(ep, group, enabled=False))
            logger.debug("Plugin '%s' (%s) disabled, skipping", ep.name, group)
            continue
        try:
            with attributing_to(ep.name):
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


def get_plugin_hooks() -> dict[str, list]:
    """Return all registered plugin hooks, keyed by phase name."""
    return _plugin_hooks


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


def load_module_only_plugins(group: str, plugin_config: dict | None = None) -> list[PluginInfo]:
    """Import module-only plugins for a group whose registration is a pure import
    side effect, so the loader just imports each module and has nothing to consume.

    Shared by the decorator group (tsugite.plugins, @tool/@hook/@subscribe), the
    daemon-command group (tsugite.commands, @adapter_command), and the
    context-provider group (tsugite.context_providers, register_context_provider()).
    """
    return _load_plugin_group(group, plugin_config, on_loaded=lambda _: None)


def load_adapter_plugins(
    plugin_config, session_store, identity_map, agents_config
) -> list[tuple[PluginInfo, object | None]]:
    """Discover and instantiate adapter plugins.

    Each entry point should resolve to a factory callable that accepts
    (config, agents_config, session_store, identity_map) and returns a BaseAdapter instance.
    """
    results = []
    for ep, cfg, enabled in _iter_plugins(GROUP_ADAPTERS, _resolve_plugin_config(plugin_config)):
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
