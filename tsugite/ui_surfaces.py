"""Registry for web UI pages and authenticated HTTP routes a plugin contributes.

Plugins register at import time from a `tsugite.plugins` module; the daemon reads
this at startup and mounts what it finds.

Nothing here may import starlette or tsugite_daemon: a `tsugite.plugins` module is
imported by every process that lists tools. That is why a page callable returns HTML
rather than being a request handler.
"""

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional, Union

# (plugin_name, descriptor). The name is empty until the plugin loader stamps it.
_surfaces: list[tuple[str, dict]] = []
_routes: list[tuple[str, dict]] = []


def register_ui_surface(
    *,
    kind: str,
    label: Optional[str] = None,
    icon: Optional[str] = None,
    entry: Optional[str] = None,
    page: Optional[Callable[[], Union[str, Awaitable[str]]]] = None,
    assets: Optional[Path] = None,
    nav: Optional[bool] = None,
    mode: Optional[str] = None,
    params: Optional[list[str]] = None,
    events: Optional[list[str]] = None,
) -> None:
    """Register a UI surface. Use `ui_surface` instead when a page function serves it.

    Args:
        kind: Surface id, namespaced to `plugin/<plugin>/<kind>` by the daemon.
        label: Tab and nav-rail label. Defaults to `kind`.
        icon: Host icon name.
        entry: Page to frame, served under the plugin's own prefix.
        page: Returns the page's HTML, instead of `entry`.
        assets: Directory served at `/api/plugins/<plugin>/ui/`.
        nav: Adds a nav-rail row.
        mode: `full` takes the workspace region, `workspace` docks as a tab.
        params: Tab params forwarded into the frame's query string.
        events: Daemon event types forwarded into the frame.
    """
    declared = {
        "kind": kind,
        "label": label,
        "icon": icon,
        "entry": entry,
        "page": page,
        "assets": assets,
        "nav": nav,
        "mode": mode,
        "params": params,
        "events": events,
    }
    _surfaces.append(("", {key: value for key, value in declared.items() if value is not None}))


def ui_surface(**kwargs):
    """Decorate a function returning the page's HTML, and register it as a surface.

    Takes the same keywords as `register_ui_surface`, minus `page`.

    The page is served unauthenticated, so return an HTML shell and fetch anything
    private from the plugin's authenticated routes.
    """

    def decorator(fn):
        register_ui_surface(page=fn, **kwargs)
        return fn

    return decorator


def http_route(path: str, methods: Optional[list[str]] = None):
    """Decorate `handler(request)`, sync or async, as an authenticated route at
    `/api/plugins/<plugin>/<path>`. The plugin's surface token can call it.

    A returned response is sent as is, and anything else is sent as JSON.
    """

    def decorator(fn):
        _routes.append(("", {"path": "/" + path.lstrip("/"), "methods": methods or ["GET"], "handler": fn}))
        return fn

    return decorator


def _grouped(registry: list[tuple[str, dict]]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for name, descriptor in registry:
        grouped.setdefault(name, []).append(descriptor)
    return grouped


def registered_ui_surfaces() -> dict[str, list[dict]]:
    """Registered surfaces by plugin name, with an empty name for anything
    registered outside a plugin load."""
    return _grouped(_surfaces)


def registered_http_routes() -> dict[str, list[dict]]:
    return _grouped(_routes)


def reset_ui_surfaces() -> None:
    """Drop every registration. For tests."""
    _surfaces.clear()
    _routes.clear()


@contextmanager
def stamping(registry: list[tuple[str, Any]], plugin_name: str):
    """Stamp `plugin_name` on the (name, descriptor) entries the block appends to `registry`.
    A block that raises drops its entries."""
    start = len(registry)
    try:
        yield
    except Exception:
        del registry[start:]
        raise
    registry[start:] = [(plugin_name, descriptor) for _name, descriptor in registry[start:]]


@contextmanager
def attributing_to(plugin_name: str):
    """Attribute surfaces and routes registered inside the block to `plugin_name`.

    The plugin loader wraps each plugin load in this, because a registration does not know
    its own entry-point name.
    """
    with stamping(_surfaces, plugin_name), stamping(_routes, plugin_name):
        yield
