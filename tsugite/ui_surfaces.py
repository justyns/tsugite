"""Registry for web UI pages a plugin contributes.

Plugins register at import time from a `tsugite.plugins` module; the daemon reads
this at startup and mounts what it finds.

Nothing here may import starlette or tsugite_daemon: a `tsugite.plugins` module is
imported by every process that lists tools. That is why a page callable returns HTML
rather than being a request handler.
"""

from contextlib import contextmanager
from pathlib import Path
from typing import Awaitable, Callable, Optional, Union

# (plugin_name, descriptor). The name is empty until the plugin loader stamps it.
_surfaces: list[tuple[str, dict]] = []


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


def registered_ui_surfaces() -> dict[str, list[dict]]:
    """Registered surfaces by plugin name, with an empty name for anything
    registered outside a plugin load."""
    grouped: dict[str, list[dict]] = {}
    for name, descriptor in _surfaces:
        grouped.setdefault(name, []).append(descriptor)
    return grouped


def reset_ui_surfaces() -> None:
    """Drop every registration. For tests."""
    _surfaces.clear()


@contextmanager
def attributing_to(plugin_name: str):
    """Attribute surfaces registered inside the block to `plugin_name`.

    A registration running at import time cannot know its own entry-point name, so
    the plugin loader wraps each import in this. An import that raises leaves nothing
    behind.
    """
    start = len(_surfaces)
    try:
        yield
    except Exception:
        del _surfaces[start:]
        raise
    _surfaces[start:] = [(plugin_name, descriptor) for _name, descriptor in _surfaces[start:]]
