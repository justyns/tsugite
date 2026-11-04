"""Registry for plugins that turn a link in chat into a preview.

The web UI tests links against the same hosts and path regex, so the regex must mean the
same in Python and JavaScript.

Nothing here may import starlette or tsugite_daemon, for the same reason as
`tsugite.ui_surfaces`.
"""

import logging
import re
from typing import Any, Awaitable, Callable, Optional, Union
from urllib.parse import urlsplit

from tsugite.ui_surfaces import stamping

logger = logging.getLogger(__name__)

Resolve = Callable[[str], Union[Optional[dict], Awaitable[Optional[dict]]]]

# The plugin name is empty until the plugin loader stamps it.
_resolvers: list[tuple[str, dict]] = []


def register_link_resolver(*, name: str, hosts: list[str], path: Optional[str] = None, resolve: Resolve) -> None:
    """Register a resolver for links on `hosts` whose path `path` matches (re.search)."""
    if path is not None:
        re.compile(path)
    _resolvers.append(("", {"name": name, "hosts": [host.lower() for host in hosts], "path": path, "resolve": resolve}))


def link_resolver(*, name: str, hosts: list[str], path: Optional[str] = None):
    def decorator(fn):
        register_link_resolver(name=name, hosts=hosts, path=path, resolve=fn)
        return fn

    return decorator


def registered_link_resolvers() -> list[dict[str, Any]]:
    return [{"plugin": plugin, **descriptor} for plugin, descriptor in _resolvers]


def link_resolver_patterns() -> list[dict]:
    """What the web UI needs to decide which links to ask the daemon about."""
    return [{key: value for key, value in r.items() if key != "resolve"} for r in registered_link_resolvers()]


def _matches(resolver: dict, parts) -> bool:
    return parts.hostname in resolver["hosts"] and (
        resolver["path"] is None or re.search(resolver["path"], parts.path) is not None
    )


def resolver_for(url: str) -> Optional[dict]:
    """The first registered resolver matching an http(s) `url`, or None."""
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https"):
        return None
    matches = [r for r in registered_link_resolvers() if _matches(r, parts)]
    if len(matches) > 1:
        logger.warning(
            "%s matches more than one link resolver, using %s:%s over %s",
            url,
            matches[0]["plugin"],
            matches[0]["name"],
            ", ".join(f"{r['plugin']}:{r['name']}" for r in matches[1:]),
        )
    return matches[0] if matches else None


def reset_link_resolvers() -> None:
    """Drop every registration. For tests."""
    _resolvers.clear()


def attributing_to(plugin_name: str):
    """Attribute resolvers registered inside the block to `plugin_name`."""
    return stamping(_resolvers, plugin_name)
