"""Permissions store for approval-gated actions.

Merges two YAML sources, the same way hooks does:

1. ``runtime_path``: a mutable ``permissions.yaml`` next to ``daemon.yaml``.
   "Always allow" appends to ``web.fetch_allowlist`` and "Always deny" to
   ``web.preview_denylist``, so they survive a restart.
2. ``workspace_dir``: an optional, read-only ``<workspace_dir>/.tsugite/permissions.yaml``,
   hand-authored per agent (exactly like ``.tsugite/hooks.yaml``). The store never
   writes it.

``is_listed`` is the union of both files. ``add`` only ever writes the runtime
file, atomically, preserving every unrelated section.

The file groups lists by section, e.g.::

    web:
      fetch_allowlist:
        - example.com
      fetch_denylist:
        - ads.example.com
      preview_denylist:
        - news.example.com

The generic ``is_listed(section, key)`` / ``add(section, key)`` address a list by
dotted path (``"web.fetch_allowlist"``). The ``web_fetch_*`` wrappers hide that
layout so callers stay decoupled from it. The store applies no precedence between
lists. The caller decides.
"""

import contextvars
from pathlib import Path
from typing import List, Optional

import yaml

from tsugite.events.helpers import current_session_id
from tsugite.utils import atomic_write_text

WEB_FETCH_ALLOWLIST = "web.fetch_allowlist"
WEB_FETCH_DENYLIST = "web.fetch_denylist"
WEB_PREVIEW_DENYLIST = "web.preview_denylist"


def _load_yaml_mapping(path: Optional[Path]) -> dict:
    """Load a yaml file as a mapping, treating missing/empty/non-mapping as ``{}``."""
    if path is None or not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def _get_list(data: dict, parts: List[str]) -> list:
    """Return the list at the dotted path, or ``[]`` if absent/not a list."""
    node = data
    for part in parts:
        if not isinstance(node, dict):
            return []
        node = node.get(part)
    return node if isinstance(node, list) else []


def _ensure_parent(data: dict, parts: List[str]) -> tuple:
    """Descend to the parent of the leaf, creating intermediate dicts. Returns (parent, leaf)."""
    node = data
    for part in parts[:-1]:
        child = node.get(part)
        if not isinstance(child, dict):
            child = {}
            node[part] = child
        node = child
    return node, parts[-1]


def _atomic_write_yaml(path: Path, data: dict) -> None:
    """Serialize ``data`` to yaml and write it atomically (see ``atomic_write_text``)."""
    atomic_write_text(path, yaml.safe_dump(data, default_flow_style=False, sort_keys=False))


class Permissions:
    """Union of a mutable runtime permissions file and a read-only workspace one."""

    def __init__(self, runtime_path: Path, workspace_dir: Optional[Path] = None):
        self.runtime_path = Path(runtime_path)
        self._workspace_path = (Path(workspace_dir) / ".tsugite" / "permissions.yaml") if workspace_dir else None

    def listed(self, section: str) -> list:
        """Entries at dotted ``section`` across both sources."""
        parts = section.split(".")
        return _get_list(_load_yaml_mapping(self.runtime_path), parts) + _get_list(
            _load_yaml_mapping(self._workspace_path), parts
        )

    def is_listed(self, section: str, key: str) -> bool:
        """True if ``key`` is listed at dotted ``section`` in EITHER source."""
        return key in self.listed(section)

    def add(self, section: str, key: str) -> None:
        """Append ``key`` to the RUNTIME file's list at dotted ``section``.

        Only ever touches the runtime file (never the workspace file), writes
        atomically, and preserves every unrelated section. A no-op if already
        present.
        """
        data = _load_yaml_mapping(self.runtime_path)
        parent, leaf = _ensure_parent(data, section.split("."))
        values = parent.get(leaf)
        if not isinstance(values, list):
            values = []
            parent[leaf] = values
        if key not in values:
            values.append(key)
            _atomic_write_yaml(self.runtime_path, data)

    def web_fetch_allowed(self, domain: str) -> bool:
        """True if ``domain`` is on the web fetch allowlist (either source)."""
        return self.is_listed(WEB_FETCH_ALLOWLIST, domain.lower())

    def web_fetch_allow(self, domain: str) -> None:
        """Persist ``domain`` to the runtime web fetch allowlist."""
        self.add(WEB_FETCH_ALLOWLIST, domain.lower())

    def web_fetch_denied(self, domain: str) -> bool:
        """True if ``domain`` is on the web fetch denylist (either source)."""
        return self.is_listed(WEB_FETCH_DENYLIST, domain.lower())

    def web_fetch_deny(self, domain: str) -> None:
        """Persist ``domain`` to the runtime web fetch denylist."""
        self.add(WEB_FETCH_DENYLIST, domain.lower())

    def web_preview_denied(self, domain: str) -> bool:
        """True if link previews of ``domain`` are suppressed (either source). Agent fetches ignore this list."""
        return self.is_listed(WEB_PREVIEW_DENYLIST, domain.lower())

    def web_preview_deny(self, domain: str) -> None:
        """Persist ``domain`` to the runtime preview denylist."""
        self.add(WEB_PREVIEW_DENYLIST, domain.lower())


_session_denied_hosts: dict[Optional[str], set[str]] = {}
_session_preview_denied_hosts: dict[Optional[str], set[str]] = {}


def deny_web_fetch_for_session(host: str) -> None:
    """Block the agent's fetches of ``host`` for the current session only. Never written to ``permissions.yaml``."""
    _session_denied_hosts.setdefault(current_session_id(), set()).add(host.lower())


def allow_web_for_session(host: str) -> None:
    """Lift the session fetch block and preview deny for ``host``."""
    sid = current_session_id()
    _session_denied_hosts.get(sid, set()).discard(host.lower())
    _session_preview_denied_hosts.get(sid, set()).discard(host.lower())


def deny_web_preview_for_session(host: str) -> None:
    """Suppress link previews of ``host`` for the current session only."""
    _session_preview_denied_hosts.setdefault(current_session_id(), set()).add(host.lower())


def web_preview_denied_in_session(host: str) -> bool:
    return host.lower() in _session_preview_denied_hosts.get(current_session_id(), set())


def session_blocked_hosts() -> set[str]:
    return _session_denied_hosts.get(current_session_id(), set())


def web_fetch_blocked(host: str, perms: Optional[Permissions]) -> bool:
    """True if ``host`` is denied for the current session or on ``perms``' denylist."""
    return host.lower() in session_blocked_hosts() or bool(perms and perms.web_fetch_denied(host))


def web_fetch_blocked_hosts(perms: Optional[Permissions]) -> list[str]:
    """Every host ``web_fetch_blocked`` refuses, for seeding a subprocess executor's child."""
    persisted = perms.listed(WEB_FETCH_DENYLIST) if perms else []
    return sorted(session_blocked_hosts() | {h for h in persisted if isinstance(h, str)})


_permissions_var: contextvars.ContextVar[Optional[Permissions]] = contextvars.ContextVar("permissions", default=None)


def set_permissions(permissions: Optional[Permissions]) -> None:
    """Set the permissions store for the current context."""
    _permissions_var.set(permissions)


def get_permissions() -> Optional[Permissions]:
    """Get the permissions store from the current context."""
    return _permissions_var.get()
