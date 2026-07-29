"""Permissions store for approval-gated actions.

Merges two YAML sources, the same way hooks does:

1. ``runtime_path``: a mutable ``permissions.yaml`` next to ``daemon.yaml``.
   "Always allow" decisions append here, so they survive a restart.
2. ``workspace_dir``: an optional, read-only ``<workspace_dir>/.tsugite/permissions.yaml``,
   hand-authored per agent (exactly like ``.tsugite/hooks.yaml``). The store never
   writes it.

``is_allowed`` is the union of both files. ``allow`` only ever writes the runtime
file, atomically, preserving every unrelated section.

The file groups allowlists by section, e.g.::

    web:
      fetch_allowlist:
        - example.com

The generic ``is_allowed(section, key)`` / ``allow(section, key)`` address a list
by dotted path (``"web.fetch_allowlist"``). The ``web_fetch_*`` wrappers hide that
layout so callers stay decoupled from it.
"""

import contextvars
from pathlib import Path
from typing import List, Optional

import yaml

from tsugite.utils import atomic_write_text

WEB_FETCH_ALLOWLIST = "web.fetch_allowlist"


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
    """Union of a mutable runtime allowlist and a read-only workspace allowlist."""

    def __init__(self, runtime_path: Path, workspace_dir: Optional[Path] = None):
        self.runtime_path = Path(runtime_path)
        self._workspace_path = (Path(workspace_dir) / ".tsugite" / "permissions.yaml") if workspace_dir else None

    def is_allowed(self, section: str, key: str) -> bool:
        """True if ``key`` is listed at dotted ``section`` in EITHER source."""
        parts = section.split(".")
        return key in _get_list(_load_yaml_mapping(self.runtime_path), parts) or key in _get_list(
            _load_yaml_mapping(self._workspace_path), parts
        )

    def allow(self, section: str, key: str) -> None:
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
        return self.is_allowed(WEB_FETCH_ALLOWLIST, domain.lower())

    def web_fetch_allow(self, domain: str) -> None:
        """Persist ``domain`` to the runtime web fetch allowlist."""
        self.allow(WEB_FETCH_ALLOWLIST, domain.lower())


_permissions_var: contextvars.ContextVar[Optional[Permissions]] = contextvars.ContextVar("permissions", default=None)


def set_permissions(permissions: Optional[Permissions]) -> None:
    """Set the permissions store for the current context."""
    _permissions_var.set(permissions)


def get_permissions() -> Optional[Permissions]:
    """Get the permissions store from the current context."""
    return _permissions_var.get()
