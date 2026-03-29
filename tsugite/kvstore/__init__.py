"""Pluggable KV store for Tsugite."""

import importlib.metadata
import logging
import os
from typing import Optional

from .backend import KVBackend

logger = logging.getLogger(__name__)

_backend: Optional[KVBackend] = None


def get_backend() -> KVBackend:
    global _backend
    if _backend is None:
        _backend = _create_backend()
    return _backend


def set_backend(backend: KVBackend) -> None:
    global _backend
    _backend = backend


def _create_backend(config: dict | None = None) -> KVBackend:
    config = config or {}
    name = os.environ.get("TSUGITE_KV_BACKEND") or config.get("backend", "sqlite")

    if name == "sqlite":
        from .sqlite import SqliteKVBackend

        return SqliteKVBackend()

    for ep in importlib.metadata.entry_points(group="tsugite.kvstore"):
        if ep.name == name:
            factory = ep.load()
            return factory(config)

    raise ValueError(f"Unknown KV backend: {name}")


def configure_from_daemon(daemon_config) -> None:
    config = getattr(daemon_config, "plugins", {}).get("kv", {})
    set_backend(_create_backend(config))
