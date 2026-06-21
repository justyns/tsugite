"""Resolve the configured history backend (default: jsonl).

Mirrors the secrets backend pattern: built-in names resolve directly, anything
else is looked up in the `tsugite.history` entry-point group.
"""

import logging
import os

from tsugite.config import load_config
from tsugite.plugins import load_backend_entry_point

from .base import HistoryBackend
from .storage import JsonlHistoryBackend

logger = logging.getLogger(__name__)

GROUP = "tsugite.history"

_backend: HistoryBackend | None = None


def _create_backend() -> HistoryBackend:
    cfg = load_config().history
    config = cfg.model_dump() if cfg else {}
    name = os.environ.get("TSUGITE_HISTORY_BACKEND") or config.get("backend", "jsonl")

    if name == "jsonl":
        return JsonlHistoryBackend()

    factory = load_backend_entry_point(GROUP, name)
    if factory is None:
        raise ValueError(f"Unknown history backend: {name}")
    backend = factory(config)
    logger.info("Loaded history backend '%s'", name)
    return backend


def get_history_backend() -> HistoryBackend:
    """Return the process-wide history backend, creating it on first use."""
    global _backend
    if _backend is None:
        _backend = _create_backend()
    return _backend


def set_history_backend(backend: HistoryBackend | None) -> None:
    """Override the backend (used by tests and embedders)."""
    global _backend
    _backend = backend


def reset_history_backend() -> None:
    """Clear the cached backend so the next call re-reads config."""
    global _backend
    _backend = None
