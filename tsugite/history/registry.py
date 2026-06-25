"""Resolve the configured history backend (default: sqlite).

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


def _warn_unmigrated_jsonl(backend: HistoryBackend) -> None:
    """Heads-up (once per process) if legacy JSONL sessions exist but aren't imported yet.

    The default flipped to sqlite, which never reads the old per-session ``*.jsonl`` files.
    Without this, an upgraded user's history silently reads empty until they run the import.
    """
    from .storage import list_session_files

    try:
        files = list_session_files()
    except Exception:
        return
    # Quiet if there's nothing to migrate, or if any legacy file is already in the db.
    if not files or any(backend.exists(f.stem) for f in files[:50]):
        return
    logger.warning(
        "%d legacy JSONL conversation file(s) are not in the sqlite history database; they won't "
        "appear in `tsugite history list` or the web UI until you run `tsugite history import` "
        "(originals are left untouched).",
        len(files),
    )


def _create_backend() -> HistoryBackend:
    cfg = load_config().history
    config = cfg.model_dump() if cfg else {}
    name = os.environ.get("TSUGITE_HISTORY_BACKEND") or config.get("backend", "sqlite")

    if name == "jsonl":
        logger.warning(
            "The 'jsonl' history backend is deprecated (sqlite is the default) and retained "
            "mainly for importing legacy files; it may be removed in a future release."
        )
        return JsonlHistoryBackend()

    if name == "sqlite":
        from .sqlite_backend import SqliteHistoryBackend

        backend = SqliteHistoryBackend()
        _warn_unmigrated_jsonl(backend)
        return backend

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
