"""Resolve the configured executor backend (default: subprocess).

Built-in names resolve directly; anything else is looked up in the
`tsugite.executors` entry-point group, so remote backends (ssh, k8s, docker, ...)
can ship as plugins implementing the `Executor` protocol.
"""

import os
from typing import Optional

from tsugite.plugins import load_backend_entry_point

GROUP = "tsugite.executors"


def _configured_backend() -> str:
    try:
        from tsugite.config import load_config

        cfg = load_config().executor
        return (cfg.backend if cfg else None) or "subprocess"
    except Exception:
        return "subprocess"


def get_executor_class(backend: Optional[str] = None):
    """Return the configured Executor class (default 'subprocess')."""
    name = backend or os.environ.get("TSUGITE_EXECUTOR_BACKEND") or _configured_backend()

    if name == "local":
        from tsugite.core.executor import LocalExecutor

        return LocalExecutor
    if name == "subprocess":
        from tsugite.core.subprocess_executor import SubprocessExecutor

        return SubprocessExecutor

    cls = load_backend_entry_point(GROUP, name)
    if cls is None:
        raise ValueError(f"Unknown executor backend: {name}")
    return cls
