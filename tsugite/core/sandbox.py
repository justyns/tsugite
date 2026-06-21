"""Sandbox seam: policy types + backend resolution.

The sandbox *mechanism* (bubblewrap command building) lives in the `tsugite-sandbox`
plugin, registered under the `tsugite.sandbox` entry-point group. This module keeps the
policy type (`SandboxConfig`) and the "is-sandboxed" decision in core, and resolves the
configured backend, defaulting to `bwrap`. The egress proxy (`core/proxy.py`) is generic
and also stays in core; the executor orchestrates it.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Protocol, runtime_checkable

from tsugite.plugins import load_backend_entry_point

GROUP = "tsugite.sandbox"


@dataclass
class SandboxConfig:
    """Configuration for sandbox execution."""

    allowed_domains: List[str] = field(default_factory=list)
    no_network: bool = False
    extra_ro_binds: List[Path] = field(default_factory=list)
    extra_rw_binds: List[Path] = field(default_factory=list)
    pass_env: List[str] = field(default_factory=list)  # extra env var NAMES to pass through


@runtime_checkable
class Sandbox(Protocol):
    """Wraps a command for isolated execution.

    Backends are constructed with (config, proxy_socket=None, workspace_dir=None,
    state_dir=None) and expose a classmethod/staticmethod `check_available() -> bool`.
    """

    def build_command(self, inner_cmd: List[str]) -> List[str]: ...


def _configured_backend() -> str:
    try:
        from tsugite.config import load_config

        cfg = load_config().sandbox
        return (cfg.backend if cfg else None) or "bwrap"
    except Exception:
        return "bwrap"


def get_sandbox_class(backend: Optional[str] = None):
    """Resolve the configured sandbox backend class, or None if it isn't installed."""
    name = backend or os.environ.get("TSUGITE_SANDBOX_BACKEND") or _configured_backend()
    return load_backend_entry_point(GROUP, name)


def sandbox_available(backend: Optional[str] = None) -> bool:
    """True if a sandbox backend is installed and usable on this host."""
    cls = get_sandbox_class(backend)
    if cls is None:
        return False
    check = getattr(cls, "check_available", None)
    return check() if check else True
