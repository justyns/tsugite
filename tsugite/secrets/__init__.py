"""Pluggable secret store for Tsugite."""

import importlib.metadata
import logging
import os
from typing import Optional

from .backend import SecretBackend

logger = logging.getLogger(__name__)

_backend: Optional[SecretBackend] = None


def get_backend() -> SecretBackend:
    global _backend
    if _backend is None:
        _backend = _create_backend()
    return _backend


def set_backend(backend: SecretBackend) -> None:
    global _backend
    _backend = backend


def _create_backend(config: dict | None = None) -> SecretBackend:
    if not config:
        from tsugite.config import load_config

        sc = load_config().secrets
        config = sc.model_dump() if sc else {}
    name = os.environ.get("TSUGITE_SECRETS_BACKEND") or config.get("provider", "env")

    if name == "env":
        from .env import EnvSecretBackend

        return EnvSecretBackend(prefix=config.get("prefix") or os.environ.get("TSUGITE_SECRETS_PREFIX", ""))

    if name == "file":
        from .file import FileSecretBackend

        return FileSecretBackend(config)

    if name == "sqlite":
        from .sqlite import SqliteSecretBackend

        return SqliteSecretBackend(config)

    if name == "exec":
        from .exec import ExecSecretBackend

        return ExecSecretBackend(config)

    # Plugin lookup
    for ep in importlib.metadata.entry_points(group="tsugite.secrets"):
        if ep.name == name:
            factory = ep.load()
            return factory(config)

    raise ValueError(f"Unknown secrets backend: {name}")


def configure_from_daemon(daemon_config) -> None:
    config = getattr(daemon_config, "plugins", {}).get("secrets", {})
    set_backend(_create_backend(config))


def init_cli(no_secrets: bool = False) -> None:
    """Initialize secrets for CLI commands. Call early while stdin is a TTY."""
    if no_secrets:
        from .env import EnvSecretBackend

        set_backend(EnvSecretBackend())
        return
    try:
        get_backend()
    except Exception as e:
        logger.warning("Failed to initialize secrets backend: %s", e)
        from .env import EnvSecretBackend

        set_backend(EnvSecretBackend())
