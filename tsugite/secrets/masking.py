"""Logging filter for secret masking."""

import logging

from .registry import get_registry

_installed = False


class SecretMaskingFilter(logging.Filter):
    """Masks secret values in log records."""

    def filter(self, record):
        registry = get_registry()
        if isinstance(record.msg, str):
            record.msg = registry.mask(record.msg)
        if record.args:
            if isinstance(record.args, dict):
                record.args = {k: registry.mask(v) if isinstance(v, str) else v for k, v in record.args.items()}
            else:
                record.args = tuple(registry.mask(a) if isinstance(a, str) else a for a in record.args)
        return True


def install_masking_filter():
    """Install the secret masking filter on the root logger (idempotent)."""
    global _installed
    if _installed:
        return
    logging.getLogger().addFilter(SecretMaskingFilter())
    _installed = True
