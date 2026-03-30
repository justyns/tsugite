"""Secret masking registry.

Tracks all resolved secret values and masks them in output.
"""

import logging
import threading

logger = logging.getLogger(__name__)


class SecretRegistry:
    """Tracks active secrets for output masking."""

    def __init__(self):
        self._lock = threading.Lock()
        self._active: dict[str, str] = {}  # name → raw value
        self._sorted: list[tuple[str, str]] = []  # cached length-descending order
        self._filter_installed = False

    def _ensure_log_filter(self):
        if not self._filter_installed:
            from tsugite.secrets.masking import install_masking_filter

            install_masking_filter()
            self._filter_installed = True

    def register(self, name: str, value: str, agent: str = "unknown") -> str:
        """Register a secret value for masking. Returns the raw value."""
        if not value:
            return value
        self._ensure_log_filter()
        with self._lock:
            self._active[name] = value
            self._sorted = sorted(self._active.items(), key=lambda x: len(x[1]), reverse=True)
        logger.info("Secret '%s' accessed by agent '%s'", name, agent)
        return value

    def mask(self, text: str) -> str:
        """Replace all known secret values with '***'."""
        if not text:
            return text
        # _sorted is replaced atomically, safe to read without lock
        snapshot = self._sorted
        if not snapshot:
            return text
        for _name, value in snapshot:
            if value in text:
                text = text.replace(value, "***")
        return text

    def clear(self):
        with self._lock:
            self._active.clear()
            self._sorted.clear()


_registry = SecretRegistry()


def get_registry() -> SecretRegistry:
    return _registry
