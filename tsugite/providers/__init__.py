"""Provider registry with plugin support for LLM backends."""

from __future__ import annotations

import importlib
import importlib.metadata
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import Provider

from tsugite.plugins import GROUP_PROVIDERS

logger = logging.getLogger(__name__)

# Built-in provider module paths — same structure as external plugins
_BUILTIN_PROVIDERS: dict[str, str] = {
    "openai": "tsugite.providers.openai_compat",
    "ollama": "tsugite.providers.ollama",
    "openrouter": "tsugite.providers.openrouter",
    "together": "tsugite.providers.openai_compat",
    "mistral": "tsugite.providers.openai_compat",
    "github_copilot": "tsugite.providers.openai_compat",
    "anthropic": "tsugite.providers.anthropic",
    "claude_code": "tsugite.providers.claude_code",
}

_cache: dict[str, Provider] = {}


def list_all_providers() -> list[str]:
    """All known provider names (built-in + plugins)."""
    names = set(_BUILTIN_PROVIDERS.keys())
    try:
        for ep in importlib.metadata.entry_points(group=GROUP_PROVIDERS):
            names.add(ep.name)
    except Exception:
        pass
    return sorted(names)


def clear_cache() -> None:
    """Clear the provider cache. Useful for tests."""
    _cache.clear()


def get_provider(name: str) -> Provider:
    """Get a provider instance by name.

    Resolution order:
    1. Cache
    2. External plugins (tsugite.providers entry points)
    3. Built-in providers
    4. Fallback to OpenAI-compatible
    """
    if name in _cache:
        return _cache[name]

    provider = _load_plugin_provider(name) or _load_builtin_provider(name)
    if provider is None:
        # Fallback: treat unknown providers as OpenAI-compatible
        provider = _load_builtin_provider(name, fallback=True)

    _cache[name] = provider
    return provider


def _load_plugin_provider(name: str) -> Provider | None:
    """Try loading a provider from entry points."""
    try:
        for ep in importlib.metadata.entry_points(group=GROUP_PROVIDERS):
            if ep.name == name:
                factory = ep.load()
                provider = factory(name=name)
                logger.info("Loaded provider plugin '%s'", name)
                return provider
    except Exception as e:
        logger.warning("Failed to load provider plugin '%s': %s", name, e)
    return None


def _load_builtin_provider(name: str, fallback: bool = False) -> Provider | None:
    """Load a built-in provider by name."""
    module_path = _BUILTIN_PROVIDERS.get(name)
    if module_path is None and not fallback:
        return None

    if module_path is None:
        module_path = "tsugite.providers.openai_compat"

    try:
        module = importlib.import_module(module_path)
        return module.create_provider(name=name)
    except Exception as e:
        logger.error("Failed to load built-in provider '%s': %s", name, e)
        raise
