"""Cached model discovery using a JSON file with TTL."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .base import ModelInfo

logger = logging.getLogger(__name__)

CACHE_TTL = 86400  # 24 hours


def _cache_path() -> Path:
    from tsugite.config import get_xdg_cache_path

    return get_xdg_cache_path() / "models.json"


def _read_all() -> dict:
    path = _cache_path()
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception as e:
        logger.debug("Cache read failed: %s", e)
        return {}


def _write_all(data: dict) -> None:
    try:
        path = _cache_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))
    except Exception as e:
        logger.debug("Cache write failed: %s", e)


def _read_cache(provider_name: str) -> list[dict] | None:
    data = _read_all()
    entry = data.get(provider_name)
    if entry and (time.time() - entry.get("cached_at", 0)) < CACHE_TTL:
        return entry.get("models")
    return None


def _write_cache(provider_name: str, models: list[dict]) -> None:
    data = _read_all()
    data[provider_name] = {"models": models, "cached_at": time.time()}
    _write_all(data)


def clear_model_cache(provider_name: str | None = None) -> None:
    """Clear cached model lists. If provider_name is None, clear all."""
    try:
        if provider_name:
            data = _read_all()
            data.pop(provider_name, None)
            _write_all(data)
        else:
            path = _cache_path()
            if path.exists():
                path.unlink()
    except Exception as e:
        logger.debug("Cache clear failed: %s", e)


def _info_to_dict(info: ModelInfo | None) -> dict | None:
    return asdict(info) if info else None


def _dict_to_info(d: dict | None) -> ModelInfo | None:
    if not d:
        return None
    return ModelInfo(**{k: v for k, v in d.items() if k in ModelInfo.__dataclass_fields__})  # pylint: disable=no-member


async def get_provider_models(provider_name: str, refresh: bool = False) -> list[dict[str, Any]]:
    """Get models for a provider with caching.

    Returns list of {"name": str, "info": ModelInfo | None} dicts.
    """
    if not refresh:
        cached = _read_cache(provider_name)
        if cached is not None:
            return [{"name": m["name"], "info": _dict_to_info(m.get("info"))} for m in cached]

    from . import get_provider

    try:
        provider = get_provider(provider_name)
        model_names = await provider.list_models()
    except Exception as e:
        logger.debug("Failed to list models for %s: %s", provider_name, e)
        return []

    enriched = []
    for name in model_names:
        info = provider.get_model_info(name)
        enriched.append({"name": name, "info": info})

    cache_data = [{"name": m["name"], "info": _info_to_dict(m["info"])} for m in enriched]
    _write_cache(provider_name, cache_data)

    return enriched


def get_provider_models_sync(provider_name: str, refresh: bool = False) -> list[dict[str, Any]]:
    """Sync wrapper for get_provider_models."""
    return asyncio.run(get_provider_models(provider_name, refresh=refresh))


async def get_all_models(providers: list[str] | None = None, refresh: bool = False) -> dict[str, list[dict[str, Any]]]:
    """Get models from all (or specified) providers.

    Returns {provider_name: [{"name": str, "info": ModelInfo | None}, ...]}.
    """
    if providers is None:
        from . import list_all_providers

        providers = list_all_providers()

    results = await asyncio.gather(
        *[get_provider_models(p, refresh=refresh) for p in providers],
        return_exceptions=True,
    )

    out = {}
    for provider_name, result in zip(providers, results):
        if isinstance(result, Exception):
            logger.debug("Failed to get models for %s: %s", provider_name, result)
            out[provider_name] = []
        else:
            out[provider_name] = result

    return out
