"""Cached model discovery using KVStore with TTL."""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import asdict
from typing import Any

from .base import ModelInfo

logger = logging.getLogger(__name__)

CACHE_NAMESPACE = "model_cache"
CACHE_TTL = 86400  # 24 hours


def _get_store():
    from tsugite.kvstore.sqlite import SqliteKVBackend

    return SqliteKVBackend()


def _read_cache(provider_name: str) -> list[dict] | None:
    try:
        store = _get_store()
        raw = store.get(CACHE_NAMESPACE, provider_name)
        if raw:
            return json.loads(raw)
    except Exception as e:
        logger.debug("Cache read failed for %s: %s", provider_name, e)
    return None


def _write_cache(provider_name: str, models: list[dict]) -> None:
    try:
        store = _get_store()
        store.set(CACHE_NAMESPACE, provider_name, json.dumps(models), ttl_seconds=CACHE_TTL)
    except Exception as e:
        logger.debug("Cache write failed for %s: %s", provider_name, e)


def clear_model_cache(provider_name: str | None = None) -> None:
    """Clear cached model lists. If provider_name is None, clear all."""
    try:
        store = _get_store()
        if provider_name:
            store.delete(CACHE_NAMESPACE, provider_name)
        else:
            for key in store.list_keys(CACHE_NAMESPACE):
                store.delete(CACHE_NAMESPACE, key)
    except Exception as e:
        logger.debug("Cache clear failed: %s", e)


def _info_to_dict(info: ModelInfo | None) -> dict | None:
    return asdict(info) if info else None


def _dict_to_info(d: dict | None) -> ModelInfo | None:
    if not d:
        return None
    return ModelInfo(**{k: v for k, v in d.items() if k in ModelInfo.__dataclass_fields__})


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

    # Cache as serializable dicts
    cache_data = [{"name": m["name"], "info": _info_to_dict(m["info"])} for m in enriched]
    _write_cache(provider_name, cache_data)

    return enriched


def get_provider_models_sync(provider_name: str, refresh: bool = False) -> list[dict[str, Any]]:
    """Sync wrapper for get_provider_models."""
    return asyncio.run(get_provider_models(provider_name, refresh=refresh))


async def get_all_models(
    providers: list[str] | None = None, refresh: bool = False
) -> dict[str, list[dict[str, Any]]]:
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
