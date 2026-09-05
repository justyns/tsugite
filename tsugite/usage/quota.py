"""Provider quota sources: a normalized read of how much of a provider's
subscription or budget the user has consumed ("week 73%, resets Aug 19").

A source is contributed by a plugin through the ``tsugite.usage_providers``
entry point (a module-only entry point whose import registers the source) and
fetches from whatever local state that provider keeps. The normalized shape
deliberately carries neither a severity nor a plan: the UI already derives its
own warn threshold, and a second source of truth would let them disagree.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class QuotaWindow:
    """One limit window inside a report. ``key`` is stable within the report
    ("session", "weekly_all"); ``label`` is what the UI shows."""

    key: str
    label: str
    used_pct: float
    resets_at: str | None = None


@dataclass(frozen=True)
class QuotaReport:
    """One provider's quota. ``as_of`` is when the provider last refreshed its
    own cache; ``error`` is set instead of windows when local state was
    unreadable."""

    provider: str
    label: str
    as_of: str | None = None
    windows: tuple[QuotaWindow, ...] = ()
    error: str | None = None


FetchFn = Callable[[], QuotaReport]


@dataclass(frozen=True)
class QuotaSource:
    provider: str
    label: str
    fetch: FetchFn


_registry: dict[str, QuotaSource] = {}
_loaded = False


def register_quota_source(provider: str, label: str, fetch: FetchFn) -> None:
    """Register a source (idempotent by ``provider``; the last registration wins)."""
    if provider in _registry:
        logger.debug("Quota source '%s' re-registered", provider)
    _registry[provider] = QuotaSource(provider=provider, label=label, fetch=fetch)


def reset_quota_sources() -> None:
    """Clear the registry and the load flag (tests)."""
    global _loaded
    _registry.clear()
    _loaded = False


def ensure_loaded() -> None:
    """Import the ``tsugite.usage_providers`` entry-point modules once, so every
    ``register_quota_source`` call has run before the registry is read."""
    global _loaded
    if _loaded:
        return
    _loaded = True
    try:
        from tsugite.plugins import GROUP_USAGE_PROVIDERS, load_module_only_plugins

        load_module_only_plugins(GROUP_USAGE_PROVIDERS)
    except Exception as e:  # never let plugin discovery break a read
        logger.warning("Loading usage-provider plugins failed: %s", e)


def get_quota_sources() -> list[QuotaSource]:
    ensure_loaded()
    return list(_registry.values())


def _as_dict(report: QuotaReport) -> dict:
    return {
        "provider": report.provider,
        "label": report.label,
        "as_of": report.as_of,
        "error": report.error,
        "windows": [
            {"key": w.key, "label": w.label, "used_pct": w.used_pct, "resets_at": w.resets_at} for w in report.windows
        ],
    }


def collect_quota_reports() -> list[dict]:
    """Every source's report as JSON-ready dicts, ordered by provider.

    A source that raises becomes a row carrying its error rather than a missing
    row, so the UI can show which provider it failed to read.
    """
    rows = []
    for source in sorted(get_quota_sources(), key=lambda s: s.provider):
        try:
            report = source.fetch()
        except Exception as e:
            logger.warning("Quota source '%s' failed: %s", source.provider, e)
            report = QuotaReport(provider=source.provider, label=source.label, error=str(e))
        rows.append(_as_dict(report))
    return rows
