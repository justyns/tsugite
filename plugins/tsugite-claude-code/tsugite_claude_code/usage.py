"""Claude Code subscription limits as a tsugite quota source.

Read from ``limits[]`` in the usage endpoint Claude Code calls, with its OAuth
token. When that fails, the usage copy cached in Claude Code's config is used
while it is under an hour old. The list is self-describing, so a limit kind
tsugite has never seen still shows up.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

import httpx

from tsugite.usage.quota import QuotaReport, QuotaWindow, register_quota_source
from tsugite.user_agent import set_user_agent_header

logger = logging.getLogger(__name__)

PROVIDER = "claude_code"
LABEL = "Claude Code"
USAGE_URL = "https://api.anthropic.com/api/oauth/usage"
CACHE_MAX_AGE_MS = 60 * 60 * 1000
LIVE_TTL_S = 60

_last_live: tuple[float, list] | None = None


def _config_path() -> Path:
    base = os.environ.get("CLAUDE_CONFIG_DIR") or os.path.expanduser("~")
    return Path(base) / ".claude.json"


def _credentials_path() -> Path:
    base = os.environ.get("CLAUDE_CONFIG_DIR") or os.path.expanduser("~/.claude")
    return Path(base) / ".credentials.json"


def _live_limits() -> list:
    global _last_live
    if _last_live and time.monotonic() - _last_live[0] < LIVE_TTL_S:
        return _last_live[1]
    token = json.loads(_credentials_path().read_text())["claudeAiOauth"]["accessToken"]
    headers = {"Authorization": f"Bearer {token}", "anthropic-beta": "oauth-2025-04-20"}
    set_user_agent_header(headers)
    response = httpx.get(USAGE_URL, headers=headers, timeout=10)
    response.raise_for_status()
    limits = response.json()["limits"]
    _last_live = (time.monotonic(), limits)
    return limits


def _cached_limits() -> list:
    cache = json.loads(_config_path().read_text())["cachedUsageUtilization"]
    if time.time() * 1000 - cache["fetchedAtMs"] > CACHE_MAX_AGE_MS:
        raise ValueError("cached usage is over an hour old")
    return cache["utilization"]["limits"]


def _label(limit: dict) -> str:
    kind = limit["kind"]
    if kind == "weekly_all":
        return "week"
    if kind == "weekly_scoped":
        model = (limit.get("scope") or {}).get("model") or {}
        name = model.get("display_name")
        return f"week · {name}" if name else kind
    return kind


def _error(message: str) -> QuotaReport:
    return QuotaReport(provider=PROVIDER, label=LABEL, error=message)


def fetch() -> QuotaReport:
    try:
        limits = _live_limits()
    except Exception as live_error:
        try:
            limits = _cached_limits()
        except Exception as cache_error:
            logger.warning("Claude Code usage unavailable: live %r, cache %r", live_error, cache_error)
            return _error("Claude Code's usage limits could not be fetched, and no recent cached copy exists")

    try:
        windows = tuple(
            QuotaWindow(
                key=limit["kind"],
                label=_label(limit),
                used_pct=float(limit["percent"]),
                resets_at=limit.get("resets_at"),
            )
            for limit in limits
            if limit.get("percent") is not None
        )
    except Exception:
        return _error("Claude Code's usage limits are not in a shape tsugite understands")

    return QuotaReport(provider=PROVIDER, label=LABEL, windows=windows)


register_quota_source(PROVIDER, LABEL, fetch)
