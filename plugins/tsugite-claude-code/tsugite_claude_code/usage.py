"""Claude Code subscription limits as a tsugite quota source.

Read from ``limits[]`` in the usage endpoint Claude Code calls, with its OAuth
token. The endpoint rate-limits hard, so it is called at most hourly and backs
off after a failure, and each turn's ``rate_limit_event`` keeps the session and
weekly windows current in between. With no endpoint reading, the usage copy
cached in Claude Code's config is used while it is under an hour old. The list
is self-describing, so a limit kind tsugite has never seen still shows up.
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

from tsugite.usage.quota import QuotaReport, QuotaWindow, register_quota_source
from tsugite.user_agent import set_user_agent_header

logger = logging.getLogger(__name__)

PROVIDER = "claude_code"
LABEL = "Claude Code"
USAGE_URL = "https://api.anthropic.com/api/oauth/usage"
CACHE_MAX_AGE_MS = 60 * 60 * 1000
LIVE_TTL_S = 60 * 60
BACKOFF_S = (5 * 60, 10 * 60, 20 * 60, 60 * 60)
TURN_WINDOWS = {"five_hour": "session", "seven_day": "weekly_all"}

_last_live: tuple[float, list] | None = None
_failures = 0
_retry_at = 0.0
_turn: dict[str, tuple[float, QuotaWindow]] = {}


def _config_path() -> Path:
    base = os.environ.get("CLAUDE_CONFIG_DIR") or os.path.expanduser("~")
    return Path(base) / ".claude.json"


def _credentials_path() -> Path:
    base = os.environ.get("CLAUDE_CONFIG_DIR") or os.path.expanduser("~/.claude")
    return Path(base) / ".credentials.json"


def _request_limits() -> list:
    token = json.loads(_credentials_path().read_text())["claudeAiOauth"]["accessToken"]
    headers = {"Authorization": f"Bearer {token}", "anthropic-beta": "oauth-2025-04-20"}
    set_user_agent_header(headers)
    response = httpx.get(USAGE_URL, headers=headers, timeout=10)
    response.raise_for_status()
    return response.json()["limits"]


def _refresh_live() -> None:
    global _last_live, _failures, _retry_at
    now = time.time()
    if now < _retry_at:
        return
    try:
        _last_live = (now, _request_limits())
    except Exception as e:
        logger.warning("Claude Code usage endpoint failed: %r", e)
        _retry_at = now + BACKOFF_S[min(_failures, len(BACKOFF_S) - 1)]
        _failures += 1
        return
    _failures, _retry_at = 0, now + LIVE_TTL_S


def _cached_limits() -> tuple[float, list]:
    cache = json.loads(_config_path().read_text())["cachedUsageUtilization"]
    if time.time() * 1000 - cache["fetchedAtMs"] > CACHE_MAX_AGE_MS:
        raise ValueError("cached usage is over an hour old")
    return cache["fetchedAtMs"] / 1000, cache["utilization"]["limits"]


def _label(limit: dict) -> str:
    kind = limit["kind"]
    if kind == "weekly_all":
        return "week"
    if kind == "weekly_scoped":
        model = (limit.get("scope") or {}).get("model") or {}
        name = model.get("display_name")
        return f"week · {name}" if name else kind
    return kind


def record_rate_limit(info: dict) -> None:
    """Keep the windows from a turn's ``rate_limit_event`` (``rate_limit_info``)."""
    windows = info.get("unifiedWindows") or {}
    for name, key in TURN_WINDOWS.items():
        window = windows.get(name)
        if not window or window.get("utilization") is None:
            continue
        resets_at = window.get("resetsAt")
        _turn[key] = (
            time.time(),
            QuotaWindow(
                key=key,
                label=_label({"kind": key}),
                used_pct=round(window["utilization"] * 100, 1),
                resets_at=datetime.fromtimestamp(resets_at, timezone.utc).isoformat() if resets_at else None,
            ),
        )


def _error(message: str) -> QuotaReport:
    return QuotaReport(provider=PROVIDER, label=LABEL, error=message)


def fetch() -> QuotaReport:
    try:
        _refresh_live()
        fetched_at, limits = _last_live or _cached_limits()
    except Exception as e:
        if not _turn:
            logger.warning("Claude Code usage unavailable: %r", e)
            return _error("Claude Code's usage limits could not be fetched, and no recent cached copy exists")
        fetched_at, limits = 0.0, []

    try:
        windows = [
            QuotaWindow(
                key=limit["kind"],
                label=_label(limit),
                used_pct=float(limit["percent"]),
                resets_at=limit.get("resets_at"),
            )
            for limit in limits
            if limit.get("percent") is not None
        ]
    except Exception:
        return _error("Claude Code's usage limits are not in a shape tsugite understands")

    turn = {key: window for key, (read_at, window) in _turn.items() if read_at > fetched_at}
    windows = [*turn.values(), *(w for w in windows if w.key not in turn)]
    return QuotaReport(provider=PROVIDER, label=LABEL, windows=tuple(windows))


register_quota_source(PROVIDER, LABEL, fetch)
