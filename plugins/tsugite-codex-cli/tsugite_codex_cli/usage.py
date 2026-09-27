"""ChatGPT subscription limits for Codex as a tsugite quota source.

Read from ``rate_limit`` in the ChatGPT usage endpoint, with the Codex CLI's
OAuth token. Each window is labeled by its duration, so a window length
tsugite has never seen still shows up.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone

import httpx

from tsugite.usage.quota import QuotaReport, QuotaWindow, register_quota_source
from tsugite.user_agent import set_user_agent_header
from tsugite_codex_cli.auth import CodexAuthStore

logger = logging.getLogger(__name__)

PROVIDER = "codex_cli"
LABEL = "Codex"
USAGE_URL = "https://chatgpt.com/backend-api/wham/usage"
LIVE_TTL_S = 60

_last_live: tuple[float, dict] | None = None


def _live_rate_limit() -> dict:
    global _last_live
    if _last_live and time.monotonic() - _last_live[0] < LIVE_TTL_S:
        return _last_live[1]
    access_token, account_id = CodexAuthStore().sync_get_or_refresh()
    headers = {"Authorization": f"Bearer {access_token}", "ChatGPT-Account-ID": account_id}
    set_user_agent_header(headers)
    response = httpx.get(USAGE_URL, headers=headers, timeout=10)
    response.raise_for_status()
    rate_limit = response.json()["rate_limit"]
    _last_live = (time.monotonic(), rate_limit)
    return rate_limit


def _label(seconds: int) -> str:
    if seconds == 7 * 24 * 3600:
        return "week"
    if seconds % 3600:
        return f"{seconds // 60}m"
    return f"{seconds // 3600}h"


def _error(message: str) -> QuotaReport:
    return QuotaReport(provider=PROVIDER, label=LABEL, error=message)


def fetch() -> QuotaReport:
    try:
        rate_limit = _live_rate_limit()
    except Exception as e:
        # A failed token refresh can quote the auth server's response body.
        logger.warning("Codex usage unavailable: %s", type(e).__name__)
        return _error("Codex's usage limits could not be fetched")

    try:
        windows = tuple(
            QuotaWindow(
                key=key,
                label=_label(window["limit_window_seconds"]),
                used_pct=float(window["used_percent"]),
                resets_at=datetime.fromtimestamp(window["reset_at"], timezone.utc).isoformat(),
            )
            for key in ("primary", "secondary")
            if (window := rate_limit.get(f"{key}_window"))
        )
    except Exception:
        return _error("Codex's usage limits are not in a shape tsugite understands")

    return QuotaReport(provider=PROVIDER, label=LABEL, windows=windows)


register_quota_source(PROVIDER, LABEL, fetch)
