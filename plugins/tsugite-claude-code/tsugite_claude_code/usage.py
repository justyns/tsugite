"""Claude Code subscription limits as a tsugite quota source.

Read from ``utilization.limits[]`` in the usage cache Claude Code writes to its
own config. That list is self-describing, so a limit kind tsugite has never seen
still shows up. Its sibling keys hold an account id, and a spend figure that
reads 100% while disabled; neither belongs in a report.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from tsugite.usage.quota import QuotaReport, QuotaWindow, register_quota_source

PROVIDER = "claude_code"
LABEL = "Claude Code"


def _config_path() -> Path:
    base = os.environ.get("CLAUDE_CONFIG_DIR") or os.path.expanduser("~")
    return Path(base) / ".claude.json"


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
        data = json.loads(_config_path().read_text())
    except Exception:
        return _error("Claude Code's config could not be read")

    cache = data.get("cachedUsageUtilization")
    if not cache:
        return _error("Claude Code has not cached its usage limits yet")

    try:
        windows = tuple(
            QuotaWindow(
                key=limit["kind"],
                label=_label(limit),
                used_pct=float(limit["percent"]),
                resets_at=limit.get("resets_at"),
            )
            for limit in cache["utilization"]["limits"]
        )
    except Exception:
        return _error("Claude Code's cached usage limits are not in a shape tsugite understands")

    return QuotaReport(provider=PROVIDER, label=LABEL, windows=windows)


register_quota_source(PROVIDER, LABEL, fetch)
