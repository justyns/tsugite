"""Per-job Claude Code settings.json builder + on-disk lifecycle.

The settings file registers the three HTTP hooks the driver needs (Stop,
StopFailure, Notification), all pointing at the plugin's per-job hook URL. Per the
spike, SessionStart does NOT fire from --settings, so it is intentionally absent;
the driving protocol is baked into the initial prompt instead (see hooks.py).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

# The hook events the driver registers. Order is irrelevant to Claude Code but
# kept stable for deterministic tests.
HOOK_EVENTS = ("Stop", "StopFailure", "Notification")


def build_settings(hook_url: str) -> dict:
    """Build the Claude Code settings dict registering the driver's HTTP hooks.

    Every event points at the same per-job hook URL; the receiver dispatches on
    the payload's `hook_event_name`.
    """
    return {"hooks": {event: [{"hooks": [{"type": "http", "url": hook_url}]}] for event in HOOK_EVENTS}}


def write_run_settings(state_dir: str | Path, job_id: str, settings: dict) -> Path:
    """Write `settings` to `<state_dir>/<job_id>/settings.json` and return its path."""
    run_dir = Path(state_dir) / job_id
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "settings.json"
    path.write_text(json.dumps(settings, indent=2))
    return path


def cleanup(state_dir: str | Path, job_id: str) -> None:
    """Remove the job's settings directory. Best-effort; never raises."""
    shutil.rmtree(Path(state_dir) / job_id, ignore_errors=True)
