"""Cron-like scheduler for recurring and one-off agent tasks."""

# Required: Scheduler.list() shadows builtin list, breaking list[str] annotations
from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Coroutine
from zoneinfo import ZoneInfo

from cronsim import CronSim, CronSimError

logger = logging.getLogger(__name__)


@dataclass
class ScheduleEntry:
    id: str
    agent: str
    prompt: str
    schedule_type: str  # "cron" | "once"
    cron_expr: str | None = None
    run_at: str | None = None  # ISO datetime for one-off
    enabled: bool = True
    created_at: str = ""
    last_run: str | None = None
    next_run: str | None = None  # computed
    last_status: str | None = None  # "success" | "error" | "skipped"
    last_error: str | None = None
    notify: list[str] = field(default_factory=list)
    notify_tool: bool = False
    inject_history: bool = True
    auto_reply: bool = False
    model: str | None = None
    misfire_grace_seconds: int = 300
    timezone: str = "UTC"
    agent_file: str | None = None
    max_turns: int | None = None
    execution_type: str = "agent"  # "agent" | "script"
    command: str | None = None
    script_timeout: int = 60
    # Auto-expiry
    expires_at: str | None = None  # ISO datetime, auto-disable after this time
    max_runs: int | None = None  # Auto-disable after N successful executions
    run_count: int = 0
    disabled_reason: str | None = None  # "expired" | "max_runs_reached"
    # Per-run session isolation
    session_id: str | None = None  # If set, reuse this session across runs; otherwise each run gets a unique session
    run_history: list[dict] = field(default_factory=list)  # Last N runs [{timestamp, status, error, session_id}]

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()
        if self.schedule_type not in ("cron", "once"):
            raise ValueError(f"schedule_type must be 'cron' or 'once', got '{self.schedule_type}'")
        if self.schedule_type == "cron" and not self.cron_expr:
            raise ValueError("cron_expr required for cron schedules")
        if self.schedule_type == "once" and not self.run_at:
            raise ValueError("run_at required for one-off schedules")
        if self.execution_type not in ("agent", "script"):
            raise ValueError(f"execution_type must be 'agent' or 'script', got '{self.execution_type}'")
        if self.execution_type == "script" and not self.command:
            raise ValueError("command required for script execution type")


@dataclass
class RunResult:
    output: str
    session_id: str | None = None
    tokens: int | None = None
    cost: float | None = None


RunCallback = Callable[["ScheduleEntry"], Coroutine[None, None, "RunResult"]]


class Scheduler:
    def __init__(self, schedules_path: Path, run_callback: RunCallback, script_callback: RunCallback | None = None):
        self._path = schedules_path
        self._run_callback = run_callback
        self._script_callback = script_callback
        self._schedules: dict[str, ScheduleEntry] = {}
        self._wakeup = asyncio.Event()
        self._running = False
        self._active_tasks: set[asyncio.Task] = set()
        self._entry_locks: dict[str, asyncio.Lock] = {}

    async def start(self):
        self._load()
        for entry in self._schedules.values():
            entry.next_run = self._compute_next_run_iso(entry)
        self._save()
        self.cleanup_history()
        self._running = True
        logger.info("Scheduler started with %d schedule(s)", len(self._schedules))
        await self._main_loop()

    async def stop(self):
        self._running = False
        self._wakeup.set()
        if self._active_tasks:
            logger.info("Waiting for %d active task(s) to finish...", len(self._active_tasks))
            await asyncio.gather(*self._active_tasks, return_exceptions=True)
        self._save()
        logger.info("Scheduler stopped")

    async def _main_loop(self):
        while self._running:
            now = datetime.now(timezone.utc)
            next_fire = self._find_next_fire_time()

            if next_fire is None:
                # No schedules due — wait for wakeup
                self._wakeup.clear()
                try:
                    await asyncio.wait_for(self._wakeup.wait(), timeout=60)
                except asyncio.TimeoutError:
                    pass
                continue

            delay = (next_fire - now).total_seconds()
            if delay > 0:
                self._wakeup.clear()
                try:
                    await asyncio.wait_for(self._wakeup.wait(), timeout=delay)
                except asyncio.TimeoutError:
                    pass
                if not self._running:
                    break

            self._fire_due_schedules()
            await asyncio.sleep(0)  # Yield to event loop

    def _find_next_fire_time(self) -> datetime | None:
        earliest = None
        for entry in self._schedules.values():
            if not entry.enabled or not entry.next_run:
                continue
            t = datetime.fromisoformat(entry.next_run)
            if earliest is None or t < earliest:
                earliest = t
        return earliest

    def _auto_disable(self, entry: ScheduleEntry, reason: str):
        entry.enabled = False
        entry.disabled_reason = reason
        entry.next_run = None
        self._save()
        logger.info("Schedule '%s' auto-disabled: %s", entry.id, reason)

    def _fire_due_schedules(self):
        now = datetime.now(timezone.utc)
        for entry in list(self._schedules.values()):
            if not entry.enabled or not entry.next_run:
                continue

            # Auto-expiry checks
            if entry.expires_at:
                expires_dt = datetime.fromisoformat(entry.expires_at)
                if expires_dt.tzinfo is None:
                    expires_dt = expires_dt.replace(tzinfo=timezone.utc)
                if now >= expires_dt:
                    self._auto_disable(entry, "expired")
                    continue
            if entry.max_runs is not None and entry.run_count >= entry.max_runs:
                self._auto_disable(entry, "max_runs_reached")
                continue

            next_dt = datetime.fromisoformat(entry.next_run)
            if next_dt > now:
                continue

            # Misfire check: skip if too far past due
            grace = timedelta(seconds=entry.misfire_grace_seconds)
            if now - next_dt > grace:
                logger.warning("Schedule '%s' misfired (past grace period), skipping to next run", entry.id)
                entry.next_run = self._compute_next_run_iso(entry)
                self._save()
                continue

            # Update next_run IMMEDIATELY so the main loop doesn't busy-spin
            if entry.schedule_type == "once":
                entry.next_run = None
            else:
                entry.next_run = self._compute_next_run_iso(entry)

            task = asyncio.create_task(self._fire_schedule(entry))
            self._active_tasks.add(task)
            task.add_done_callback(self._active_tasks.discard)

    async def _fire_schedule(self, entry: ScheduleEntry):
        from tsugite.agent_runner.models import AgentSkippedError

        lock = self._entry_locks.setdefault(entry.id, asyncio.Lock())
        if lock.locked():
            logger.info("Schedule '%s' still running, skipping", entry.id)
            return

        async with lock:
            logger.info("Firing schedule '%s' (type=%s, agent=%s)", entry.id, entry.execution_type, entry.agent)
            run_conv_id = None
            run_result = None
            try:
                if entry.execution_type == "script":
                    if not self._script_callback:
                        raise RuntimeError("No script callback configured — cannot run script schedules")
                    await self._script_callback(entry)
                else:
                    run_result = await self._run_callback(entry)
                    run_conv_id = run_result.session_id
                entry.last_status = "success"
                entry.last_error = None
                entry.run_count += 1
            except AgentSkippedError as e:
                logger.info("Schedule '%s' skipped: %s", entry.id, e.reason)
                entry.last_status = "skipped"
                entry.last_error = None
            except Exception as e:
                logger.error("Schedule '%s' failed: %s", entry.id, e)
                entry.last_status = "error"
                entry.last_error = str(e)

            entry.last_run = datetime.now(timezone.utc).isoformat()
            run_record = {
                "timestamp": entry.last_run,
                "status": entry.last_status,
                "error": entry.last_error,
                "session_id": run_conv_id,
            }
            if run_result is not None:
                if run_result.tokens is not None:
                    run_record["tokens"] = run_result.tokens
                if run_result.cost is not None:
                    run_record["cost"] = run_result.cost
            entry.run_history.append(run_record)
            if len(entry.run_history) > 20:
                entry.run_history = entry.run_history[-20:]

            if entry.schedule_type == "once":
                self._schedules.pop(entry.id, None)
                self._entry_locks.pop(entry.id, None)
            self._save()

    def _compute_next_run_iso(self, entry: ScheduleEntry) -> str | None:
        now = datetime.now(timezone.utc)

        if entry.schedule_type == "once":
            if not entry.run_at:
                return None
            run_dt = datetime.fromisoformat(entry.run_at)
            if run_dt.tzinfo is None:
                run_dt = run_dt.replace(tzinfo=timezone.utc)
            else:
                run_dt = run_dt.astimezone(timezone.utc)
            return run_dt.isoformat() if run_dt > now else None

        if entry.schedule_type == "cron" and entry.cron_expr:
            tz = ZoneInfo(entry.timezone)
            now_local = now.astimezone(tz)
            it = CronSim(entry.cron_expr, now_local)
            try:
                next_local = next(it)
                return next_local.astimezone(timezone.utc).isoformat()
            except StopIteration:
                return None

        return None

    # CRUD operations

    def add(self, entry: ScheduleEntry) -> ScheduleEntry:
        if entry.id in self._schedules:
            raise ValueError(f"Schedule '{entry.id}' already exists")
        # Validate cron expression early
        if entry.schedule_type == "cron" and entry.cron_expr:
            try:
                CronSim(entry.cron_expr, datetime.now(timezone.utc))
            except (ValueError, KeyError, CronSimError) as e:
                raise ValueError(f"Invalid cron expression '{entry.cron_expr}': {e}") from e
        entry.next_run = self._compute_next_run_iso(entry)
        self._schedules[entry.id] = entry
        self._save()
        self._wakeup.set()
        logger.info("Added schedule '%s'", entry.id)
        return entry

    def remove(self, schedule_id: str):
        if schedule_id not in self._schedules:
            raise ValueError(f"Schedule '{schedule_id}' not found")
        del self._schedules[schedule_id]
        self._entry_locks.pop(schedule_id, None)
        self._save()
        self._wakeup.set()
        logger.info("Removed schedule '%s'", schedule_id)

    def enable(self, schedule_id: str):
        entry = self.get(schedule_id)
        entry.enabled = True
        entry.disabled_reason = None
        entry.next_run = self._compute_next_run_iso(entry)
        self._save()
        self._wakeup.set()

    def disable(self, schedule_id: str):
        entry = self.get(schedule_id)
        entry.enabled = False
        self._save()

    def get_running_ids(self) -> list[str]:
        return [sid for sid, lock in self._entry_locks.items() if lock.locked()]

    def list(self) -> list[ScheduleEntry]:
        return list(self._schedules.values())

    def get(self, schedule_id: str) -> ScheduleEntry:
        if schedule_id not in self._schedules:
            raise ValueError(f"Schedule '{schedule_id}' not found")
        return self._schedules[schedule_id]

    MIN_INTERVAL_SECONDS = 120  # 2 minutes — prevent agents from burning tokens

    def _validate_cron_interval(self, entry: ScheduleEntry) -> None:
        """Reject cron expressions that fire more frequently than MIN_INTERVAL_SECONDS."""
        if entry.schedule_type != "cron" or not entry.cron_expr:
            return
        try:
            tz = ZoneInfo(entry.timezone)
            now_local = datetime.now(timezone.utc).astimezone(tz)
            it = CronSim(entry.cron_expr, now_local)
            first = next(it)
            second = next(it)
            interval = (second - first).total_seconds()
            if interval < self.MIN_INTERVAL_SECONDS:
                raise ValueError(
                    f"Cron expression '{entry.cron_expr}' fires every {int(interval)}s "
                    f"(minimum interval: {self.MIN_INTERVAL_SECONDS}s)"
                )
        except StopIteration:
            pass

    def update(self, schedule_id: str, **fields) -> ScheduleEntry:
        entry = self.get(schedule_id)
        for key, value in fields.items():
            if key in ("id", "created_at"):
                continue
            if not hasattr(entry, key):
                raise ValueError(f"Unknown field '{key}'")
            setattr(entry, key, value)
        if "cron_expr" in fields and entry.cron_expr:
            try:
                CronSim(entry.cron_expr, datetime.now(timezone.utc))
            except (ValueError, KeyError, CronSimError) as e:
                raise ValueError(f"Invalid cron expression '{entry.cron_expr}': {e}") from e
        entry.next_run = self._compute_next_run_iso(entry)
        self._validate_cron_interval(entry)
        self._save()
        self._wakeup.set()
        logger.info("Updated schedule '%s'", schedule_id)
        return entry

    def cleanup(self) -> list[str]:
        """Remove disabled one-off schedules and auto-disabled (expired/max_runs) schedules."""
        to_remove = [
            sid
            for sid, entry in self._schedules.items()
            if (entry.schedule_type == "once" and not entry.enabled) or entry.disabled_reason
        ]
        for sid in to_remove:
            del self._schedules[sid]
            self._entry_locks.pop(sid, None)
        if to_remove:
            self._save()
            logger.info("Cleaned up %d orphaned schedule(s)", len(to_remove))
        return to_remove

    def cleanup_history(self, max_age_days: int = 30, max_files_per_schedule: int = 50) -> int:
        """Delete old per-run schedule session files."""
        from tsugite.history import get_history_dir

        history_dir = get_history_dir()
        if not history_dir.exists():
            return 0

        removed = 0
        cutoff_ts = (datetime.now(timezone.utc) - timedelta(days=max_age_days)).timestamp()

        # Active schedule prefixes for matching
        active_prefixes = set()
        for sid in self._schedules:
            active_prefixes.add(f"sched_{sid.replace(':', '_')}_")

        # Single glob for all schedule session files
        all_files: dict[str, list[tuple[Path, float]]] = {}
        for f in history_dir.glob("sched_*.jsonl"):
            try:
                mtime = f.stat().st_mtime
            except FileNotFoundError:
                continue
            # Find which prefix this file belongs to
            name = f.name
            matched_prefix = None
            for prefix in active_prefixes:
                if name.startswith(prefix):
                    matched_prefix = prefix
                    break

            if matched_prefix:
                all_files.setdefault(matched_prefix, []).append((f, mtime))
            elif mtime < cutoff_ts:
                # Orphaned file from deleted schedule — remove if old
                f.unlink(missing_ok=True)
                removed += 1

        # Per-schedule: remove excess by count, then by age
        for prefix, file_list in all_files.items():
            file_list.sort(key=lambda x: x[1])
            for f, _ in file_list[:-max_files_per_schedule]:
                f.unlink(missing_ok=True)
                removed += 1
            for f, mtime in file_list[-max_files_per_schedule:]:
                if mtime < cutoff_ts:
                    f.unlink(missing_ok=True)
                    removed += 1

        if removed:
            logger.info("Cleaned up %d old schedule session file(s)", removed)
        return removed

    def fire_now(self, schedule_id: str) -> None:
        """Fire a schedule immediately in the background."""
        entry = self.get(schedule_id)
        if not entry.enabled:
            raise ValueError(f"Schedule '{schedule_id}' is disabled")
        task = asyncio.create_task(self._fire_schedule(entry))
        self._active_tasks.add(task)
        task.add_done_callback(self._active_tasks.discard)

    # Persistence

    def _load(self):
        if not self._path.exists():
            self._schedules = {}
            return
        try:
            data = json.loads(self._path.read_text())
            for sid, entry_data in data.get("schedules", {}).items():
                self._schedules[sid] = ScheduleEntry(**entry_data)
        except (json.JSONDecodeError, TypeError, KeyError) as e:
            logger.error("Failed to load schedules from %s: %s", self._path, e)
            self._schedules = {}

    def _save(self):
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = {"schedules": {sid: asdict(entry) for sid, entry in self._schedules.items()}}
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2))
        os.replace(str(tmp), str(self._path))
