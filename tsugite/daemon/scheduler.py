"""Cron-like scheduler for recurring and one-off agent tasks."""

import asyncio
import json
import logging
import os
from dataclasses import asdict, dataclass
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
    last_status: str | None = None  # "success" | "error"
    last_error: str | None = None
    misfire_grace_seconds: int = 300
    timezone: str = "UTC"

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()
        if self.schedule_type not in ("cron", "once"):
            raise ValueError(f"schedule_type must be 'cron' or 'once', got '{self.schedule_type}'")
        if self.schedule_type == "cron" and not self.cron_expr:
            raise ValueError("cron_expr required for cron schedules")
        if self.schedule_type == "once" and not self.run_at:
            raise ValueError("run_at required for one-off schedules")


RunCallback = Callable[[str, str, str], Coroutine[None, None, str]]


class Scheduler:
    def __init__(self, schedules_path: Path, run_callback: RunCallback):
        self._path = schedules_path
        self._run_callback = run_callback
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

    def _fire_due_schedules(self):
        now = datetime.now(timezone.utc)
        for entry in list(self._schedules.values()):
            if not entry.enabled or not entry.next_run:
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
        lock = self._entry_locks.setdefault(entry.id, asyncio.Lock())
        if lock.locked():
            logger.info("Schedule '%s' still running, skipping", entry.id)
            return

        async with lock:
            logger.info("Firing schedule '%s' (agent=%s)", entry.id, entry.agent)
            try:
                await self._run_callback(entry.agent, entry.prompt, entry.id)
                entry.last_status = "success"
                entry.last_error = None
            except Exception as e:
                logger.error("Schedule '%s' failed: %s", entry.id, e)
                entry.last_status = "error"
                entry.last_error = str(e)

            entry.last_run = datetime.now(timezone.utc).isoformat()
            if entry.schedule_type == "once":
                entry.enabled = False
                entry.next_run = None
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
        entry.next_run = self._compute_next_run_iso(entry)
        self._save()
        self._wakeup.set()

    def disable(self, schedule_id: str):
        entry = self.get(schedule_id)
        entry.enabled = False
        self._save()

    def list(self) -> list[ScheduleEntry]:
        return list(self._schedules.values())

    def get(self, schedule_id: str) -> ScheduleEntry:
        if schedule_id not in self._schedules:
            raise ValueError(f"Schedule '{schedule_id}' not found")
        return self._schedules[schedule_id]

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
        self._save()
        self._wakeup.set()
        logger.info("Updated schedule '%s'", schedule_id)
        return entry

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
