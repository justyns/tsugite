"""Cron-like scheduler for recurring and one-off agent tasks."""

# Required: Scheduler.list() shadows builtin list, breaking list[str] annotations
from __future__ import annotations

import asyncio
import copy
import json
import logging
import os
from dataclasses import dataclass, field
from dataclasses import fields as dataclass_fields
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Coroutine
from zoneinfo import ZoneInfo

from cronsim import CronSim, CronSimError

logger = logging.getLogger(__name__)


# target_session sentinel values (see ScheduleEntry.target_session field below).
TARGET_SESSION_PRIMARY = "primary"
TARGET_SESSION_ORIGINATING = "originating"
TARGET_SESSION_NONE = "none"
TARGET_SESSION_NAME_PREFIX = "name:"


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
    last_scheduled_for: str | None = None  # planned fire time of last run (ISO)
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
    # Failure surfacing: notify once after N consecutive errors, re-arm on success.
    consecutive_failures: int = 0
    failure_notified: bool = False
    notify_on_failure: int = 3  # notify after this many consecutive failures; 0 disables
    # Completion callbacks
    originating_session_id: str | None = None  # Session that spawned this task
    on_complete: dict | None = None  # {"action": "reply"} to auto-reply on completion
    chain_depth: int = 0  # How many chained completions deep (safety limit)
    # Where the inject_history synthetic turn lands. Legal forms:
    #   None         -> fallback chain: primary -> originating -> none
    #   "primary"    -> primary lookup only (no fallback)
    #   "originating"-> originating_session_id only
    #   "none"       -> skip injection
    #   "name:<n>"   -> find_named_session(name)
    #   "<sid>"      -> bare session id
    target_session: str | None = None

    # Concurrency lock for this entry's runs. Per-entry so two fires of the
    # same schedule can't overlap; per-instance (not persisted) so the lock
    # belongs to the entry it gates and disappears with it.
    lock: asyncio.Lock = field(
        default_factory=asyncio.Lock,
        repr=False,
        compare=False,
        metadata={"persist": False},
    )

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


_PERSISTED_FIELDS = frozenset(f.name for f in dataclass_fields(ScheduleEntry) if f.metadata.get("persist", True))


def entry_to_dict(entry: "ScheduleEntry") -> dict:
    """Serialize a ScheduleEntry, skipping fields marked metadata.persist=False."""
    return {name: getattr(entry, name) for name in _PERSISTED_FIELDS}


@dataclass
class RunResult:
    output: str
    session_id: str | None = None


RunCallback = Callable[["ScheduleEntry"], Coroutine[None, None, "RunResult"]]


class Scheduler:
    def __init__(
        self,
        schedules_path: Path,
        run_callback: RunCallback,
        script_callback: RunCallback | None = None,
        on_repeated_failure: Callable[["ScheduleEntry"], None] | None = None,
    ):
        self._path = schedules_path
        self._run_callback = run_callback
        self._script_callback = script_callback
        # Called once when a schedule crosses its consecutive-failure threshold,
        # so the daemon/adapter can surface it (Discord/notify). None = log only.
        self._on_repeated_failure = on_repeated_failure
        self._schedules: dict[str, ScheduleEntry] = {}
        self._wakeup = asyncio.Event()
        self._running = False
        self._active_tasks: set[asyncio.Task] = set()

    async def start(self):
        self._load()
        self._arm_loaded_schedules()
        self._save()
        self.cleanup_history()
        self._running = True
        logger.info("Scheduler started with %d schedule(s)", len(self._schedules))
        await self._main_loop()

    def _arm_loaded_schedules(self) -> None:
        """Recompute next_run for everything loaded from disk.

        Per-entry isolation: one corrupt record (bad timezone, garbage dates)
        must not prevent the scheduler from starting - that bricked the daemon
        until schedules.json was hand-edited. One-offs whose time passed beyond
        grace while the daemon was down get a disabled_reason so cleanup()
        reaps them instead of leaving enabled zombies that can never fire.
        """
        for entry in self._schedules.values():
            try:
                entry.next_run = self._compute_next_run_iso(entry)
            except Exception as e:
                logger.exception("Schedule '%s' has invalid data; disabling: %s", entry.id, e)
                entry.enabled = False
                entry.disabled_reason = f"invalid schedule data: {e}"
                entry.next_run = None
                continue
            if entry.schedule_type == "once" and entry.enabled and not entry.next_run and not entry.disabled_reason:
                logger.warning("Schedule '%s' missed its one-off run while the daemon was down; disabling", entry.id)
                entry.enabled = False
                entry.disabled_reason = "missed one-off (run_at passed while daemon was down)"

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
            try:
                self._fire_if_due(entry, now)
            except Exception as e:
                # One corrupt entry (e.g. unparseable expires_at from a legacy
                # or hand-edited record) must not kill the whole fire loop -
                # that silently stopped EVERY schedule until restart, and then
                # crashed again on the next loop.
                logger.exception("Schedule '%s' has invalid data; disabling: %s", entry.id, e)
                self._auto_disable(entry, f"invalid schedule data: {e}")

    def _fire_if_due(self, entry: ScheduleEntry, now: datetime):
        if not entry.enabled or not entry.next_run:
            return

        # Auto-expiry checks
        if entry.expires_at:
            expires_dt = datetime.fromisoformat(entry.expires_at)
            if expires_dt.tzinfo is None:
                expires_dt = expires_dt.replace(tzinfo=timezone.utc)
            if now >= expires_dt:
                self._auto_disable(entry, "expired")
                return
        if entry.max_runs is not None and entry.run_count >= entry.max_runs:
            self._auto_disable(entry, "max_runs_reached")
            return

        next_dt = datetime.fromisoformat(entry.next_run)
        if next_dt > now:
            return

        # Misfire check: skip if too far past due
        grace = timedelta(seconds=entry.misfire_grace_seconds)
        if now - next_dt > grace:
            if entry.schedule_type == "once":
                # No "next run" exists for a one-off - disable with a reason
                # so cleanup() reaps it instead of leaving an enabled zombie
                # that can never fire.
                self._auto_disable(entry, "missed one-off (past misfire grace)")
                return
            logger.warning("Schedule '%s' misfired (past grace period), skipping to next run", entry.id)
            entry.next_run = self._compute_next_run_iso(entry)
            self._save()
            return

        # If the previous run is still in progress, suppress this overlapping fire
        # without corrupting drift metadata: roll next_run forward to avoid busy-spin,
        # but leave last_scheduled_for/last_run reflecting the run that actually happened.
        # (_fire_schedule guards on the same lock; checking here avoids the metadata
        # write + task spawn for a fire that would just be dropped.)
        if entry.lock.locked():
            logger.info("Schedule '%s' still running, skipping overlapping fire", entry.id)
            entry.next_run = None if entry.schedule_type == "once" else self._compute_next_run_iso(entry)
            self._save()
            return

        # Capture the planned fire time before rolling next_run forward,
        # so the adapter can surface scheduled_for vs actual_fire_time to
        # the agent (drift detection on misfires/queue delays).
        entry.last_scheduled_for = entry.next_run

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

        if entry.lock.locked():
            logger.info("Schedule '%s' still running, skipping", entry.id)
            return

        async with entry.lock:
            logger.info("Firing schedule '%s' (type=%s, agent=%s)", entry.id, entry.execution_type, entry.agent)
            run_conv_id = None
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

            if entry.last_status == "error":
                entry.consecutive_failures += 1
                self._maybe_notify_repeated_failure(entry)
            elif entry.last_status == "success":
                # A real run clears the streak and re-arms the notification.
                entry.consecutive_failures = 0
                entry.failure_notified = False
            # "skipped" leaves the streak unchanged: not a failure, not a real run.

            entry.last_run = datetime.now(timezone.utc).isoformat()
            entry.run_history.append(
                {
                    "timestamp": entry.last_run,
                    "status": entry.last_status,
                    "error": entry.last_error,
                    "session_id": run_conv_id,
                }
            )
            if len(entry.run_history) > 20:
                entry.run_history = entry.run_history[-20:]

            if entry.schedule_type == "once":
                self._schedules.pop(entry.id, None)
            self._save()

    def _maybe_notify_repeated_failure(self, entry: ScheduleEntry):
        """Surface a schedule that keeps failing instead of letting it die silently.

        Fires once when the consecutive-failure count first reaches
        notify_on_failure, then stays quiet until a success re-arms it (so a
        permanently-broken schedule doesn't notify on every run). Always logs at
        ERROR; the optional hook adds Discord/notify delivery on top.
        """
        threshold = entry.notify_on_failure
        if threshold <= 0 or entry.consecutive_failures < threshold or entry.failure_notified:
            return
        entry.failure_notified = True
        logger.error(
            "Schedule '%s' has failed %d consecutive times; latest error: %s",
            entry.id,
            entry.consecutive_failures,
            entry.last_error,
        )
        if self._on_repeated_failure is None:
            return
        try:
            self._on_repeated_failure(entry)
        except Exception:
            logger.exception("on_repeated_failure hook raised for schedule '%s'", entry.id)

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
            # A slightly-late one-off (daemon restart spanning the fire time)
            # is still fireable within its misfire grace - _fire_due_schedules
            # applies the same window. Beyond grace it's a miss (None).
            grace = timedelta(seconds=entry.misfire_grace_seconds)
            return run_dt.isoformat() if run_dt > now - grace else None

        if entry.schedule_type == "cron" and entry.cron_expr:
            tz = self._zoneinfo(entry.timezone)
            now_local = now.astimezone(tz)
            it = CronSim(entry.cron_expr, now_local)
            try:
                next_local = next(it)
                return next_local.astimezone(timezone.utc).isoformat()
            except StopIteration:
                return None

        return None

    @staticmethod
    def _zoneinfo(name: str) -> ZoneInfo:
        """ZoneInfo that raises ValueError (not KeyError) on unknown names so
        callers' ValueError handling (HTTP 400 mapping) works."""
        try:
            return ZoneInfo(name)
        except Exception as e:
            raise ValueError(f"Invalid timezone '{name}': {e}") from e

    def _validate_entry(self, entry: ScheduleEntry) -> str | None:
        """Validate everything user-settable and return the computed next_run.

        Raises ValueError on bad cron syntax, sub-minimum cron interval,
        unknown timezone, or unparseable expires_at - shared by add() and
        update() so no creation path can skip a guard.
        """
        if entry.schedule_type == "cron" and entry.cron_expr:
            try:
                CronSim(entry.cron_expr, datetime.now(timezone.utc))
            except (ValueError, KeyError, CronSimError) as e:
                raise ValueError(f"Invalid cron expression '{entry.cron_expr}': {e}") from e
        if entry.expires_at:
            try:
                datetime.fromisoformat(entry.expires_at)
            except ValueError as e:
                raise ValueError(f"Invalid expires_at '{entry.expires_at}': {e}") from e
        next_run = self._compute_next_run_iso(entry)  # raises ValueError on bad timezone
        self._validate_cron_interval(entry)
        return next_run

    # CRUD operations

    def add(self, entry: ScheduleEntry) -> ScheduleEntry:
        if entry.id in self._schedules:
            raise ValueError(f"Schedule '{entry.id}' already exists")
        entry.next_run = self._validate_entry(entry)
        self._schedules[entry.id] = entry
        self._save()
        self._wakeup.set()
        logger.info("Added schedule '%s'", entry.id)
        return entry

    def remove(self, schedule_id: str):
        if schedule_id not in self._schedules:
            raise ValueError(f"Schedule '{schedule_id}' not found")
        del self._schedules[schedule_id]
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
        return [sid for sid, entry in self._schedules.items() if entry.lock.locked()]

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
        updates = {k: v for k, v in fields.items() if k not in ("id", "created_at")}
        for key in updates:
            if not hasattr(entry, key):
                raise ValueError(f"Unknown field '{key}'")
        # Validate against a throwaway copy BEFORE touching the live entry: a
        # rejected update (bad timezone/cron/expires_at) must not poison
        # in-memory state that a later unrelated _save() would persist - that
        # bricked scheduler startup until schedules.json was hand-edited.
        candidate = copy.copy(entry)
        for key, value in updates.items():
            setattr(candidate, key, value)
        next_run = self._validate_entry(candidate)
        for key, value in updates.items():
            setattr(entry, key, value)
        entry.next_run = next_run
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
        if to_remove:
            self._save()
            logger.info("Cleaned up %d orphaned schedule(s)", len(to_remove))
        return to_remove

    def cleanup_history(self, max_age_days: int = 30, max_files_per_schedule: int = 50) -> int:
        """Delete old per-run schedule sessions: excess by count, then by age."""
        from tsugite.history import get_history_backend

        backend = get_history_backend()
        cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)
        active_prefixes = {f"sched_{sid.replace(':', '_')}_" for sid in self._schedules}

        removed = 0
        try:
            # list_sessions() is recency-ordered (newest first). Schedule runs are id-prefixed.
            sched_ids = [s for s in backend.list_sessions() if s.startswith("sched_")]
            old_ids = set(backend.list_sessions(before=cutoff))

            by_prefix: dict[str, list[str]] = {}
            orphans: list[str] = []
            for sid in sched_ids:
                prefix = next((p for p in active_prefixes if sid.startswith(p)), None)
                if prefix:
                    by_prefix.setdefault(prefix, []).append(sid)
                else:
                    orphans.append(sid)

            # Orphaned runs from deleted schedules: remove if old.
            for sid in orphans:
                if sid in old_ids and backend.delete_session(sid):
                    removed += 1
            # Per-schedule: drop the oldest beyond the count cap, then any remaining that are too old.
            for ids in by_prefix.values():
                for sid in ids[max_files_per_schedule:]:
                    if backend.delete_session(sid):
                        removed += 1
                for sid in ids[:max_files_per_schedule]:
                    if sid in old_ids and backend.delete_session(sid):
                        removed += 1
        except NotImplementedError:
            return 0  # the deprecated jsonl backend has no db-side retention

        if removed:
            logger.info("Cleaned up %d old schedule session(s)", removed)
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
            valid_fields = _PERSISTED_FIELDS
            for sid, entry_data in data.get("schedules", {}).items():
                self._schedules[sid] = ScheduleEntry(**{k: v for k, v in entry_data.items() if k in valid_fields})
        except (json.JSONDecodeError, TypeError, KeyError) as e:
            logger.error("Failed to load schedules from %s: %s", self._path, e)
            self._schedules = {}

    def _save(self):
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = {"schedules": {sid: entry_to_dict(entry) for sid, entry in self._schedules.items()}}
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2))
        os.replace(str(tmp), str(self._path))
