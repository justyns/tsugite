"""Tests for daemon scheduler."""

import asyncio
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from tsugite.daemon.scheduler import ScheduleEntry, Scheduler


@pytest.fixture
def schedules_path(tmp_path):
    return tmp_path / "schedules.json"


@pytest.fixture
def run_callback():
    return AsyncMock(return_value="done")


@pytest.fixture
def scheduler(schedules_path, run_callback):
    return Scheduler(schedules_path, run_callback)


class TestScheduleEntry:
    def test_cron_entry(self):
        entry = ScheduleEntry(id="test", agent="bot", prompt="hello", schedule_type="cron", cron_expr="0 9 * * *")
        assert entry.schedule_type == "cron"
        assert entry.enabled is True
        assert entry.created_at  # auto-set

    def test_once_entry(self):
        entry = ScheduleEntry(
            id="test", agent="bot", prompt="hello", schedule_type="once", run_at="2099-01-01T00:00:00Z"
        )
        assert entry.schedule_type == "once"

    def test_invalid_type(self):
        with pytest.raises(ValueError, match="schedule_type must be"):
            ScheduleEntry(id="test", agent="bot", prompt="hello", schedule_type="bad")

    def test_cron_requires_expr(self):
        with pytest.raises(ValueError, match="cron_expr required"):
            ScheduleEntry(id="test", agent="bot", prompt="hello", schedule_type="cron")

    def test_once_requires_run_at(self):
        with pytest.raises(ValueError, match="run_at required"):
            ScheduleEntry(id="test", agent="bot", prompt="hello", schedule_type="once")


class TestSchedulerCRUD:
    def test_add_and_list(self, scheduler):
        entry = ScheduleEntry(id="job1", agent="bot", prompt="hi", schedule_type="cron", cron_expr="0 9 * * *")
        scheduler.add(entry)
        assert len(scheduler.list()) == 1
        assert scheduler.get("job1").id == "job1"

    def test_add_duplicate_raises(self, scheduler):
        entry = ScheduleEntry(id="job1", agent="bot", prompt="hi", schedule_type="cron", cron_expr="0 9 * * *")
        scheduler.add(entry)
        with pytest.raises(ValueError, match="already exists"):
            scheduler.add(entry)

    def test_remove(self, scheduler):
        entry = ScheduleEntry(id="job1", agent="bot", prompt="hi", schedule_type="cron", cron_expr="0 9 * * *")
        scheduler.add(entry)
        scheduler.remove("job1")
        assert len(scheduler.list()) == 0

    def test_remove_not_found(self, scheduler):
        with pytest.raises(ValueError, match="not found"):
            scheduler.remove("nonexistent")

    def test_enable_disable(self, scheduler):
        entry = ScheduleEntry(id="job1", agent="bot", prompt="hi", schedule_type="cron", cron_expr="0 9 * * *")
        scheduler.add(entry)
        scheduler.disable("job1")
        assert not scheduler.get("job1").enabled
        scheduler.enable("job1")
        assert scheduler.get("job1").enabled

    def test_invalid_cron_rejected(self, scheduler):
        entry = ScheduleEntry(
            id="bad", agent="bot", prompt="hi", schedule_type="cron", cron_expr="not-a-cron"
        )
        with pytest.raises(ValueError, match="Invalid cron"):
            scheduler.add(entry)


class TestSchedulerPersistence:
    def test_save_and_load(self, schedules_path, run_callback):
        sched1 = Scheduler(schedules_path, run_callback)
        entry = ScheduleEntry(id="job1", agent="bot", prompt="hi", schedule_type="cron", cron_expr="0 9 * * *")
        sched1.add(entry)

        # Load from same file
        sched2 = Scheduler(schedules_path, run_callback)
        sched2._load()
        assert len(sched2.list()) == 1
        assert sched2.get("job1").agent == "bot"

    def test_load_missing_file(self, tmp_path, run_callback):
        sched = Scheduler(tmp_path / "does-not-exist.json", run_callback)
        sched._load()
        assert len(sched.list()) == 0

    def test_load_corrupt_file(self, schedules_path, run_callback):
        schedules_path.write_text("not json at all")
        sched = Scheduler(schedules_path, run_callback)
        sched._load()
        assert len(sched.list()) == 0

    def test_atomic_write(self, schedules_path, run_callback):
        sched = Scheduler(schedules_path, run_callback)
        entry = ScheduleEntry(id="job1", agent="bot", prompt="hi", schedule_type="cron", cron_expr="0 9 * * *")
        sched.add(entry)

        # Verify no .tmp file left behind
        assert not schedules_path.with_suffix(".tmp").exists()
        # Verify JSON is valid
        data = json.loads(schedules_path.read_text())
        assert "job1" in data["schedules"]


class TestNextRunComputation:
    def test_cron_next_run_is_future(self, scheduler):
        entry = ScheduleEntry(id="job1", agent="bot", prompt="hi", schedule_type="cron", cron_expr="* * * * *")
        scheduler.add(entry)
        stored = scheduler.get("job1")
        assert stored.next_run is not None
        next_dt = datetime.fromisoformat(stored.next_run)
        assert next_dt > datetime.now(timezone.utc)

    def test_once_future(self, scheduler):
        future = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
        entry = ScheduleEntry(id="job1", agent="bot", prompt="hi", schedule_type="once", run_at=future)
        scheduler.add(entry)
        assert scheduler.get("job1").next_run is not None

    def test_once_past_returns_none(self, scheduler):
        past = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        entry = ScheduleEntry(id="job1", agent="bot", prompt="hi", schedule_type="once", run_at=past)
        scheduler.add(entry)
        assert scheduler.get("job1").next_run is None

    def test_cron_with_timezone(self, scheduler):
        entry = ScheduleEntry(
            id="job1", agent="bot", prompt="hi", schedule_type="cron", cron_expr="0 9 * * *", timezone="US/Eastern"
        )
        scheduler.add(entry)
        stored = scheduler.get("job1")
        assert stored.next_run is not None


class TestSchedulerExecution:
    @pytest.mark.asyncio
    async def test_fire_schedule_success(self, scheduler, run_callback):
        entry = ScheduleEntry(id="job1", agent="bot", prompt="hi", schedule_type="cron", cron_expr="* * * * *")
        scheduler.add(entry)
        await scheduler._fire_schedule(scheduler.get("job1"))

        run_callback.assert_awaited_once_with("bot", "hi", "job1")
        stored = scheduler.get("job1")
        assert stored.last_status == "success"
        assert stored.last_run is not None

    @pytest.mark.asyncio
    async def test_fire_schedule_error(self, scheduler, run_callback):
        run_callback.side_effect = RuntimeError("agent crashed")
        entry = ScheduleEntry(id="job1", agent="bot", prompt="hi", schedule_type="cron", cron_expr="* * * * *")
        scheduler.add(entry)
        await scheduler._fire_schedule(scheduler.get("job1"))

        stored = scheduler.get("job1")
        assert stored.last_status == "error"
        assert "agent crashed" in stored.last_error

    @pytest.mark.asyncio
    async def test_once_auto_disables(self, scheduler, run_callback):
        future = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
        entry = ScheduleEntry(id="job1", agent="bot", prompt="hi", schedule_type="once", run_at=future)
        scheduler.add(entry)
        await scheduler._fire_schedule(scheduler.get("job1"))

        stored = scheduler.get("job1")
        assert not stored.enabled
        assert stored.next_run is None

    @pytest.mark.asyncio
    async def test_misfire_grace(self, scheduler):
        entry = ScheduleEntry(
            id="job1", agent="bot", prompt="hi", schedule_type="cron",
            cron_expr="* * * * *", misfire_grace_seconds=0,
        )
        scheduler.add(entry)
        # Set next_run far in the past
        scheduler.get("job1").next_run = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        scheduler._save()

        scheduler._fire_due_schedules()
        # Should have been skipped (past grace), next_run advanced
        stored = scheduler.get("job1")
        next_dt = datetime.fromisoformat(stored.next_run)
        assert next_dt > datetime.now(timezone.utc)
