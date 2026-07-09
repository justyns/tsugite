"""Tests for daemon scheduler."""

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock

import pytest
from tsugite_daemon.scheduler import RunResult, ScheduleEntry, Scheduler


@pytest.fixture
def schedules_path(tmp_path):
    return tmp_path / "schedules.json"


@pytest.fixture
def run_callback():
    return AsyncMock(return_value=RunResult(output="done"))


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
        entry = ScheduleEntry(id="bad", agent="bot", prompt="hi", schedule_type="cron", cron_expr="not-a-cron")
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

    def test_durable_write_through(self, schedules_path, run_callback):
        """Saves go write-through to daemon.db; no legacy JSON is written and a
        fresh Scheduler over the same dir sees the entry with no shutdown."""
        sched = Scheduler(schedules_path, run_callback)
        entry = ScheduleEntry(id="job1", agent="bot", prompt="hi", schedule_type="cron", cron_expr="0 9 * * *")
        sched.add(entry)

        assert not schedules_path.exists(), "legacy schedules.json must not be written"
        reloaded = Scheduler(schedules_path, run_callback)
        reloaded._load()
        assert reloaded.get("job1") is not None


class TestNextRunComputation:
    def test_cron_next_run_is_future(self, scheduler):
        entry = ScheduleEntry(id="job1", agent="bot", prompt="hi", schedule_type="cron", cron_expr="*/5 * * * *")
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
        entry = ScheduleEntry(id="job1", agent="bot", prompt="hi", schedule_type="cron", cron_expr="*/5 * * * *")
        scheduler.add(entry)
        await scheduler._fire_schedule(scheduler.get("job1"))

        run_callback.assert_awaited_once()
        called_entry = run_callback.call_args[0][0]
        assert called_entry.agent == "bot"
        assert called_entry.prompt == "hi"
        assert called_entry.id == "job1"
        stored = scheduler.get("job1")
        assert stored.last_status == "success"
        assert stored.last_run is not None

    @pytest.mark.asyncio
    async def test_fire_schedule_error(self, scheduler, run_callback):
        run_callback.side_effect = RuntimeError("agent crashed")
        entry = ScheduleEntry(id="job1", agent="bot", prompt="hi", schedule_type="cron", cron_expr="*/5 * * * *")
        scheduler.add(entry)
        await scheduler._fire_schedule(scheduler.get("job1"))

        stored = scheduler.get("job1")
        assert stored.last_status == "error"
        assert "agent crashed" in stored.last_error

    @pytest.mark.asyncio
    async def test_once_auto_removed(self, scheduler, run_callback):
        future = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
        entry = ScheduleEntry(id="job1", agent="bot", prompt="hi", schedule_type="once", run_at=future)
        scheduler.add(entry)
        await scheduler._fire_schedule(scheduler.get("job1"))

        # One-off schedules are auto-removed after firing
        with pytest.raises(ValueError, match="not found"):
            scheduler.get("job1")

    @pytest.mark.asyncio
    async def test_misfire_grace(self, scheduler):
        entry = ScheduleEntry(
            id="job1",
            agent="bot",
            prompt="hi",
            schedule_type="cron",
            cron_expr="*/5 * * * *",
            misfire_grace_seconds=0,
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


class TestSchedulerUpdate:
    def test_update_prompt(self, scheduler):
        entry = ScheduleEntry(id="job1", agent="bot", prompt="old", schedule_type="cron", cron_expr="0 9 * * *")
        scheduler.add(entry)
        scheduler.update("job1", prompt="new prompt")
        assert scheduler.get("job1").prompt == "new prompt"

    def test_update_cron_expr(self, scheduler):
        entry = ScheduleEntry(id="job1", agent="bot", prompt="hi", schedule_type="cron", cron_expr="0 9 * * *")
        scheduler.add(entry)
        scheduler.update("job1", cron_expr="30 8 * * *")
        assert scheduler.get("job1").cron_expr == "30 8 * * *"

    def test_update_notify_tool(self, scheduler):
        entry = ScheduleEntry(
            id="job1", agent="bot", prompt="hi", schedule_type="cron", cron_expr="0 9 * * *", notify=["discord"]
        )
        scheduler.add(entry)
        scheduler.update("job1", notify_tool=True)
        assert scheduler.get("job1").notify_tool is True


class TestSchedulerCleanup:
    def test_cleanup_removes_disabled_once(self, scheduler):
        future = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
        entry = ScheduleEntry(id="oneshot", agent="bot", prompt="hi", schedule_type="once", run_at=future)
        scheduler.add(entry)
        # Simulate fired: disable + clear next_run
        scheduler.get("oneshot").enabled = False
        scheduler.get("oneshot").next_run = None
        scheduler._save()

        removed = scheduler.cleanup()
        assert removed == ["oneshot"]
        assert len(scheduler.list()) == 0

    def test_cleanup_skips_cron(self, scheduler):
        entry = ScheduleEntry(id="cron1", agent="bot", prompt="hi", schedule_type="cron", cron_expr="0 9 * * *")
        scheduler.add(entry)
        scheduler.disable("cron1")

        removed = scheduler.cleanup()
        assert removed == []
        assert len(scheduler.list()) == 1

    def test_cleanup_skips_enabled_once(self, scheduler):
        future = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
        entry = ScheduleEntry(id="pending", agent="bot", prompt="hi", schedule_type="once", run_at=future)
        scheduler.add(entry)

        removed = scheduler.cleanup()
        assert removed == []
        assert len(scheduler.list()) == 1

    def test_cleanup_empty(self, scheduler):
        removed = scheduler.cleanup()
        assert removed == []


class TestModelField:
    def test_model_defaults_none(self):
        entry = ScheduleEntry(id="test", agent="bot", prompt="hi", schedule_type="cron", cron_expr="0 9 * * *")
        assert entry.model is None

    def test_model_serialization_roundtrip(self, schedules_path, run_callback):
        sched1 = Scheduler(schedules_path, run_callback)
        entry = ScheduleEntry(
            id="job1", agent="bot", prompt="hi", schedule_type="cron", cron_expr="0 9 * * *", model="openai:gpt-4o-mini"
        )
        sched1.add(entry)

        sched2 = Scheduler(schedules_path, run_callback)
        sched2._load()
        assert sched2.get("job1").model == "openai:gpt-4o-mini"

    def test_model_none_serialization_roundtrip(self, schedules_path, run_callback):
        sched1 = Scheduler(schedules_path, run_callback)
        entry = ScheduleEntry(id="job1", agent="bot", prompt="hi", schedule_type="cron", cron_expr="0 9 * * *")
        sched1.add(entry)

        sched2 = Scheduler(schedules_path, run_callback)
        sched2._load()
        assert sched2.get("job1").model is None


class TestScriptEntry:
    def test_script_entry_creation(self):
        entry = ScheduleEntry(
            id="test",
            agent="bot",
            prompt="",
            schedule_type="cron",
            cron_expr="0 * * * *",
            execution_type="script",
            command="echo hello",
        )
        assert entry.execution_type == "script"
        assert entry.command == "echo hello"
        assert entry.script_timeout == 60

    def test_script_requires_command(self):
        with pytest.raises(ValueError, match="command required"):
            ScheduleEntry(
                id="test",
                agent="bot",
                prompt="",
                schedule_type="cron",
                cron_expr="0 * * * *",
                execution_type="script",
            )

    def test_invalid_execution_type(self):
        with pytest.raises(ValueError, match="execution_type must be"):
            ScheduleEntry(
                id="test",
                agent="bot",
                prompt="hi",
                schedule_type="cron",
                cron_expr="0 * * * *",
                execution_type="bad",
            )

    def test_agent_defaults(self):
        entry = ScheduleEntry(id="test", agent="bot", prompt="hi", schedule_type="cron", cron_expr="0 9 * * *")
        assert entry.execution_type == "agent"
        assert entry.command is None
        assert entry.script_timeout == 60

    def test_custom_script_timeout(self):
        entry = ScheduleEntry(
            id="test",
            agent="bot",
            prompt="",
            schedule_type="cron",
            cron_expr="0 * * * *",
            execution_type="script",
            command="curl http://example.com",
            script_timeout=120,
        )
        assert entry.script_timeout == 120


class TestScriptDispatch:
    @pytest.mark.asyncio
    async def test_script_uses_script_callback(self, schedules_path):
        run_cb = AsyncMock(return_value="agent result")
        script_cb = AsyncMock(return_value="script result")
        sched = Scheduler(schedules_path, run_cb, script_callback=script_cb)

        entry = ScheduleEntry(
            id="s1",
            agent="bot",
            prompt="",
            schedule_type="cron",
            cron_expr="*/5 * * * *",
            execution_type="script",
            command="echo hi",
        )
        sched.add(entry)
        await sched._fire_schedule(sched.get("s1"))

        script_cb.assert_awaited_once()
        run_cb.assert_not_awaited()
        assert sched.get("s1").last_status == "success"

    @pytest.mark.asyncio
    async def test_agent_uses_run_callback(self, schedules_path):
        run_cb = AsyncMock(return_value="agent result")
        script_cb = AsyncMock(return_value="script result")
        sched = Scheduler(schedules_path, run_cb, script_callback=script_cb)

        entry = ScheduleEntry(
            id="a1",
            agent="bot",
            prompt="hi",
            schedule_type="cron",
            cron_expr="*/5 * * * *",
        )
        sched.add(entry)
        await sched._fire_schedule(sched.get("a1"))

        run_cb.assert_awaited_once()
        script_cb.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_missing_script_callback_raises(self, schedules_path):
        run_cb = AsyncMock(return_value="done")
        sched = Scheduler(schedules_path, run_cb)  # no script_callback

        entry = ScheduleEntry(
            id="s1",
            agent="bot",
            prompt="",
            schedule_type="cron",
            cron_expr="*/5 * * * *",
            execution_type="script",
            command="echo hi",
        )
        sched.add(entry)
        await sched._fire_schedule(sched.get("s1"))

        assert sched.get("s1").last_status == "error"
        assert "No script callback" in sched.get("s1").last_error


class TestScriptPersistence:
    def test_script_entry_roundtrip(self, schedules_path, run_callback):
        sched1 = Scheduler(schedules_path, run_callback)
        entry = ScheduleEntry(
            id="script1",
            agent="bot",
            prompt="",
            schedule_type="cron",
            cron_expr="0 * * * *",
            execution_type="script",
            command="df -h /",
            script_timeout=30,
        )
        sched1.add(entry)

        sched2 = Scheduler(schedules_path, run_callback)
        sched2._load()
        loaded = sched2.get("script1")
        assert loaded.execution_type == "script"
        assert loaded.command == "df -h /"
        assert loaded.script_timeout == 30

    def test_backward_compat_agent_entry(self, schedules_path, run_callback):
        """Old entries without execution_type fields load fine with defaults."""
        sched1 = Scheduler(schedules_path, run_callback)
        entry = ScheduleEntry(id="old1", agent="bot", prompt="hi", schedule_type="cron", cron_expr="0 9 * * *")
        sched1.add(entry)

        sched2 = Scheduler(schedules_path, run_callback)
        sched2._load()
        loaded = sched2.get("old1")
        assert loaded.execution_type == "agent"
        assert loaded.command is None


class TestCallHelper:
    """Test the _call helper in schedule tools forwards kwargs."""

    def test_call_forwards_kwargs(self):
        """Regression: _call must forward **kwargs to the target function."""
        from threading import Thread
        from unittest.mock import MagicMock

        from tsugite.tools.schedule import _call, set_scheduler

        loop = asyncio.new_event_loop()
        t = Thread(target=loop.run_forever, daemon=True)
        t.start()

        try:
            captured = {}

            def fake_update(id, **fields):
                captured.update({"id": id, **fields})
                return id

            mock_sched = MagicMock()
            set_scheduler(mock_sched, loop)

            result = _call(fake_update, "job1", prompt="new", cron_expr="0 8 * * *")
            assert result == "job1"
            assert captured == {"id": "job1", "prompt": "new", "cron_expr": "0 8 * * *"}
        finally:
            loop.call_soon_threadsafe(loop.stop)
            t.join(timeout=2)
            loop.close()
            set_scheduler(None)


class TestAutoExpiry:
    @pytest.mark.asyncio
    async def test_expires_at_disables(self, scheduler, run_callback):
        past = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        entry = ScheduleEntry(
            id="job1", agent="bot", prompt="hi", schedule_type="cron", cron_expr="*/5 * * * *", expires_at=past
        )
        scheduler.add(entry)
        scheduler._fire_due_schedules()

        stored = scheduler.get("job1")
        assert not stored.enabled
        assert stored.disabled_reason == "expired"
        assert stored.next_run is None
        run_callback.assert_not_called()

    @pytest.mark.asyncio
    async def test_expires_at_future_fires(self, scheduler, run_callback):
        future_expires = (datetime.now(timezone.utc) + timedelta(days=1)).isoformat()
        entry = ScheduleEntry(
            id="job1",
            agent="bot",
            prompt="hi",
            schedule_type="cron",
            cron_expr="*/5 * * * *",
            expires_at=future_expires,
        )
        scheduler.add(entry)
        # Set next_run to the past so it fires
        scheduler.get("job1").next_run = (datetime.now(timezone.utc) - timedelta(seconds=5)).isoformat()
        scheduler._fire_due_schedules()

        # Should have created a task (fires normally)
        assert run_callback.call_count == 0  # callback is async, check task was created
        stored = scheduler.get("job1")
        assert stored.enabled

    @pytest.mark.asyncio
    async def test_max_runs_disables(self, scheduler, run_callback):
        entry = ScheduleEntry(
            id="job1", agent="bot", prompt="hi", schedule_type="cron", cron_expr="*/5 * * * *", max_runs=2
        )
        scheduler.add(entry)

        # Fire twice
        await scheduler._fire_schedule(scheduler.get("job1"))
        assert scheduler.get("job1").run_count == 1
        await scheduler._fire_schedule(scheduler.get("job1"))
        assert scheduler.get("job1").run_count == 2

        # Next fire_due_schedules should disable it
        scheduler.get("job1").next_run = (datetime.now(timezone.utc) - timedelta(seconds=5)).isoformat()
        scheduler._fire_due_schedules()
        stored = scheduler.get("job1")
        assert not stored.enabled
        assert stored.disabled_reason == "max_runs_reached"

    @pytest.mark.asyncio
    async def test_run_count_not_incremented_on_error(self, scheduler, run_callback):
        run_callback.side_effect = RuntimeError("boom")
        entry = ScheduleEntry(
            id="job1", agent="bot", prompt="hi", schedule_type="cron", cron_expr="*/5 * * * *", max_runs=2
        )
        scheduler.add(entry)
        await scheduler._fire_schedule(scheduler.get("job1"))

        assert scheduler.get("job1").run_count == 0
        assert scheduler.get("job1").last_status == "error"

    @pytest.mark.asyncio
    async def test_enable_clears_disabled_reason(self, scheduler):
        entry = ScheduleEntry(id="job1", agent="bot", prompt="hi", schedule_type="cron", cron_expr="*/5 * * * *")
        scheduler.add(entry)
        scheduler.get("job1").enabled = False
        scheduler.get("job1").disabled_reason = "expired"
        scheduler._save()

        scheduler.enable("job1")
        stored = scheduler.get("job1")
        assert stored.enabled
        assert stored.disabled_reason is None

    def test_expiry_fields_serialization_roundtrip(self, schedules_path, run_callback):
        sched1 = Scheduler(schedules_path, run_callback)
        expires = (datetime.now(timezone.utc) + timedelta(days=1)).isoformat()
        entry = ScheduleEntry(
            id="job1",
            agent="bot",
            prompt="hi",
            schedule_type="cron",
            cron_expr="*/5 * * * *",
            expires_at=expires,
            max_runs=10,
            run_count=3,
            disabled_reason=None,
        )
        sched1.add(entry)

        sched2 = Scheduler(schedules_path, run_callback)
        sched2._load()
        stored = sched2.get("job1")
        assert stored.expires_at == expires
        assert stored.max_runs == 10
        assert stored.run_count == 3


class TestAutoCleanup:
    def test_cleanup_removes_disabled_with_reason(self, scheduler):
        entry = ScheduleEntry(id="job1", agent="bot", prompt="hi", schedule_type="cron", cron_expr="*/5 * * * *")
        scheduler.add(entry)
        scheduler.get("job1").enabled = False
        scheduler.get("job1").disabled_reason = "expired"
        scheduler._save()

        removed = scheduler.cleanup()
        assert "job1" in removed

    def test_cleanup_skips_manually_disabled_cron(self, scheduler):
        entry = ScheduleEntry(id="job1", agent="bot", prompt="hi", schedule_type="cron", cron_expr="*/5 * * * *")
        scheduler.add(entry)
        scheduler.disable("job1")

        removed = scheduler.cleanup()
        assert removed == []
        assert len(scheduler.list()) == 1


class TestAgentSkippedError:
    @pytest.mark.asyncio
    async def test_skipped_sets_status(self, scheduler, run_callback):
        from tsugite.agent_runner.models import AgentSkippedError

        run_callback.side_effect = AgentSkippedError("run_if guard")
        entry = ScheduleEntry(id="job1", agent="bot", prompt="hi", schedule_type="cron", cron_expr="*/5 * * * *")
        scheduler.add(entry)
        await scheduler._fire_schedule(scheduler.get("job1"))

        stored = scheduler.get("job1")
        assert stored.last_status == "skipped"
        assert stored.last_error is None
        assert stored.run_count == 0  # skipped runs don't count


class TestSchedulerHardening:
    """Bug-hunt fixes: rejected updates must not poison state, garbage
    expires_at must not kill the fire loop, the min-interval guard applies on
    create, and missed one-offs are reaped instead of becoming zombies."""

    def test_rejected_update_leaves_entry_untouched(self, scheduler):
        entry = ScheduleEntry(id="tz", agent="bot", prompt="p", schedule_type="cron", cron_expr="0 9 * * *")
        scheduler.add(entry)
        with pytest.raises(ValueError):
            scheduler.update("tz", timezone="America/Chicgo")
        live = scheduler.get("tz")
        assert live.timezone != "America/Chicgo", "rejected update must not mutate the live entry"
        # And the scheduler must still compute next runs (state not poisoned).
        assert scheduler._compute_next_run_iso(live) is not None

    def test_add_rejects_invalid_timezone(self, scheduler):
        entry = ScheduleEntry(
            id="tz2", agent="bot", prompt="p", schedule_type="cron", cron_expr="0 9 * * *", timezone="Mars/Olympus"
        )
        with pytest.raises(ValueError, match="timezone"):
            scheduler.add(entry)
        assert "tz2" not in [e.id for e in scheduler.list()]

    def test_add_rejects_invalid_expires_at(self, scheduler):
        entry = ScheduleEntry(
            id="exp", agent="bot", prompt="p", schedule_type="cron", cron_expr="0 9 * * *", expires_at="next friday"
        )
        with pytest.raises(ValueError, match="expires_at"):
            scheduler.add(entry)

    def test_garbage_expires_at_on_disk_does_not_kill_fire_loop(self, scheduler):
        """Defense in depth: a legacy/hand-edited entry with a bad expires_at
        must be auto-disabled, not crash _fire_due_schedules (which killed the
        whole scheduler task)."""
        entry = ScheduleEntry(id="bad", agent="bot", prompt="p", schedule_type="cron", cron_expr="* * * * *")
        scheduler._schedules["bad"] = entry  # bypass add() validation, like a bad record on disk
        entry.expires_at = "next friday"
        entry.next_run = "2000-01-01T00:00:00+00:00"  # due now
        scheduler._fire_due_schedules()  # must not raise
        assert scheduler.get("bad").enabled is False
        assert "invalid" in (scheduler.get("bad").disabled_reason or "")

    def test_add_enforces_min_cron_interval(self, scheduler):
        entry = ScheduleEntry(id="fast", agent="bot", prompt="p", schedule_type="cron", cron_expr="* * * * *")
        with pytest.raises(ValueError, match="minimum interval"):
            scheduler.add(entry)

    def test_missed_once_within_grace_still_fires(self, scheduler):
        from datetime import datetime, timedelta
        from datetime import timezone as tz

        recent_past = (datetime.now(tz.utc) - timedelta(seconds=60)).isoformat()
        entry = ScheduleEntry(
            id="late", agent="bot", prompt="p", schedule_type="once", run_at=recent_past, misfire_grace_seconds=300
        )
        scheduler._schedules["late"] = entry
        next_run = scheduler._compute_next_run_iso(entry)
        assert next_run is not None, "a one-off within its misfire grace must remain fireable"

    def test_missed_once_past_grace_is_disabled_not_zombie(self, scheduler):
        """A one-off whose time passed beyond grace must get a disabled_reason
        so cleanup() reaps it - previously it stayed enabled forever with
        next_run=None and could never fire or be cleaned."""
        from datetime import datetime, timedelta
        from datetime import timezone as tz

        old = (datetime.now(tz.utc) - timedelta(hours=2)).isoformat()
        entry = ScheduleEntry(
            id="zombie", agent="bot", prompt="p", schedule_type="once", run_at=old, misfire_grace_seconds=300
        )
        scheduler._schedules["zombie"] = entry
        entry.next_run = old  # was armed when the daemon died
        scheduler._fire_due_schedules()
        z = scheduler.get("zombie")
        assert z.enabled is False
        assert z.disabled_reason, "missed one-off must carry a disabled_reason so cleanup() removes it"

    def test_arm_loaded_schedules_isolates_corrupt_entries_and_reaps_missed_once(self, scheduler):
        from datetime import datetime, timedelta
        from datetime import timezone as tz

        good = ScheduleEntry(id="good", agent="bot", prompt="p", schedule_type="cron", cron_expr="0 9 * * *")
        bad_tz = ScheduleEntry(id="badtz", agent="bot", prompt="p", schedule_type="cron", cron_expr="0 9 * * *")
        bad_tz.timezone = "Mars/Olympus"
        missed = ScheduleEntry(
            id="missed",
            agent="bot",
            prompt="p",
            schedule_type="once",
            run_at=(datetime.now(tz.utc) - timedelta(hours=3)).isoformat(),
        )
        for e in (good, bad_tz, missed):
            scheduler._schedules[e.id] = e  # as loaded from disk

        scheduler._arm_loaded_schedules()  # must not raise

        assert scheduler.get("good").next_run is not None
        assert scheduler.get("badtz").enabled is False
        assert "invalid" in scheduler.get("badtz").disabled_reason
        assert scheduler.get("missed").enabled is False
        assert "missed one-off" in scheduler.get("missed").disabled_reason
        # cleanup() can now reap both
        removed = scheduler.cleanup()
        assert set(removed) >= {"badtz", "missed"}


class TestRepeatedFailureNotification:
    """Schedules must not fail silently: notify once after N consecutive errors."""

    @pytest.fixture
    def failing_callback(self):
        cb = AsyncMock()
        # The real-world trigger: an agent-load error surfaces as a bare RuntimeError.
        cb.side_effect = RuntimeError("Failed to create tools: Invalid tool 'final_answer': not found")
        return cb

    def _entry(self, **kw):
        return ScheduleEntry(id="job1", agent="bot", prompt="hi", schedule_type="cron", cron_expr="0 9 * * *", **kw)

    @pytest.mark.asyncio
    async def test_notifies_once_at_threshold_then_suppresses_until_success(self, schedules_path, failing_callback):
        notified = []
        sched = Scheduler(schedules_path, failing_callback, on_repeated_failure=lambda e: notified.append(e.id))
        sched.add(self._entry(notify_on_failure=3))

        # Below threshold: count climbs, no notification yet.
        await sched._fire_schedule(sched.get("job1"))
        await sched._fire_schedule(sched.get("job1"))
        assert sched.get("job1").consecutive_failures == 2
        assert notified == []

        # Third consecutive failure crosses the threshold -> notify exactly once.
        await sched._fire_schedule(sched.get("job1"))
        assert sched.get("job1").consecutive_failures == 3
        assert notified == ["job1"]

        # Further failures are suppressed (no notification spam).
        await sched._fire_schedule(sched.get("job1"))
        assert notified == ["job1"]

        # A success resets the streak and re-arms the notification.
        failing_callback.side_effect = None
        failing_callback.return_value = RunResult(output="ok")
        await sched._fire_schedule(sched.get("job1"))
        assert sched.get("job1").consecutive_failures == 0
        assert sched.get("job1").failure_notified is False

        # Failing again to the threshold notifies a second time.
        failing_callback.side_effect = RuntimeError("boom")
        for _ in range(3):
            await sched._fire_schedule(sched.get("job1"))
        assert notified == ["job1", "job1"]

    @pytest.mark.asyncio
    async def test_notify_on_failure_zero_disables(self, schedules_path, failing_callback):
        notified = []
        sched = Scheduler(schedules_path, failing_callback, on_repeated_failure=lambda e: notified.append(e.id))
        sched.add(self._entry(notify_on_failure=0))
        for _ in range(5):
            await sched._fire_schedule(sched.get("job1"))
        assert notified == []
        assert sched.get("job1").consecutive_failures == 5


@pytest.mark.asyncio
async def test_overlapping_fire_does_not_corrupt_drift_metadata(scheduler):
    """When a fire is due but the previous run is still in progress, the overlap must be
    suppressed WITHOUT overwriting last_scheduled_for to the dropped fire's planned time
    (which would mislead the adapter's drift detection)."""
    scheduler.add(ScheduleEntry(id="job1", agent="bot", prompt="hi", schedule_type="cron", cron_expr="*/5 * * * *"))
    e = scheduler.get("job1")

    # Simulate the previous run still holding the lock.
    await e.lock.acquire()
    try:
        e.last_scheduled_for = "1999-01-01T00:00:00+00:00"  # the last run that actually happened
        e.last_run = "1999-01-01T00:00:01+00:00"
        e.next_run = (datetime.now(timezone.utc) - timedelta(seconds=60)).isoformat()  # due, within grace

        scheduler._fire_if_due(e, datetime.now(timezone.utc))
        await asyncio.sleep(0)  # flush any spawned task

        # Drift metadata preserved (not overwritten to the dropped fire) ...
        assert e.last_scheduled_for == "1999-01-01T00:00:00+00:00"
        assert e.last_run == "1999-01-01T00:00:01+00:00"
        # ... and next_run rolled forward so the loop doesn't busy-spin.
        assert e.next_run is not None
        assert datetime.fromisoformat(e.next_run) > datetime.now(timezone.utc)
    finally:
        e.lock.release()


class TestNotifyValidation:
    """Agents pass booleans and bare strings as `notify` in practice; those must
    produce a clear message or coerce, never "'bool' object is not iterable"."""

    def _validate(self, notify, notify_tool=False, channels={"discord", "web"}):
        from tsugite.tools import schedule as sched_tools

        old = sched_tools._channel_names
        sched_tools._channel_names = channels
        try:
            return sched_tools._validate_notify(notify, notify_tool)
        finally:
            sched_tools._channel_names = old

    def test_notify_true_gives_clear_error(self):
        with pytest.raises(ValueError, match="notify must be a list.*got True.*discord"):
            self._validate(True)

    def test_notify_false_means_no_notify(self):
        assert self._validate(False) is None

    def test_notify_bare_string_coerced_to_list(self):
        assert self._validate("discord") == ["discord"]

    def test_notify_other_type_gives_clear_error(self):
        with pytest.raises(ValueError, match="notify must be a list.*got int"):
            self._validate(3)

    def test_notify_list_passes_through(self):
        assert self._validate(["discord"]) == ["discord"]

    def test_unknown_channel_still_rejected(self):
        with pytest.raises(ValueError, match="Unknown notification channel"):
            self._validate(["nope"])

    def test_notify_tool_still_requires_notify(self):
        with pytest.raises(ValueError, match="notify_tool=True requires"):
            self._validate(None, notify_tool=True)
