"""Tests for background task status tracking."""

import asyncio
from threading import Thread
from unittest.mock import MagicMock

import pytest

from tsugite.daemon.adapters.base import ChannelContext
from tsugite.daemon.scheduler import ScheduleEntry, Scheduler


@pytest.fixture
def schedules_path(tmp_path):
    return tmp_path / "schedules.json"


@pytest.fixture
def scheduler(schedules_path):
    async def noop(_entry):
        return "done"

    return Scheduler(schedules_path, noop)


@pytest.fixture
def tool_loop():
    from tsugite.tools.schedule import set_scheduler

    loop = asyncio.new_event_loop()
    t = Thread(target=loop.run_forever, daemon=True)
    t.start()
    yield loop
    loop.call_soon_threadsafe(loop.stop)
    t.join(timeout=2)
    loop.close()
    set_scheduler(None)


class TestGetRunningIds:
    def test_empty_when_nothing_running(self, scheduler):
        assert scheduler.get_running_ids() == []

    def test_empty_with_schedules_not_running(self, scheduler):
        entry = ScheduleEntry(id="job1", agent="bot", prompt="hi", schedule_type="cron", cron_expr="* * * * *")
        scheduler.add(entry)
        assert scheduler.get_running_ids() == []

    @pytest.mark.asyncio
    async def test_returns_ids_while_lock_held(self, scheduler):
        entry = ScheduleEntry(id="job1", agent="bot", prompt="hi", schedule_type="cron", cron_expr="* * * * *")
        scheduler.add(entry)

        lock = scheduler._entry_locks.setdefault("job1", asyncio.Lock())
        await lock.acquire()
        try:
            assert scheduler.get_running_ids() == ["job1"]
        finally:
            lock.release()

        assert scheduler.get_running_ids() == []


class TestScheduleStatusTool:
    def test_returns_correct_fields(self, tool_loop):
        from tsugite.tools.schedule import schedule_status, set_scheduler

        mock_sched = MagicMock()
        entry = ScheduleEntry(
            id="job1", agent="bot", prompt="hello", schedule_type="cron", cron_expr="0 9 * * *",
            last_status="success", last_run="2026-01-01T00:00:00Z",
        )
        mock_sched.get.return_value = entry
        mock_sched.get_running_ids.return_value = []
        set_scheduler(mock_sched, tool_loop)

        result = schedule_status(id="job1")
        assert result["id"] == "job1"
        assert result["agent"] == "bot"
        assert result["is_running"] is False
        assert result["last_status"] == "success"
        assert result["enabled"] is True

    def test_shows_running_true(self, tool_loop):
        from tsugite.tools.schedule import schedule_status, set_scheduler

        mock_sched = MagicMock()
        entry = ScheduleEntry(id="job1", agent="bot", prompt="hi", schedule_type="cron", cron_expr="0 9 * * *")
        mock_sched.get.return_value = entry
        mock_sched.get_running_ids.return_value = ["job1"]
        set_scheduler(mock_sched, tool_loop)

        result = schedule_status(id="job1")
        assert result["is_running"] is True


class TestListRunningTasksTool:
    def test_returns_running_entries(self, tool_loop):
        from tsugite.tools.schedule import list_running_tasks, set_scheduler

        mock_sched = MagicMock()
        entry = ScheduleEntry(id="bg-abc", agent="bot", prompt="do stuff", schedule_type="once", run_at="2099-01-01T00:00:00Z")
        mock_sched.get_running_ids.return_value = ["bg-abc"]
        mock_sched.get.return_value = entry
        set_scheduler(mock_sched, tool_loop)

        result = list_running_tasks()
        assert len(result) == 1
        assert result[0]["id"] == "bg-abc"
        assert result[0]["agent"] == "bot"
        assert result[0]["prompt"] == "do stuff"

    def test_empty_when_nothing_running(self, tool_loop):
        from tsugite.tools.schedule import list_running_tasks, set_scheduler

        mock_sched = MagicMock()
        mock_sched.get_running_ids.return_value = []
        set_scheduler(mock_sched, tool_loop)

        assert list_running_tasks() == []

    def test_truncates_long_prompts(self, tool_loop):
        from tsugite.tools.schedule import list_running_tasks, set_scheduler

        mock_sched = MagicMock()
        entry = ScheduleEntry(id="bg-long", agent="bot", prompt="x" * 500, schedule_type="once", run_at="2099-01-01T00:00:00Z")
        mock_sched.get_running_ids.return_value = ["bg-long"]
        mock_sched.get.return_value = entry
        set_scheduler(mock_sched, tool_loop)

        result = list_running_tasks()
        assert len(result[0]["prompt"]) == 200


class TestAgentContextRunningTasks:
    def test_includes_running_tasks_from_metadata(self):
        from tsugite.daemon.adapters.base import BaseAdapter

        ctx = ChannelContext(
            source="scheduler", channel_id=None, user_id="scheduler:bot",
            reply_to="scheduler:bot", metadata={"schedule_id": "job1", "running_tasks": ["job1", "job2"]},
        )
        result = BaseAdapter._build_agent_context(None, ctx)
        assert result["running_tasks"] == ["job1", "job2"]

    def test_defaults_to_empty_list(self):
        from tsugite.daemon.adapters.base import BaseAdapter

        ctx = ChannelContext(
            source="cli", channel_id=None, user_id="user1", reply_to="user1",
        )
        result = BaseAdapter._build_agent_context(None, ctx)
        assert result["running_tasks"] == []

    def test_defaults_when_no_metadata(self):
        from tsugite.daemon.adapters.base import BaseAdapter

        ctx = ChannelContext(
            source="cli", channel_id=None, user_id="user1", reply_to="user1", metadata=None,
        )
        result = BaseAdapter._build_agent_context(None, ctx)
        assert result["running_tasks"] == []
