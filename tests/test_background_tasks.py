"""Tests for background task tools and auto-reply."""

import asyncio
from pathlib import Path
from threading import Thread
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tsugite.daemon.adapters.scheduler_adapter import SchedulerAdapter
from tsugite.daemon.config import NotificationChannelConfig
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


@pytest.fixture
def tool_loop():
    """Provide a background event loop for testing schedule tools thread-safely."""
    from tsugite.tools.schedule import set_scheduler

    loop = asyncio.new_event_loop()
    t = Thread(target=loop.run_forever, daemon=True)
    t.start()
    yield loop
    loop.call_soon_threadsafe(loop.stop)
    t.join(timeout=2)
    loop.close()
    set_scheduler(None)


class TestFireNow:
    @pytest.mark.asyncio
    async def test_fires_enabled_entry(self, scheduler, run_callback):
        entry = ScheduleEntry(id="job1", agent="bot", prompt="hi", schedule_type="cron", cron_expr="* * * * *")
        scheduler.add(entry)
        scheduler.fire_now("job1")

        # Wait for the background task to complete
        await asyncio.gather(*scheduler._active_tasks)

        run_callback.assert_awaited_once()
        assert run_callback.call_args[0][0].id == "job1"

    def test_rejects_disabled_entry(self, scheduler):
        entry = ScheduleEntry(id="job1", agent="bot", prompt="hi", schedule_type="cron", cron_expr="* * * * *")
        scheduler.add(entry)
        scheduler.disable("job1")
        with pytest.raises(ValueError, match="disabled"):
            scheduler.fire_now("job1")

    def test_rejects_nonexistent(self, scheduler):
        with pytest.raises(ValueError, match="not found"):
            scheduler.fire_now("nope")


class TestAutoReplyField:
    def test_defaults_to_false(self):
        entry = ScheduleEntry(id="t", agent="bot", prompt="hi", schedule_type="cron", cron_expr="0 9 * * *")
        assert entry.auto_reply is False

    def test_explicit_true(self):
        entry = ScheduleEntry(
            id="t", agent="bot", prompt="hi", schedule_type="cron", cron_expr="0 9 * * *", auto_reply=True
        )
        assert entry.auto_reply is True

    def test_serialization_roundtrip(self, schedules_path, run_callback):
        sched1 = Scheduler(schedules_path, run_callback)
        entry = ScheduleEntry(
            id="j1", agent="bot", prompt="hi", schedule_type="cron", cron_expr="0 9 * * *", auto_reply=True
        )
        sched1.add(entry)

        sched2 = Scheduler(schedules_path, run_callback)
        sched2._load()
        assert sched2.get("j1").auto_reply is True

    def test_old_schedules_default(self):
        """Schedules saved before auto_reply existed should default to False."""
        from dataclasses import asdict

        entry = ScheduleEntry(id="t", agent="bot", prompt="hi", schedule_type="cron", cron_expr="0 9 * * *")
        data = asdict(entry)
        del data["auto_reply"]
        restored = ScheduleEntry(**data)
        assert restored.auto_reply is False


class TestScheduleRunTool:
    def test_calls_fire_now(self, tool_loop):
        from tsugite.tools.schedule import schedule_run, set_scheduler

        mock_sched = MagicMock()
        set_scheduler(mock_sched, tool_loop)

        result = schedule_run(id="job1")
        assert result == {"status": "triggered", "id": "job1"}
        mock_sched.fire_now.assert_called_once_with("job1")


class TestBackgroundTaskTool:
    def test_creates_and_fires(self, tool_loop):
        from tsugite.tools.schedule import background_task, set_scheduler

        mock_sched = MagicMock()
        set_scheduler(mock_sched, tool_loop, channel_names={"my-discord"})

        with patch("tsugite.agent_runner.helpers.get_current_agent", return_value="bot"):
            result = background_task(prompt="list files", notify=["my-discord"])

        assert result["status"] == "started"
        assert result["id"].startswith("bg-")

        mock_sched.add.assert_called_once()
        added_entry = mock_sched.add.call_args[0][0]
        assert added_entry.prompt == "list files"
        assert added_entry.auto_reply is True
        assert added_entry.schedule_type == "once"
        assert added_entry.agent == "bot"

        mock_sched.fire_now.assert_called_once_with(result["id"])

    def test_rejects_unknown_channel(self, tool_loop):
        from tsugite.tools.schedule import background_task, set_scheduler

        mock_sched = MagicMock()
        set_scheduler(mock_sched, tool_loop, channel_names={"discord-dm"})

        with pytest.raises(ValueError, match="Unknown notification"):
            background_task(prompt="test", notify=["nonexistent"])


def _make_discord_channel(user_id="123456789", bot="my-bot") -> NotificationChannelConfig:
    return NotificationChannelConfig(type="discord", user_id=user_id, bot=bot)


def _make_scheduler_adapter(identity_map=None, notification_channels=None) -> tuple[SchedulerAdapter, MagicMock]:
    mock_adapter = MagicMock()
    mock_adapter.agent_name = "bot"
    mock_adapter.handle_message = AsyncMock(return_value="Here's a summary of the results.")
    return (
        SchedulerAdapter(
            adapters={"bot": mock_adapter},
            schedules_path=Path("/tmp/test-schedules.json"),
            notification_channels=notification_channels or {},
            identity_map=identity_map or {},
        ),
        mock_adapter,
    )


class TestAutoReply:
    @pytest.mark.asyncio
    async def test_auto_reply_calls_handle_message_on_user_session(self):
        sa, mock_adapter = _make_scheduler_adapter(
            identity_map={"discord:123456789": "justyn"},
            notification_channels={"dm": _make_discord_channel()},
        )

        entry = ScheduleEntry(
            id="bg-test", agent="bot", prompt="list files", schedule_type="once",
            run_at="2099-01-01T00:00:00Z", notify=["dm"], auto_reply=True,
        )

        with patch("tsugite.daemon.adapters.scheduler_adapter.send_notification"):
            await sa._auto_reply(mock_adapter, entry, "file1.txt\nfile2.txt", [("dm", _make_discord_channel())])

        mock_adapter.handle_message.assert_awaited_once()
        call_kwargs = mock_adapter.handle_message.call_args[1]
        assert call_kwargs["user_id"] == "justyn"
        assert "background_task" in call_kwargs["message"]
        assert "file1.txt" in call_kwargs["message"]
        assert call_kwargs["channel_context"].source == "background_task"

    @pytest.mark.asyncio
    async def test_auto_reply_skips_inject_history(self):
        """When auto_reply=True, _inject_into_user_sessions should NOT be called."""
        sa, mock_adapter = _make_scheduler_adapter(
            identity_map={"discord:123456789": "justyn"},
            notification_channels={"dm": _make_discord_channel()},
        )

        entry = ScheduleEntry(
            id="bg-test", agent="bot", prompt="test", schedule_type="once",
            run_at="2099-01-01T00:00:00Z", notify=["dm"], auto_reply=True,
            inject_history=True,
        )

        with (
            patch("tsugite.daemon.adapters.scheduler_adapter.send_notification"),
            patch("tsugite.interaction.set_interaction_backend"),
            patch.object(sa, "_inject_into_user_sessions") as mock_inject,
            patch.object(sa, "_auto_reply", new_callable=AsyncMock) as mock_auto,
        ):
            await sa._run_agent(entry)

        mock_auto.assert_awaited_once()
        mock_inject.assert_not_called()

    @pytest.mark.asyncio
    async def test_regular_schedule_uses_old_behavior(self):
        """auto_reply=False should use notification + inject_history as before."""
        sa, mock_adapter = _make_scheduler_adapter(
            notification_channels={"dm": _make_discord_channel()},
        )

        entry = ScheduleEntry(
            id="cron-job", agent="bot", prompt="test", schedule_type="cron",
            cron_expr="0 9 * * *", notify=["dm"], auto_reply=False, inject_history=True,
        )

        with (
            patch("tsugite.daemon.adapters.scheduler_adapter.send_notification") as mock_notify,
            patch("tsugite.interaction.set_interaction_backend"),
            patch.object(sa, "_inject_into_user_sessions") as mock_inject,
            patch.object(sa, "_auto_reply") as mock_auto,
        ):
            await sa._run_agent(entry)

        mock_auto.assert_not_called()
        mock_notify.assert_called_once()
        mock_inject.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_auto_reply_falls_back_on_error(self):
        """If handle_message fails, fall back to raw notification."""
        sa, mock_adapter = _make_scheduler_adapter(
            identity_map={"discord:123456789": "justyn"},
        )
        mock_adapter.handle_message = AsyncMock(side_effect=RuntimeError("agent crashed"))

        entry = ScheduleEntry(
            id="bg-fail", agent="bot", prompt="test", schedule_type="once",
            run_at="2099-01-01T00:00:00Z", notify=["dm"], auto_reply=True,
        )

        with patch("tsugite.daemon.adapters.scheduler_adapter.send_notification") as mock_notify:
            await sa._auto_reply(mock_adapter, entry, "raw result", [("dm", _make_discord_channel())])

        # Fallback notification should have been attempted
        assert mock_notify.call_count >= 1
