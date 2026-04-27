"""Tests for background task tools and auto-reply."""

import asyncio
from pathlib import Path
from threading import Thread
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tsugite.daemon.adapters.scheduler_adapter import SchedulerAdapter
from tsugite.daemon.config import NotificationChannelConfig
from tsugite.daemon.scheduler import RunResult, ScheduleEntry, Scheduler
from tsugite.exceptions import AgentExecutionError


@pytest.fixture
def schedules_path(tmp_path):
    return tmp_path / "schedules.json"


@pytest.fixture
def run_callback():
    return AsyncMock(return_value=RunResult(output="done"))


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


class TestScheduleCreateValidation:
    def test_rejects_unknown_agent(self, tool_loop):
        from tsugite.tools.schedule import schedule_create, set_scheduler

        mock_sched = MagicMock()
        set_scheduler(mock_sched, tool_loop, agent_names={"bot"})

        with pytest.raises(ValueError, match="Unknown agent 'nonexistent'"):
            schedule_create(id="test", prompt="hi", agent="nonexistent", cron="0 9 * * *")


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
        set_scheduler(mock_sched, tool_loop, channel_names={"my-discord"}, agent_names={"bot"})

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
        set_scheduler(mock_sched, tool_loop, channel_names={"discord-dm"}, agent_names={"bot"})

        with pytest.raises(ValueError, match="Unknown notification"):
            background_task(prompt="test", notify=["nonexistent"])

    def test_rejects_unknown_agent(self, tool_loop):
        from tsugite.tools.schedule import background_task, set_scheduler

        mock_sched = MagicMock()
        set_scheduler(mock_sched, tool_loop, agent_names={"bot"})

        with pytest.raises(ValueError, match="Unknown agent 'nonexistent'"):
            background_task(prompt="test", agent="nonexistent")


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
            id="bg-test",
            agent="bot",
            prompt="list files",
            schedule_type="once",
            run_at="2099-01-01T00:00:00Z",
            notify=["dm"],
            auto_reply=True,
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
            id="bg-test",
            agent="bot",
            prompt="test",
            schedule_type="once",
            run_at="2099-01-01T00:00:00Z",
            notify=["dm"],
            auto_reply=True,
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
            id="cron-job",
            agent="bot",
            prompt="test",
            schedule_type="cron",
            cron_expr="0 9 * * *",
            notify=["dm"],
            auto_reply=False,
            inject_history=True,
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
            id="bg-fail",
            agent="bot",
            prompt="test",
            schedule_type="once",
            run_at="2099-01-01T00:00:00Z",
            notify=["dm"],
            auto_reply=True,
        )

        with patch("tsugite.daemon.adapters.scheduler_adapter.send_notification") as mock_notify:
            await sa._auto_reply(mock_adapter, entry, "raw result", [("dm", _make_discord_channel())])

        # Fallback notification should have been attempted
        assert mock_notify.call_count >= 1


class TestAgentExecutionErrorNotification:
    @pytest.mark.asyncio
    async def test_sends_failure_notification_on_agent_execution_error(self):
        sa, mock_adapter = _make_scheduler_adapter(
            notification_channels={"dm": _make_discord_channel()},
        )
        mock_adapter.handle_message = AsyncMock(side_effect=AgentExecutionError("Agent reached max_turns (5)"))

        entry = ScheduleEntry(
            id="bg-fail",
            agent="bot",
            prompt="do something",
            schedule_type="once",
            run_at="2099-01-01T00:00:00Z",
            notify=["dm"],
        )

        with (
            patch("tsugite.daemon.adapters.scheduler_adapter.send_notification") as mock_notify,
            patch("tsugite.interaction.set_interaction_backend"),
        ):
            with pytest.raises(AgentExecutionError):
                await sa._run_agent(entry)

        mock_notify.assert_called_once()
        notification_text = mock_notify.call_args[0][0]
        assert "failed" in notification_text
        assert "bg-fail" in notification_text

    @pytest.mark.asyncio
    async def test_reraises_after_notification(self):
        sa, mock_adapter = _make_scheduler_adapter(
            notification_channels={"dm": _make_discord_channel()},
        )
        mock_adapter.handle_message = AsyncMock(side_effect=AgentExecutionError("Agent reached max_turns (5)"))

        entry = ScheduleEntry(
            id="bg-fail",
            agent="bot",
            prompt="do something",
            schedule_type="once",
            run_at="2099-01-01T00:00:00Z",
            notify=["dm"],
        )

        with (
            patch("tsugite.daemon.adapters.scheduler_adapter.send_notification"),
            patch("tsugite.interaction.set_interaction_backend"),
        ):
            with pytest.raises(AgentExecutionError, match="max_turns"):
                await sa._run_agent(entry)

    @pytest.mark.asyncio
    async def test_no_notification_when_no_channels(self):
        sa, mock_adapter = _make_scheduler_adapter()
        mock_adapter.handle_message = AsyncMock(side_effect=AgentExecutionError("Agent reached max_turns (5)"))

        entry = ScheduleEntry(
            id="bg-fail",
            agent="bot",
            prompt="do something",
            schedule_type="once",
            run_at="2099-01-01T00:00:00Z",
        )

        with (
            patch("tsugite.daemon.adapters.scheduler_adapter.send_notification") as mock_notify,
            patch("tsugite.interaction.set_interaction_backend"),
        ):
            with pytest.raises(AgentExecutionError):
                await sa._run_agent(entry)

        mock_notify.assert_not_called()


class TestBackgroundTaskMaxTurns:
    def test_passes_max_turns_to_entry(self, tool_loop):
        from tsugite.tools.schedule import background_task, set_scheduler

        mock_sched = MagicMock()
        set_scheduler(mock_sched, tool_loop, channel_names={"my-discord"}, agent_names={"bot"})

        with patch("tsugite.agent_runner.helpers.get_current_agent", return_value="bot"):
            result = background_task(prompt="test", notify=["my-discord"], max_turns=5)

        assert result["status"] == "started"
        added_entry = mock_sched.add.call_args[0][0]
        assert added_entry.max_turns == 5

    def test_max_turns_defaults_to_none(self, tool_loop):
        from tsugite.tools.schedule import background_task, set_scheduler

        mock_sched = MagicMock()
        set_scheduler(mock_sched, tool_loop, channel_names={"my-discord"}, agent_names={"bot"})

        with patch("tsugite.agent_runner.helpers.get_current_agent", return_value="bot"):
            background_task(prompt="test", notify=["my-discord"])

        added_entry = mock_sched.add.call_args[0][0]
        assert added_entry.max_turns is None


class TestPartialOutputOnMaxTurns:
    def test_agent_execution_error_carries_partial_output(self):
        err = AgentExecutionError(
            "Agent reached max_turns",
            partial_output="I was analyzing the files when I ran out of turns",
        )
        assert err.partial_output == "I was analyzing the files when I ran out of turns"

    def test_agent_execution_error_partial_output_defaults_none(self):
        err = AgentExecutionError("Agent reached max_turns")
        assert err.partial_output is None

    @pytest.mark.asyncio
    async def test_failure_notification_includes_partial_output(self):
        sa, mock_adapter = _make_scheduler_adapter(
            notification_channels={"dm": _make_discord_channel()},
        )
        mock_adapter.handle_message = AsyncMock(
            side_effect=AgentExecutionError(
                "Agent reached max_turns (5)",
                partial_output="Partial work done here",
            )
        )

        entry = ScheduleEntry(
            id="bg-partial",
            agent="bot",
            prompt="do something",
            schedule_type="once",
            run_at="2099-01-01T00:00:00Z",
            notify=["dm"],
        )

        with (
            patch("tsugite.daemon.adapters.scheduler_adapter.send_notification") as mock_notify,
            patch("tsugite.interaction.set_interaction_backend"),
        ):
            with pytest.raises(AgentExecutionError):
                await sa._run_agent(entry)

        mock_notify.assert_called_once()
        notification_text = mock_notify.call_args[0][0]
        assert "Partial output" in notification_text
        assert "Partial work done here" in notification_text

    @pytest.mark.asyncio
    async def test_failure_notification_without_partial_output(self):
        sa, mock_adapter = _make_scheduler_adapter(
            notification_channels={"dm": _make_discord_channel()},
        )
        mock_adapter.handle_message = AsyncMock(side_effect=AgentExecutionError("Agent reached max_turns (5)"))

        entry = ScheduleEntry(
            id="bg-no-partial",
            agent="bot",
            prompt="do something",
            schedule_type="once",
            run_at="2099-01-01T00:00:00Z",
            notify=["dm"],
        )

        with (
            patch("tsugite.daemon.adapters.scheduler_adapter.send_notification") as mock_notify,
            patch("tsugite.interaction.set_interaction_backend"),
        ):
            with pytest.raises(AgentExecutionError):
                await sa._run_agent(entry)

        mock_notify.assert_called_once()
        notification_text = mock_notify.call_args[0][0]
        assert "Partial output" not in notification_text


class TestSchedulerStatusUpdates:
    """The scheduled-task session must reach a terminal status on every exit path,
    so the web UI sidebar doesn't leave the card pinned at "Starting...".
    """

    def _last_status(self, session_store_mock):
        """Pull the `status` kwarg from the most recent update_session call, or None."""
        for call in reversed(session_store_mock.update_session.call_args_list):
            if "status" in call.kwargs:
                return call.kwargs["status"]
        return None

    @pytest.mark.asyncio
    async def test_success_marks_session_completed(self):
        from tsugite.daemon.session_store import SessionStatus

        sa, mock_adapter = _make_scheduler_adapter()
        mock_adapter.handle_message = AsyncMock(return_value="all done")

        entry = ScheduleEntry(
            id="cron-ok",
            agent="bot",
            prompt="do work",
            schedule_type="cron",
            cron_expr="0 9 * * *",
        )

        with patch("tsugite.interaction.set_interaction_backend"):
            await sa._run_agent(entry)

        assert self._last_status(mock_adapter.session_store) == SessionStatus.COMPLETED.value

    @pytest.mark.asyncio
    async def test_agent_skipped_marks_session_cancelled(self):
        """AgentSkippedError currently re-raises without updating status, leaving the session pinned at RUNNING."""
        from tsugite.agent_runner.models import AgentSkippedError
        from tsugite.daemon.session_store import SessionStatus

        sa, mock_adapter = _make_scheduler_adapter()
        mock_adapter.handle_message = AsyncMock(side_effect=AgentSkippedError("conditions not met"))

        entry = ScheduleEntry(
            id="cron-skipped",
            agent="bot",
            prompt="maybe do work",
            schedule_type="cron",
            cron_expr="0 9 * * *",
        )

        with patch("tsugite.interaction.set_interaction_backend"):
            with pytest.raises(AgentSkippedError):
                await sa._run_agent(entry)

        assert self._last_status(mock_adapter.session_store) == SessionStatus.CANCELLED.value

    @pytest.mark.asyncio
    async def test_unexpected_exception_marks_session_failed(self):
        """Any non-AgentExecutionError that escapes handle_message must still set FAILED."""
        from tsugite.daemon.session_store import SessionStatus

        sa, mock_adapter = _make_scheduler_adapter()
        mock_adapter.handle_message = AsyncMock(side_effect=RuntimeError("kaboom"))

        entry = ScheduleEntry(
            id="cron-boom",
            agent="bot",
            prompt="do work",
            schedule_type="cron",
            cron_expr="0 9 * * *",
        )

        with patch("tsugite.interaction.set_interaction_backend"):
            with pytest.raises(RuntimeError, match="kaboom"):
                await sa._run_agent(entry)

        assert self._last_status(mock_adapter.session_store) == SessionStatus.FAILED.value

    @pytest.mark.asyncio
    async def test_update_failure_logs_warning(self, caplog):
        """A ValueError from session_store.update_session should be logged, not silently swallowed."""
        import logging

        sa, mock_adapter = _make_scheduler_adapter()
        mock_adapter.handle_message = AsyncMock(return_value="fine")
        mock_adapter.session_store.update_session = MagicMock(side_effect=ValueError("session vanished"))

        entry = ScheduleEntry(
            id="cron-update-fail",
            agent="bot",
            prompt="do work",
            schedule_type="cron",
            cron_expr="0 9 * * *",
        )

        with patch("tsugite.interaction.set_interaction_backend"):
            with caplog.at_level(logging.WARNING, logger="tsugite.daemon.adapters.scheduler_adapter"):
                await sa._run_agent(entry)

        assert any(
            "session vanished" in rec.getMessage() and "cron-update-fail" in rec.getMessage() for rec in caplog.records
        ), f"expected warning mentioning the schedule id and ValueError, got: {[r.getMessage() for r in caplog.records]}"


class TestPartialHistoryOnError:
    @pytest.mark.asyncio
    async def test_saves_history_on_agent_execution_error(self, tmp_path):
        from tsugite.daemon.adapters.base import BaseAdapter, ChannelContext

        mock_adapter = MagicMock(spec=BaseAdapter)
        mock_adapter.agent_name = "bot"
        mock_adapter.agent_config = MagicMock()
        mock_adapter.agent_config.model = "test-model"
        mock_adapter.agent_config.max_turns = None
        mock_adapter.agent_config.workspace = None
        mock_adapter.agent_config.workspace_dir = tmp_path
        mock_adapter.session_store = MagicMock()
        mock_adapter.session_store.needs_compaction.return_value = False
        mock_session = MagicMock()
        mock_session.id = "conv-123"
        mock_adapter.session_store.get_or_create_interactive.return_value = mock_session
        mock_adapter.resolve_user = MagicMock(return_value="test-user")
        mock_adapter._resolve_agent_path = MagicMock(return_value=Path("/fake/agent.yaml"))
        mock_adapter._build_message_context = MagicMock(return_value="test prompt")
        mock_adapter._build_agent_context = MagicMock(return_value={})
        mock_adapter._get_workspace_attachments = MagicMock(return_value=[])
        mock_adapter._emit_ui = MagicMock()
        mock_adapter._identity_map = {}

        channel_ctx = ChannelContext(source="test", channel_id="ch1", user_id="test-user", reply_to="test:test-user")

        error = AgentExecutionError(
            "Agent reached max_turns (5)",
            partial_output="I found 3 files",
            token_usage=100,
            cost=0.05,
            execution_steps=[{"step": 1}],
        )

        with (
            patch("tsugite.daemon.adapters.base.run_agent", side_effect=error),
            patch("tsugite.agent_runner.history_integration.save_run_to_history") as mock_save,
        ):
            mock_adapter.resolve_model = MagicMock(return_value="test-model")
            mock_adapter._save_history = BaseAdapter._save_history.__get__(mock_adapter, BaseAdapter)
            with pytest.raises(AgentExecutionError):
                await BaseAdapter.handle_message(
                    mock_adapter,
                    user_id="test-user",
                    message="test prompt",
                    channel_context=channel_ctx,
                )

        mock_save.assert_called_once()
        call_kwargs = mock_save.call_args[1]
        assert "[Error:" in call_kwargs["result"]
        assert "I found 3 files" in call_kwargs["result"]
        assert call_kwargs["token_count"] == 100
        assert call_kwargs["cost"] == 0.05
