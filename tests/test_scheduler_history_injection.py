"""Tests for scheduled task history injection into user sessions."""

import json
from dataclasses import asdict
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tsugite.daemon.adapters.scheduler_adapter import SchedulerAdapter
from tsugite.daemon.config import NotificationChannelConfig
from tsugite.daemon.scheduler import ScheduleEntry


def _make_entry(**kwargs) -> ScheduleEntry:
    defaults = dict(
        id="test-job", agent="bot", prompt="do something", schedule_type="cron", cron_expr="0 9 * * *"
    )
    defaults.update(kwargs)
    return ScheduleEntry(**defaults)


def _make_discord_channel(user_id="123456789", bot="my-bot") -> NotificationChannelConfig:
    return NotificationChannelConfig(type="discord", user_id=user_id, bot=bot)


def _make_webhook_channel() -> NotificationChannelConfig:
    return NotificationChannelConfig(type="webhook", url="https://example.com/hook")


class TestInjectHistoryField:
    def test_defaults_to_true(self):
        entry = _make_entry()
        assert entry.inject_history is True

    def test_explicit_false(self):
        entry = _make_entry(inject_history=False)
        assert entry.inject_history is False

    def test_serialization_roundtrip(self):
        entry = _make_entry(inject_history=False)
        data = asdict(entry)
        assert data["inject_history"] is False
        restored = ScheduleEntry(**data)
        assert restored.inject_history is False

    def test_old_schedules_default(self):
        """Schedules saved before this field existed should default to True."""
        data = asdict(_make_entry())
        del data["inject_history"]
        entry = ScheduleEntry(**data)
        assert entry.inject_history is True


class TestRecordSyntheticTurn:
    @staticmethod
    def _mock_adapter(session_id: str = "test-session") -> MagicMock:
        adapter = MagicMock()
        adapter.agent_name = "bot"
        adapter.resolve_model.return_value = "test-model"
        adapter.session_manager.get_or_create_session.return_value = session_id
        return adapter

    def _record_and_load(self, tmp_path, session_id, result):
        """Run _record_synthetic_turn and return parsed JSONL records."""
        mock_adapter = self._mock_adapter(session_id)
        session_path = tmp_path / "history" / f"{session_id}.jsonl"
        session_path.parent.mkdir(parents=True, exist_ok=True)

        with patch("tsugite.history.get_history_dir", return_value=tmp_path / "history"):
            SchedulerAdapter._record_synthetic_turn(mock_adapter, "justyn", _make_entry(), result)

        assert session_path.exists()
        return [json.loads(line) for line in session_path.read_text().strip().split("\n")]

    def test_writes_correct_turn_format(self, tmp_path):
        records = self._record_and_load(tmp_path, "test-session", "Task completed successfully")
        turn_record = next(r for r in records if r.get("type") == "turn")

        assert turn_record["metadata"]["synthetic"] is True
        assert turn_record["metadata"]["schedule_id"] == "test-job"
        assert len(turn_record["messages"]) == 2
        assert turn_record["messages"][0]["role"] == "user"
        assert '<scheduled_task id="test-job">' in turn_record["messages"][0]["content"]
        assert turn_record["messages"][1]["role"] == "assistant"
        assert turn_record["messages"][1]["content"] == "Task completed successfully"
        assert turn_record["final_answer"] == "Task completed successfully"

    def test_stores_result_as_is(self, tmp_path):
        """_record_synthetic_turn stores the result verbatim (caller truncates)."""
        result = "pre-truncated result"
        records = self._record_and_load(tmp_path, "trunc-session", result)
        turn_record = next(r for r in records if r.get("type") == "turn")

        assert turn_record["messages"][1]["content"] == result
        assert turn_record["final_answer"] == result


def _make_scheduler_adapter(identity_map=None, notification_channels=None) -> tuple[SchedulerAdapter, MagicMock]:
    """Create a SchedulerAdapter with a mock agent adapter."""
    mock_adapter = MagicMock()
    mock_adapter.agent_name = "bot"
    return (
        SchedulerAdapter(
            adapters={"bot": mock_adapter},
            schedules_path=Path("/tmp/test-schedules.json"),
            notification_channels=notification_channels or {},
            identity_map=identity_map or {},
        ),
        mock_adapter,
    )


class TestInjectIntoUserSessions:
    @pytest.mark.asyncio
    async def test_resolves_discord_identity(self):
        sa, mock_adapter = _make_scheduler_adapter(identity_map={"discord:123456789": "justyn"})
        entry = _make_entry()

        with patch.object(sa, "_record_synthetic_turn") as mock_record:
            await sa._inject_into_user_sessions(mock_adapter, entry, "result", [("dm", _make_discord_channel())])

        mock_record.assert_called_once()
        call_args = mock_record.call_args[0]
        assert call_args[0] is mock_adapter
        assert call_args[1] == "justyn"
        assert call_args[2] is entry
        assert call_args[3] == "result"

    @pytest.mark.asyncio
    async def test_falls_back_to_raw_id(self):
        sa, mock_adapter = _make_scheduler_adapter()
        entry = _make_entry()

        with patch.object(sa, "_record_synthetic_turn") as mock_record:
            await sa._inject_into_user_sessions(
                mock_adapter, entry, "result", [("dm", _make_discord_channel(user_id="999"))]
            )

        assert mock_record.call_args[0][1] == "999"

    @pytest.mark.asyncio
    async def test_skips_webhook_channels(self):
        sa, mock_adapter = _make_scheduler_adapter()

        with patch.object(sa, "_record_synthetic_turn") as mock_record:
            await sa._inject_into_user_sessions(
                mock_adapter, _make_entry(), "result", [("hook", _make_webhook_channel())]
            )

        mock_record.assert_not_called()

    @pytest.mark.asyncio
    async def test_inject_history_false_guard_in_run_agent(self):
        """inject_history=False prevents _inject_into_user_sessions from being called."""
        sa, mock_adapter = _make_scheduler_adapter(
            notification_channels={"dm": _make_discord_channel()},
            identity_map={"discord:123456789": "justyn"},
        )
        mock_adapter.handle_message = AsyncMock(return_value="done")

        entry = _make_entry(inject_history=False, notify=["dm"])

        with (
            patch("tsugite.daemon.adapters.scheduler_adapter.send_notification"),
            patch("tsugite.interaction.set_interaction_backend"),
            patch.object(sa, "_inject_into_user_sessions") as mock_inject,
        ):
            await sa._run_agent(entry)

        mock_inject.assert_not_called()
