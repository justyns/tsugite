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
    defaults = dict(id="test-job", agent="bot", prompt="do something", schedule_type="cron", cron_expr="0 9 * * *")
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
        mock_session = MagicMock()
        mock_session.id = session_id
        mock_session.superseded_by = None
        adapter.session_store = MagicMock()
        # Simulate a user with primary session set.
        adapter.session_store.find_primary_session.return_value = mock_session
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

    def test_writes_synthetic_user_input_and_response(self, tmp_path):
        records = self._record_and_load(tmp_path, "test-session", "Task completed successfully")
        user_event = next(r for r in records if r.get("type") == "user_input")
        response_event = next(r for r in records if r.get("type") == "model_response")

        assert user_event["data"]["metadata"]["synthetic"] is True
        assert user_event["data"]["metadata"]["schedule_id"] == "test-job"
        assert '<scheduled_task id="test-job">' in user_event["data"]["text"]
        assert response_event["data"]["raw_content"] == "Task completed successfully"
        assert response_event["data"]["metadata"]["synthetic"] is True

    def test_stores_result_as_is(self, tmp_path):
        result = "pre-truncated result"
        records = self._record_and_load(tmp_path, "trunc-session", result)
        response_event = next(r for r in records if r.get("type") == "model_response")
        assert response_event["data"]["raw_content"] == result


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


class TestTargetSessionField:
    def test_defaults_to_none(self):
        entry = _make_entry()
        assert entry.target_session is None

    def test_explicit_value(self):
        entry = _make_entry(target_session="primary")
        assert entry.target_session == "primary"

    def test_serialization_roundtrip(self):
        entry = _make_entry(target_session="name:research")
        data = asdict(entry)
        assert data["target_session"] == "name:research"
        restored = ScheduleEntry(**data)
        assert restored.target_session == "name:research"

    def test_old_schedules_default(self):
        """Schedules saved before this field existed should default to None."""
        data = asdict(_make_entry())
        del data["target_session"]
        entry = ScheduleEntry(**data)
        assert entry.target_session is None


class TestResolveTargetSession:
    @pytest.fixture
    def store(self, tmp_path):
        from tsugite.daemon.session_store import SessionStore

        return SessionStore(tmp_path / "session_store.json")

    @staticmethod
    def _add_session(store, sid, user_id="justyn", agent="bot"):
        from tsugite.daemon.session_store import Session, SessionSource

        s = Session(id=sid, agent=agent, source=SessionSource.INTERACTIVE.value, user_id=user_id)
        store.create_session(s)
        return s

    def test_null_falls_to_originating_when_no_primary(self, store):
        from tsugite.daemon.adapters.scheduler_adapter import resolve_target_session

        self._add_session(store, "orig-sess")
        entry = _make_entry(target_session=None, originating_session_id="orig-sess")
        result = resolve_target_session(entry, "justyn", store, "bot")
        assert result is not None
        assert result.id == "orig-sess"

    def test_null_no_primary_no_originating_skipped(self, store):
        from tsugite.daemon.adapters.scheduler_adapter import resolve_target_session

        entry = _make_entry(target_session=None, originating_session_id=None)
        assert resolve_target_session(entry, "justyn", store, "bot") is None

    def test_explicit_session_id(self, store):
        from tsugite.daemon.adapters.scheduler_adapter import resolve_target_session

        self._add_session(store, "explicit-id")
        entry = _make_entry(target_session="explicit-id")
        result = resolve_target_session(entry, "justyn", store, "bot")
        assert result is not None
        assert result.id == "explicit-id"

    def test_name_lookup(self, store):
        from tsugite.daemon.adapters.scheduler_adapter import resolve_target_session

        named = store.get_or_create_named_session("justyn", "bot", "research")
        entry = _make_entry(target_session="name:research")
        result = resolve_target_session(entry, "justyn", store, "bot")
        assert result is not None
        assert result.id == named.id

    def test_none_string_skips_injection(self, store):
        from tsugite.daemon.adapters.scheduler_adapter import resolve_target_session

        self._add_session(store, "orig-sess")
        entry = _make_entry(target_session="none", originating_session_id="orig-sess")
        assert resolve_target_session(entry, "justyn", store, "bot") is None

    def test_originating_explicit(self, store):
        from tsugite.daemon.adapters.scheduler_adapter import resolve_target_session

        self._add_session(store, "orig-sess")
        entry = _make_entry(target_session="originating", originating_session_id="orig-sess")
        result = resolve_target_session(entry, "justyn", store, "bot")
        assert result is not None
        assert result.id == "orig-sess"

    def test_unknown_session_id_skipped(self, store):
        from tsugite.daemon.adapters.scheduler_adapter import resolve_target_session

        entry = _make_entry(target_session="does-not-exist")
        assert resolve_target_session(entry, "justyn", store, "bot") is None

    def test_originating_explicit_skipped_when_session_missing(self, store):
        from tsugite.daemon.adapters.scheduler_adapter import resolve_target_session

        entry = _make_entry(target_session="originating", originating_session_id=None)
        assert resolve_target_session(entry, "justyn", store, "bot") is None

    def test_originating_follows_superseded_chain(self, store):
        """A compacted originating session resolves to its successor."""
        from tsugite.daemon.adapters.scheduler_adapter import resolve_target_session
        from tsugite.daemon.session_store import Session, SessionSource

        old = self._add_session(store, "orig-sess")
        new = Session(id="new-sess", agent="bot", source=SessionSource.INTERACTIVE.value, user_id="justyn")
        store.create_session(new)
        store.update_session(old.id, superseded_by=new.id)

        entry = _make_entry(target_session="originating", originating_session_id="orig-sess")
        result = resolve_target_session(entry, "justyn", store, "bot")
        assert result is not None
        assert result.id == "new-sess"

    def test_originating_cycle_protection(self, store):
        """A bogus superseded_by cycle must not loop forever."""
        from tsugite.daemon.adapters.scheduler_adapter import resolve_target_session

        a = self._add_session(store, "sess-a")
        b = self._add_session(store, "sess-b")
        store.update_session(a.id, superseded_by=b.id)
        store.update_session(b.id, superseded_by=a.id)

        entry = _make_entry(target_session="originating", originating_session_id="sess-a")
        # Returns whichever session the walk halts on, but must terminate.
        result = resolve_target_session(entry, "justyn", store, "bot")
        assert result is not None
        assert result.id in {"sess-a", "sess-b"}

    def test_null_routes_to_primary_when_set(self, store):
        from tsugite.daemon.adapters.scheduler_adapter import resolve_target_session

        primary = self._add_session(store, "primary-id")
        self._add_session(store, "orig-id")
        store.set_primary_session(primary.id)

        entry = _make_entry(target_session=None, originating_session_id="orig-id")
        result = resolve_target_session(entry, "justyn", store, "bot")
        assert result is not None
        assert result.id == "primary-id"

    def test_primary_explicit_no_fallback_to_originating(self, store):
        """target_session='primary' returns None when no primary; does NOT fall through."""
        from tsugite.daemon.adapters.scheduler_adapter import resolve_target_session

        self._add_session(store, "orig-id")
        entry = _make_entry(target_session="primary", originating_session_id="orig-id")
        assert resolve_target_session(entry, "justyn", store, "bot") is None


class TestRecordSyntheticTurnWithResolver:
    """Integration: _record_synthetic_turn uses resolve_target_session, not get_or_create_interactive."""

    def _make_real_adapter(self, store, agent="bot"):
        adapter = MagicMock()
        adapter.agent_name = agent
        adapter.resolve_model.return_value = "test-model"
        adapter.session_store = store
        return adapter

    def test_writes_to_resolved_target(self, tmp_path):
        from tsugite.daemon.session_store import Session, SessionSource, SessionStore

        store = SessionStore(tmp_path / "session_store.json")
        target = Session(id="resolved-target", agent="bot", source=SessionSource.INTERACTIVE.value, user_id="justyn")
        store.create_session(target)
        adapter = self._make_real_adapter(store)
        entry = _make_entry(target_session="resolved-target")

        history_dir = tmp_path / "history"
        history_dir.mkdir()
        with patch("tsugite.history.get_history_dir", return_value=history_dir):
            SchedulerAdapter._record_synthetic_turn(adapter, "justyn", entry, "result")

        assert (history_dir / "resolved-target.jsonl").exists()

    def test_no_target_skips_injection(self, tmp_path):
        from tsugite.daemon.session_store import SessionStore

        store = SessionStore(tmp_path / "session_store.json")
        adapter = self._make_real_adapter(store)
        entry = _make_entry(target_session="none", originating_session_id="orig-sess")

        history_dir = tmp_path / "history"
        history_dir.mkdir()
        with patch("tsugite.history.get_history_dir", return_value=history_dir):
            SchedulerAdapter._record_synthetic_turn(adapter, "justyn", entry, "result")

        assert list(history_dir.iterdir()) == []
