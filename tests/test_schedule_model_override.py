"""Tests for per-schedule model and max_turns override pipeline."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from tsugite_daemon.scheduler import ScheduleEntry


def _make_scheduler_adapter(tmp_path, agent_name="bot"):
    """Create a SchedulerAdapter with a mock adapter for testing."""
    from tsugite_daemon.adapters.scheduler_adapter import SchedulerAdapter

    adapter_mock = AsyncMock()
    adapter_mock.handle_message = AsyncMock(return_value="done")

    sa = SchedulerAdapter(
        adapters={agent_name: adapter_mock},
        # A real path: the scheduler's storage layer rejects mocks (a MagicMock
        # here used to materialize junk files named after the mock's repr).
        schedules_path=tmp_path / "schedules.json",
    )
    return sa, adapter_mock


def _resolve_model_override(metadata, fallback="anthropic:claude-3-sonnet"):
    """Replicate the model_override resolution logic from BaseAdapter.handle_message."""
    return (metadata or {}).get("model_override") or fallback


def _resolve_max_turns_override(metadata, fallback=50):
    """Replicate the max_turns_override resolution logic from BaseAdapter.handle_message."""
    return (metadata or {}).get("max_turns_override") or fallback


class TestSchedulerAdapterMetadata:
    """Test that SchedulerAdapter sets model_override in metadata."""

    @pytest.mark.asyncio
    async def test_model_override_set_in_metadata(self, tmp_path):
        sa, adapter_mock = _make_scheduler_adapter(tmp_path)

        entry = ScheduleEntry(
            id="test", agent="bot", prompt="hi", schedule_type="cron", cron_expr="0 9 * * *", model="openai:gpt-4o-mini"
        )
        await sa._run_agent(entry)

        ctx = adapter_mock.handle_message.call_args[1]["channel_context"]
        assert ctx.metadata["model_override"] == "openai:gpt-4o-mini"

    @pytest.mark.asyncio
    async def test_model_override_absent_when_none(self, tmp_path):
        sa, adapter_mock = _make_scheduler_adapter(tmp_path)

        entry = ScheduleEntry(id="test", agent="bot", prompt="hi", schedule_type="cron", cron_expr="0 9 * * *")
        await sa._run_agent(entry)

        ctx = adapter_mock.handle_message.call_args[1]["channel_context"]
        assert "model_override" not in ctx.metadata


class TestBaseAdapterModelOverride:
    """Test the model_override resolution logic used in handle_message."""

    def test_model_override_from_metadata(self, tmp_path):
        assert _resolve_model_override({"model_override": "openai:gpt-4o-mini"}) == "openai:gpt-4o-mini"

    def test_fallback_to_agent_model(self, tmp_path):
        assert _resolve_model_override({"schedule_id": "test"}) == "anthropic:claude-3-sonnet"

    def test_fallback_when_no_metadata(self, tmp_path):
        assert _resolve_model_override(None) == "anthropic:claude-3-sonnet"


class TestSchedulerAdapterMaxTurnsMetadata:
    """Test that SchedulerAdapter sets max_turns_override in metadata."""

    @pytest.mark.asyncio
    async def test_max_turns_set_in_metadata(self, tmp_path):
        sa, adapter_mock = _make_scheduler_adapter(tmp_path)

        entry = ScheduleEntry(
            id="test", agent="bot", prompt="hi", schedule_type="cron", cron_expr="0 9 * * *", max_turns=40
        )
        await sa._run_agent(entry)

        ctx = adapter_mock.handle_message.call_args[1]["channel_context"]
        assert ctx.metadata["max_turns_override"] == 40

    @pytest.mark.asyncio
    async def test_max_turns_absent_when_none(self, tmp_path):
        sa, adapter_mock = _make_scheduler_adapter(tmp_path)

        entry = ScheduleEntry(id="test", agent="bot", prompt="hi", schedule_type="cron", cron_expr="0 9 * * *")
        await sa._run_agent(entry)

        ctx = adapter_mock.handle_message.call_args[1]["channel_context"]
        assert "max_turns_override" not in ctx.metadata


class TestBaseAdapterMaxTurnsOverride:
    """Test the max_turns_override resolution logic used in handle_message."""

    def test_max_turns_from_metadata(self, tmp_path):
        assert _resolve_max_turns_override({"max_turns_override": 40}) == 40

    def test_fallback_to_agent_max_turns(self, tmp_path):
        assert _resolve_max_turns_override({"schedule_id": "test"}) == 50

    def test_fallback_when_no_metadata(self, tmp_path):
        assert _resolve_max_turns_override(None) == 50


class TestSchedulerAdapterFireTiming:
    """Scheduled-task channel context surfaces when this fire was planned for
    and when it actually fired, so the agent can spot misfires / drift.
    """

    @pytest.mark.asyncio
    async def test_actual_fire_time_present(self, tmp_path):
        sa, adapter_mock = _make_scheduler_adapter(tmp_path)
        entry = ScheduleEntry(id="test", agent="bot", prompt="hi", schedule_type="cron", cron_expr="0 9 * * *")
        await sa._run_agent(entry)

        ctx = adapter_mock.handle_message.call_args[1]["channel_context"]
        from datetime import datetime

        parsed = datetime.fromisoformat(ctx.metadata["actual_fire_time"])
        assert parsed.tzinfo is not None

    @pytest.mark.asyncio
    async def test_scheduled_for_threaded_through(self, tmp_path):
        sa, adapter_mock = _make_scheduler_adapter(tmp_path)
        entry = ScheduleEntry(id="test", agent="bot", prompt="hi", schedule_type="cron", cron_expr="0 9 * * *")
        entry.last_scheduled_for = "2026-05-04T09:00:00+00:00"

        await sa._run_agent(entry)

        ctx = adapter_mock.handle_message.call_args[1]["channel_context"]
        assert ctx.metadata["scheduled_for"] == "2026-05-04T09:00:00+00:00"

    @pytest.mark.asyncio
    async def test_scheduled_for_absent_when_unset(self, tmp_path):
        sa, adapter_mock = _make_scheduler_adapter(tmp_path)
        entry = ScheduleEntry(id="test", agent="bot", prompt="hi", schedule_type="cron", cron_expr="0 9 * * *")
        # last_scheduled_for unset (manual fire, replay, etc.)
        await sa._run_agent(entry)

        ctx = adapter_mock.handle_message.call_args[1]["channel_context"]
        assert "scheduled_for" not in ctx.metadata
