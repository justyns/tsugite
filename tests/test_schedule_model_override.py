"""Tests for per-schedule model override pipeline."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from tsugite.daemon.scheduler import ScheduleEntry


def _make_scheduler_adapter(agent_name="bot"):
    """Create a SchedulerAdapter with a mock adapter for testing."""
    from tsugite.daemon.adapters.scheduler_adapter import SchedulerAdapter

    adapter_mock = AsyncMock()
    adapter_mock.handle_message = AsyncMock(return_value="done")

    sa = SchedulerAdapter(
        adapters={agent_name: adapter_mock},
        schedules_path=MagicMock(),
    )
    return sa, adapter_mock


def _resolve_model_override(metadata, fallback="anthropic:claude-3-sonnet"):
    """Replicate the model_override resolution logic from BaseAdapter.handle_message."""
    return (metadata or {}).get("model_override") or fallback


class TestSchedulerAdapterMetadata:
    """Test that SchedulerAdapter sets model_override in metadata."""

    @pytest.mark.asyncio
    async def test_model_override_set_in_metadata(self):
        sa, adapter_mock = _make_scheduler_adapter()

        entry = ScheduleEntry(
            id="test", agent="bot", prompt="hi", schedule_type="cron", cron_expr="0 9 * * *", model="openai:gpt-4o-mini"
        )
        await sa._run_agent(entry)

        ctx = adapter_mock.handle_message.call_args[1]["channel_context"]
        assert ctx.metadata["model_override"] == "openai:gpt-4o-mini"

    @pytest.mark.asyncio
    async def test_model_override_absent_when_none(self):
        sa, adapter_mock = _make_scheduler_adapter()

        entry = ScheduleEntry(id="test", agent="bot", prompt="hi", schedule_type="cron", cron_expr="0 9 * * *")
        await sa._run_agent(entry)

        ctx = adapter_mock.handle_message.call_args[1]["channel_context"]
        assert "model_override" not in ctx.metadata


class TestBaseAdapterModelOverride:
    """Test the model_override resolution logic used in handle_message."""

    def test_model_override_from_metadata(self):
        assert _resolve_model_override({"model_override": "openai:gpt-4o-mini"}) == "openai:gpt-4o-mini"

    def test_fallback_to_agent_model(self):
        assert _resolve_model_override({"schedule_id": "test"}) == "anthropic:claude-3-sonnet"

    def test_fallback_when_no_metadata(self):
        assert _resolve_model_override(None) == "anthropic:claude-3-sonnet"
