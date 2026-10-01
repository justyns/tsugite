"""Tests for per-schedule model and max_turns override pipeline."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from tsugite_daemon.adapters.base import BaseAdapter, ChannelContext
from tsugite_daemon.config import RuntimeDefaults
from tsugite_daemon.scheduler import ScheduleEntry
from tsugite_daemon.session_store import SessionStore


def _make_scheduler_adapter(tmp_path, agent_name="bot"):
    """Create a SchedulerAdapter with a mock adapter for testing."""
    from tsugite_daemon.adapters.scheduler_adapter import SchedulerAdapter

    adapter_mock = AsyncMock()
    adapter_mock.handle_message = AsyncMock(return_value="done")

    sa = SchedulerAdapter(
        adapter=adapter_mock,
        # A real path: the scheduler's storage layer rejects mocks (a MagicMock
        # here used to materialize junk files named after the mock's repr).
        schedules_path=tmp_path / "schedules.json",
    )
    return sa, adapter_mock


class _StubAdapter(BaseAdapter):
    def get_platform_name(self) -> str:
        return "test"

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass


@pytest.fixture
def run_handle_message(tmp_path, monkeypatch):
    """Real BaseAdapter.handle_message with run_agent stubbed; returns the exec_options run_agent received."""
    ws = tmp_path / "workspace"
    ws.mkdir()
    agent_file = ws / "agent.md"
    agent_file.write_text("---\nname: test-agent\nmodel: anthropic:claude-3-sonnet\n---\n\nHi.\n")
    adapter = _StubAdapter(
        RuntimeDefaults(workspace_dir=ws, agent_file=str(agent_file), max_turns=50),
        SessionStore(tmp_path / "store.json"),
    )
    monkeypatch.setattr(adapter, "_resolve_agent_path", lambda *a, **kw: agent_file)
    monkeypatch.setattr(adapter, "_build_message_context", lambda msg, *a, **kw: msg)
    monkeypatch.setattr(adapter, "_build_agent_context", lambda *a, **kw: {})
    monkeypatch.setattr(adapter, "_save_history", lambda **kw: None)
    monkeypatch.setattr(adapter, "_update_skill_ttl", lambda *a, **kw: None)

    captured = {}

    def fake_run_agent(*args, **kwargs):
        captured.update(kwargs)
        return MagicMock(
            token_count=0,
            cost=0,
            execution_steps=[],
            provider_state={},
            last_input_tokens=0,
            cache_creation_tokens=0,
            cache_read_tokens=0,
            __str__=lambda self: "ok",
        )

    monkeypatch.setattr("tsugite_daemon.adapters.base.run_agent", fake_run_agent)

    async def run(metadata):
        adapter.session_store.get_or_create_interactive("alice")
        ctx = ChannelContext(source="http", channel_id=None, user_id="alice", reply_to="http:alice", metadata=metadata)
        await adapter.handle_message(user_id="alice", message="hi", channel_context=ctx)
        return captured["exec_options"]

    return run


class TestSchedulerAdapterMetadata:
    """Test that SchedulerAdapter sets model_override in metadata."""

    @pytest.mark.asyncio
    async def test_model_override_set_in_metadata(self, tmp_path):
        sa, adapter_mock = _make_scheduler_adapter(tmp_path)

        entry = ScheduleEntry(
            id="test", prompt="hi", schedule_type="cron", cron_expr="0 9 * * *", model="openai:gpt-4o-mini"
        )
        await sa._run_agent(entry)

        ctx = adapter_mock.handle_message.call_args[1]["channel_context"]
        assert ctx.metadata["model_override"] == "openai:gpt-4o-mini"

    @pytest.mark.asyncio
    async def test_model_override_absent_when_none(self, tmp_path):
        sa, adapter_mock = _make_scheduler_adapter(tmp_path)

        entry = ScheduleEntry(id="test", prompt="hi", schedule_type="cron", cron_expr="0 9 * * *")
        await sa._run_agent(entry)

        ctx = adapter_mock.handle_message.call_args[1]["channel_context"]
        assert "model_override" not in ctx.metadata


class TestBaseAdapterModelOverride:
    """Model resolution in BaseAdapter.handle_message."""

    @pytest.mark.asyncio
    async def test_model_override_from_metadata(self, run_handle_message):
        opts = await run_handle_message({"model_override": "openai:gpt-4o-mini"})
        assert opts.model_override == "openai:gpt-4o-mini"

    @pytest.mark.asyncio
    async def test_fallback_to_agent_model(self, run_handle_message):
        opts = await run_handle_message({"schedule_id": "test"})
        assert opts.model_override == "anthropic:claude-3-sonnet"

    @pytest.mark.asyncio
    async def test_fallback_when_no_metadata(self, run_handle_message):
        opts = await run_handle_message(None)
        assert opts.model_override == "anthropic:claude-3-sonnet"


class TestSchedulerAdapterMaxTurnsMetadata:
    """Test that SchedulerAdapter sets max_turns_override in metadata."""

    @pytest.mark.asyncio
    async def test_max_turns_set_in_metadata(self, tmp_path):
        sa, adapter_mock = _make_scheduler_adapter(tmp_path)

        entry = ScheduleEntry(id="test", prompt="hi", schedule_type="cron", cron_expr="0 9 * * *", max_turns=40)
        await sa._run_agent(entry)

        ctx = adapter_mock.handle_message.call_args[1]["channel_context"]
        assert ctx.metadata["max_turns_override"] == 40

    @pytest.mark.asyncio
    async def test_max_turns_absent_when_none(self, tmp_path):
        sa, adapter_mock = _make_scheduler_adapter(tmp_path)

        entry = ScheduleEntry(id="test", prompt="hi", schedule_type="cron", cron_expr="0 9 * * *")
        await sa._run_agent(entry)

        ctx = adapter_mock.handle_message.call_args[1]["channel_context"]
        assert "max_turns_override" not in ctx.metadata


class TestBaseAdapterMaxTurnsOverride:
    """max_turns resolution in BaseAdapter.handle_message."""

    @pytest.mark.asyncio
    async def test_max_turns_from_metadata(self, run_handle_message):
        opts = await run_handle_message({"max_turns_override": 40})
        assert opts.max_turns_override == 40

    @pytest.mark.asyncio
    async def test_fallback_to_agent_max_turns(self, run_handle_message):
        opts = await run_handle_message({"schedule_id": "test"})
        assert opts.max_turns_override == 50

    @pytest.mark.asyncio
    async def test_fallback_when_no_metadata(self, run_handle_message):
        opts = await run_handle_message(None)
        assert opts.max_turns_override == 50


class TestSchedulerAdapterFireTiming:
    """Scheduled-task channel context surfaces when this fire was planned for
    and when it actually fired, so the agent can spot misfires / drift.
    """

    @pytest.mark.asyncio
    async def test_actual_fire_time_present(self, tmp_path):
        sa, adapter_mock = _make_scheduler_adapter(tmp_path)
        entry = ScheduleEntry(id="test", prompt="hi", schedule_type="cron", cron_expr="0 9 * * *")
        await sa._run_agent(entry)

        ctx = adapter_mock.handle_message.call_args[1]["channel_context"]
        from datetime import datetime

        parsed = datetime.fromisoformat(ctx.metadata["actual_fire_time"])
        assert parsed.tzinfo is not None

    @pytest.mark.asyncio
    async def test_scheduled_for_threaded_through(self, tmp_path):
        sa, adapter_mock = _make_scheduler_adapter(tmp_path)
        entry = ScheduleEntry(id="test", prompt="hi", schedule_type="cron", cron_expr="0 9 * * *")
        entry.last_scheduled_for = "2026-05-04T09:00:00+00:00"

        await sa._run_agent(entry)

        ctx = adapter_mock.handle_message.call_args[1]["channel_context"]
        assert ctx.metadata["scheduled_for"] == "2026-05-04T09:00:00+00:00"

    @pytest.mark.asyncio
    async def test_scheduled_for_absent_when_unset(self, tmp_path):
        sa, adapter_mock = _make_scheduler_adapter(tmp_path)
        entry = ScheduleEntry(id="test", prompt="hi", schedule_type="cron", cron_expr="0 9 * * *")
        # last_scheduled_for unset (manual fire, replay, etc.)
        await sa._run_agent(entry)

        ctx = adapter_mock.handle_message.call_args[1]["channel_context"]
        assert "scheduled_for" not in ctx.metadata
