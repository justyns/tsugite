"""Tests for agent file hot-loading (Feature #8)."""

from dataclasses import asdict
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tsugite.daemon.adapters.base import BaseAdapter, ChannelContext, resolve_agent_path
from tsugite.daemon.adapters.scheduler_adapter import SchedulerAdapter
from tsugite.daemon.scheduler import ScheduleEntry


def _get_channel_context(mock):
    """Extract channel_context from a mock's call_args."""
    return mock.call_args.kwargs["channel_context"]


class TestAgentFileField:
    def test_defaults_to_none(self):
        entry = ScheduleEntry(id="t", agent="bot", prompt="hi", schedule_type="cron", cron_expr="0 9 * * *")
        assert entry.agent_file is None

    def test_explicit_value(self):
        entry = ScheduleEntry(
            id="t", agent="bot", prompt="hi", schedule_type="cron", cron_expr="0 9 * * *",
            agent_file="agents/custom.md",
        )
        assert entry.agent_file == "agents/custom.md"

    def test_serialization_roundtrip(self):
        entry = ScheduleEntry(
            id="t", agent="bot", prompt="hi", schedule_type="cron", cron_expr="0 9 * * *",
            agent_file="agents/custom.md",
        )
        data = asdict(entry)
        restored = ScheduleEntry(**data)
        assert restored.agent_file == "agents/custom.md"

    def test_old_schedules_compat(self):
        """Old schedule data without agent_file should still load."""
        data = {
            "id": "old", "agent": "bot", "prompt": "hi", "schedule_type": "cron",
            "cron_expr": "0 9 * * *", "enabled": True, "created_at": "2026-01-01T00:00:00",
            "last_run": None, "next_run": None, "last_status": None, "last_error": None,
            "notify": [], "notify_tool": False, "inject_history": True, "auto_reply": False,
            "model": None, "misfire_grace_seconds": 300, "timezone": "UTC",
        }
        entry = ScheduleEntry(**data)
        assert entry.agent_file is None


class TestRunAgentWithAgentFile:
    @pytest.fixture
    def adapter(self):
        adapter = MagicMock(spec=BaseAdapter)
        adapter.agent_config = MagicMock()
        adapter.agent_config.workspace_dir = Path("/workspace")
        adapter.handle_message = AsyncMock(return_value="result")
        adapter._workspace = None
        adapter._resolve_agent_path = lambda agent_file=None: resolve_agent_path(
            agent_file or adapter.agent_config.agent_file,
            adapter.agent_config.workspace_dir,
            adapter._workspace,
        )
        return adapter

    @pytest.fixture
    def scheduler_adapter(self, tmp_path, adapter):
        return SchedulerAdapter(
            adapters={"bot": adapter},
            schedules_path=tmp_path / "schedules.json",
        )

    @pytest.mark.asyncio
    async def test_metadata_includes_agent_file_override(self, adapter, scheduler_adapter, tmp_path):
        agent_file = tmp_path / "custom.md"
        agent_file.write_text("---\nname: custom\n---\nHello")

        entry = ScheduleEntry(
            id="t", agent="bot", prompt="hi", schedule_type="cron", cron_expr="0 9 * * *",
            agent_file=str(agent_file),
        )

        with patch("tsugite.daemon.adapters.scheduler_adapter.send_notification"):
            await scheduler_adapter._run_agent(entry)

        ctx = _get_channel_context(adapter.handle_message)
        assert ctx.metadata["agent_file_override"] == str(agent_file)

    @pytest.mark.asyncio
    async def test_relative_path_resolved_against_workspace(self, adapter, scheduler_adapter, tmp_path):
        adapter.agent_config.workspace_dir = tmp_path
        agent_file = tmp_path / "agents" / "custom.md"
        agent_file.parent.mkdir(parents=True, exist_ok=True)
        agent_file.write_text("---\nname: custom\n---\nHello")

        entry = ScheduleEntry(
            id="t", agent="bot", prompt="hi", schedule_type="cron", cron_expr="0 9 * * *",
            agent_file="agents/custom.md",
        )

        with patch("tsugite.daemon.adapters.scheduler_adapter.send_notification"):
            await scheduler_adapter._run_agent(entry)

        ctx = _get_channel_context(adapter.handle_message)
        assert ctx.metadata["agent_file_override"] == str(agent_file)

    @pytest.mark.asyncio
    async def test_missing_file_raises(self, adapter, scheduler_adapter):
        entry = ScheduleEntry(
            id="t", agent="bot", prompt="hi", schedule_type="cron", cron_expr="0 9 * * *",
            agent_file="/nonexistent/agent.md",
        )

        with pytest.raises(FileNotFoundError, match="Agent file not found"):
            await scheduler_adapter._run_agent(entry)

    @pytest.mark.asyncio
    async def test_no_agent_file_no_override(self, adapter, scheduler_adapter):
        entry = ScheduleEntry(
            id="t", agent="bot", prompt="hi", schedule_type="cron", cron_expr="0 9 * * *",
        )

        with patch("tsugite.daemon.adapters.scheduler_adapter.send_notification"):
            await scheduler_adapter._run_agent(entry)

        ctx = _get_channel_context(adapter.handle_message)
        assert "agent_file_override" not in ctx.metadata


class _ConcreteAdapter(BaseAdapter):
    async def start(self): pass
    async def stop(self): pass


def _make_base_adapter(tmp_path, **overrides):
    """Build a minimal BaseAdapter mock for handle_message tests."""
    adapter = _ConcreteAdapter.__new__(_ConcreteAdapter)
    adapter.agent_name = "test"
    adapter.agent_config = MagicMock()
    adapter.agent_config.workspace_dir = tmp_path
    adapter.agent_config.agent_file = "default"
    adapter.agent_config.model = None
    adapter.agent_config.max_turns = None
    adapter.agent_config.context_limit = 128000
    adapter.session_manager = MagicMock()
    adapter.session_manager.needs_compaction.return_value = False
    adapter.session_manager.get_or_create_session.return_value = "conv-123"
    adapter.session_manager.sessions = {}
    adapter.workspace_attachments = []
    adapter._workspace = None
    adapter._identity_map = {}
    for k, v in overrides.items():
        setattr(adapter, k, v)
    return adapter


def _mock_run_result():
    """Create a mock run_agent result."""
    result = MagicMock()
    result.__str__ = lambda self: "result"
    result.token_count = None
    return result


class TestHandleMessageOverride:
    @pytest.mark.asyncio
    async def test_uses_override_path(self, tmp_path):
        """handle_message uses agent_file_override when file exists."""
        override_file = tmp_path / "override.md"
        override_file.write_text("---\nname: override\n---\nOverride agent")

        channel_context = ChannelContext(
            source="scheduler", channel_id=None, user_id="test",
            reply_to="test", metadata={"agent_file_override": str(override_file)},
        )

        adapter = _make_base_adapter(tmp_path)
        with (
            patch.object(adapter, "_resolve_agent_path", return_value=None),
            patch("tsugite.daemon.adapters.base.run_agent", return_value=_mock_run_result()) as mock_run,
        ):
            await adapter.handle_message("test", "hi", channel_context)
            assert mock_run.call_args.kwargs["agent_path"] == override_file

    @pytest.mark.asyncio
    async def test_ignores_nonexistent_override(self, tmp_path):
        """handle_message falls back to default if override file doesn't exist."""
        channel_context = ChannelContext(
            source="scheduler", channel_id=None, user_id="test",
            reply_to="test", metadata={"agent_file_override": "/nonexistent/agent.md"},
        )

        default_agent = tmp_path / "default.md"
        default_agent.write_text("---\nname: default\n---\nDefault")

        adapter = _make_base_adapter(tmp_path)
        with (
            patch.object(adapter, "_resolve_agent_path", return_value=default_agent),
            patch("tsugite.daemon.adapters.base.run_agent", return_value=_mock_run_result()) as mock_run,
        ):
            await adapter.handle_message("test", "hi", channel_context)
            assert mock_run.call_args.kwargs["agent_path"] == default_agent
