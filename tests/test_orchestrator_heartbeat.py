"""Tests for orchestrator heartbeat loop (#54)."""

from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tsugite.daemon.adapters.base import BaseAdapter, ChannelContext, _is_recent
from tsugite.daemon.config import AgentConfig
from tsugite.daemon.session_store import Session, SessionSource, SessionStatus, SessionStore


class _StubAdapter(BaseAdapter):
    async def start(self):
        pass

    async def stop(self):
        pass


@pytest.fixture
def tmp_store(tmp_path):
    return SessionStore(tmp_path / "session_store.json")


@pytest.fixture
def workspace_dir(tmp_path):
    return tmp_path / "workspace"


def _make_adapter(workspace_dir, session_store):
    agent_config = AgentConfig(workspace_dir=workspace_dir, agent_file="default")
    return _StubAdapter("test-agent", agent_config, session_store)


# ── _is_recent helper ──


class TestIsRecent:
    def test_recent_timestamp(self):
        ts = (datetime.now(timezone.utc) - timedelta(minutes=3)).isoformat()
        assert _is_recent(ts, minutes=5) is True

    def test_old_timestamp(self):
        ts = (datetime.now(timezone.utc) - timedelta(minutes=20)).isoformat()
        assert _is_recent(ts, minutes=10) is False

    def test_empty_string(self):
        assert _is_recent("", minutes=10) is False

    def test_invalid_timestamp(self):
        assert _is_recent("not-a-date", minutes=10) is False


# ── Context population ──


class TestBuildAgentContext:
    def test_active_sessions_populated(self, workspace_dir, tmp_store):
        adapter = _make_adapter(workspace_dir, tmp_store)
        # Create a running session
        running = Session(id="sess-1", agent="worker", source="background", status="running", prompt="Do work")
        tmp_store.create_session(running)

        ctx = ChannelContext(source="scheduler", channel_id=None, user_id="test", reply_to="test", metadata={})
        result = adapter._build_agent_context(ctx)

        assert len(result["active_sessions"]) == 1
        assert result["active_sessions"][0]["id"] == "sess-1"
        assert result["active_sessions"][0]["agent"] == "worker"

    def test_recent_completions_populated(self, workspace_dir, tmp_store):
        adapter = _make_adapter(workspace_dir, tmp_store)
        # Create a recently completed session
        completed = Session(
            id="sess-2",
            agent="worker",
            source="background",
            status="completed",
            result="Done!",
            last_active=datetime.now(timezone.utc).isoformat(),
        )
        tmp_store.create_session(completed)

        ctx = ChannelContext(source="scheduler", channel_id=None, user_id="test", reply_to="test", metadata={})
        result = adapter._build_agent_context(ctx)

        assert len(result["recent_completions"]) == 1
        assert result["recent_completions"][0]["id"] == "sess-2"
        assert result["recent_completions"][0]["result"] == "Done!"

    def test_old_completions_excluded(self, workspace_dir, tmp_store):
        adapter = _make_adapter(workspace_dir, tmp_store)
        old_time = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        completed = Session(
            id="sess-old", agent="worker", source="background", status="completed", last_active=old_time
        )
        tmp_store.create_session(completed)

        ctx = ChannelContext(source="scheduler", channel_id=None, user_id="test", reply_to="test", metadata={})
        result = adapter._build_agent_context(ctx)

        assert len(result["recent_completions"]) == 0

    def test_heartbeat_window_from_metadata(self, workspace_dir, tmp_store):
        adapter = _make_adapter(workspace_dir, tmp_store)
        # Session completed 8 minutes ago
        ts = (datetime.now(timezone.utc) - timedelta(minutes=8)).isoformat()
        completed = Session(
            id="sess-3", agent="worker", source="background", status="completed", last_active=ts
        )
        tmp_store.create_session(completed)

        # Default window (10 min) should include it
        ctx = ChannelContext(source="scheduler", channel_id=None, user_id="test", reply_to="test", metadata={})
        result = adapter._build_agent_context(ctx)
        assert len(result["recent_completions"]) == 1

        # Narrow window (5 min) should exclude it
        ctx2 = ChannelContext(
            source="scheduler", channel_id=None, user_id="test", reply_to="test", metadata={"heartbeat_window": 5}
        )
        result2 = adapter._build_agent_context(ctx2)
        assert len(result2["recent_completions"]) == 0

    def test_empty_when_no_sessions(self, workspace_dir, tmp_store):
        adapter = _make_adapter(workspace_dir, tmp_store)
        ctx = ChannelContext(source="scheduler", channel_id=None, user_id="test", reply_to="test", metadata={})
        result = adapter._build_agent_context(ctx)

        assert result["active_sessions"] == []
        assert result["recent_completions"] == []


# ── run_if with session context ──


class TestRunIfWithSessions:
    def test_run_if_sessions_guard(self):
        from tsugite.agent_preparation import AgentPreparer
        from tsugite.md_agents import Agent, AgentConfig as MdAgentConfig

        config = MdAgentConfig(
            name="orchestrator",
            run_if="active_sessions | length > 0 or recent_completions | length > 0",
        )
        agent = Agent(config=config, content="Supervise.", file_path=Path("test.md"))

        # With empty sessions → should skip
        preparer = AgentPreparer()
        result = preparer.prepare(agent, prompt="heartbeat", context={"active_sessions": [], "recent_completions": []})
        assert result.skipped

        # With active sessions → should not skip
        result2 = preparer.prepare(
            agent,
            prompt="heartbeat",
            context={"active_sessions": [{"id": "s1"}], "recent_completions": []},
        )
        assert not result2.skipped


# ── Scheduler minimum interval ──


class TestSchedulerMinInterval:
    def test_update_rejects_too_frequent_cron(self, tmp_path):
        from tsugite.daemon.scheduler import ScheduleEntry, Scheduler

        path = tmp_path / "schedules.json"
        scheduler = Scheduler(path, run_callback=AsyncMock())
        entry = ScheduleEntry(id="hb", agent="orch", prompt="go", schedule_type="cron", cron_expr="*/5 * * * *")
        scheduler.add(entry)

        with pytest.raises(ValueError, match="fires every"):
            scheduler.update("hb", cron_expr="* * * * *")

    def test_update_allows_valid_cron(self, tmp_path):
        from tsugite.daemon.scheduler import ScheduleEntry, Scheduler

        path = tmp_path / "schedules.json"
        scheduler = Scheduler(path, run_callback=AsyncMock())
        entry = ScheduleEntry(id="hb", agent="orch", prompt="go", schedule_type="cron", cron_expr="*/5 * * * *")
        scheduler.add(entry)

        result = scheduler.update("hb", cron_expr="*/3 * * * *")
        assert result.cron_expr == "*/3 * * * *"


# ── session_reply tool ──


class TestSessionReply:
    def test_reply_to_session_calls_adapter(self, tmp_store):
        import asyncio

        from tsugite.daemon.session_runner import SessionRunner

        mock_adapter = MagicMock()
        mock_adapter.handle_message = AsyncMock(return_value="response text")

        runner = SessionRunner(store=tmp_store, adapters={"worker": mock_adapter})

        session = Session(id="sess-reply", agent="worker", source="background", status="completed", prompt="task")
        tmp_store.create_session(session)

        result = asyncio.run(runner.reply_to_session("sess-reply", "follow up"))

        assert result == "response text"
        mock_adapter.handle_message.assert_called_once()
        call_kwargs = mock_adapter.handle_message.call_args
        assert call_kwargs.kwargs["message"] == "follow up"
        ctx = call_kwargs.kwargs["channel_context"]
        assert ctx.metadata["conv_id_override"] == "sess-reply"
        assert ctx.metadata["session_id"] == "sess-reply"

    def test_reply_to_missing_adapter_raises(self, tmp_store):
        import asyncio

        from tsugite.daemon.session_runner import SessionRunner

        runner = SessionRunner(store=tmp_store, adapters={})

        session = Session(id="sess-no-adapter", agent="ghost", source="background", status="completed", prompt="task")
        tmp_store.create_session(session)

        with pytest.raises(ValueError, match="No adapter"):
            asyncio.run(runner.reply_to_session("sess-no-adapter", "hello"))
