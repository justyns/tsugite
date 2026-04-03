"""Tests for background task completion callbacks (on_complete feature)."""

import asyncio
import contextvars
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Thread
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tsugite.agent_runner.models import AgentExecutionResult
from tsugite.daemon.adapters.base import BaseAdapter, ChannelContext
from tsugite.daemon.adapters.scheduler_adapter import MAX_CHAIN_DEPTH, SchedulerAdapter
from tsugite.daemon.config import AgentConfig
from tsugite.daemon.scheduler import ScheduleEntry
from tsugite.daemon.session_runner import (
    SessionRunner,
    _current_session_id,
    get_current_chain_depth,
    get_current_session_id,
    set_current_chain_depth,
    set_current_session_id,
)
from tsugite.daemon.session_store import Session, SessionSource, SessionStatus, SessionStore


# --- Fixtures & helpers ---


@pytest.fixture
def session_store(tmp_path):
    return SessionStore(tmp_path / "session_store.json", context_limits={"bot": 128000})


@pytest.fixture
def mock_session_runner(session_store):
    runner = MagicMock(spec=SessionRunner)
    runner.store = session_store
    runner.is_session_running = MagicMock(return_value=False)
    runner.reply_to_session = AsyncMock(return_value="Agent processed the result.")
    return runner


def _make_sa(session_runner=None) -> tuple[SchedulerAdapter, MagicMock]:
    mock_adapter = MagicMock()
    mock_adapter.agent_name = "bot"
    mock_adapter.handle_message = AsyncMock(return_value="done")
    mock_adapter.resolve_model = MagicMock(return_value="test-model")
    sa = SchedulerAdapter(
        adapters={"bot": mock_adapter},
        schedules_path=Path("/tmp/test-schedules.json"),
    )
    if session_runner:
        sa.set_session_runner(session_runner)
    return sa, mock_adapter


def _make_entry(on_complete=None, originating_session_id=None, chain_depth=0, **kw) -> ScheduleEntry:
    return ScheduleEntry(
        id=kw.get("id", "bg-test"),
        agent="bot",
        prompt=kw.get("prompt", "do work"),
        schedule_type="once",
        run_at="2099-01-01T00:00:00Z",
        on_complete=on_complete,
        originating_session_id=originating_session_id,
        chain_depth=chain_depth,
    )


def _create_session(session_store, session_id="session-abc"):
    session = Session(
        id=session_id, agent="bot",
        source=SessionSource.INTERACTIVE.value,
        status=SessionStatus.ACTIVE.value,
    )
    session_store.create_session(session)
    return session


# --- _handle_on_complete ---


class TestHandleOnComplete:
    @pytest.mark.asyncio
    async def test_noop_when_no_on_complete(self, session_store, mock_session_runner):
        sa, _ = _make_sa(mock_session_runner)
        await sa._handle_on_complete(_make_entry(), "result")
        mock_session_runner.reply_to_session.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_noop_when_action_not_reply(self, session_store, mock_session_runner):
        sa, _ = _make_sa(mock_session_runner)
        await sa._handle_on_complete(
            _make_entry(on_complete={"action": "webhook"}, originating_session_id="s"), "result",
        )
        mock_session_runner.reply_to_session.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_noop_without_originating_session(self, session_store, mock_session_runner):
        sa, _ = _make_sa(mock_session_runner)
        await sa._handle_on_complete(
            _make_entry(on_complete={"action": "reply"}, originating_session_id=None), "result",
        )
        mock_session_runner.reply_to_session.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_noop_without_session_runner(self, session_store):
        sa, _ = _make_sa(session_runner=None)
        await sa._handle_on_complete(
            _make_entry(on_complete={"action": "reply"}, originating_session_id="s"), "result",
        )

    @pytest.mark.asyncio
    async def test_replies_to_session(self, session_store, mock_session_runner):
        sa, _ = _make_sa(mock_session_runner)
        _create_session(session_store, session_id="session-abc")
        entry = _make_entry(on_complete={"action": "reply"}, originating_session_id="session-abc")

        await sa._handle_on_complete(entry, "task output")

        mock_session_runner.reply_to_session.assert_awaited_once()
        call_kw = mock_session_runner.reply_to_session.call_args
        assert call_kw[0][0] == "session-abc"
        assert "background_task_complete" in call_kw[0][1]
        assert "task output" in call_kw[0][1]
        assert call_kw[1]["source"] == "completion_callback"

    @pytest.mark.asyncio
    async def test_replies_even_to_old_session(self, session_store, mock_session_runner):
        """Unlike the old idle check, cold sessions now get replies too."""
        sa, _ = _make_sa(mock_session_runner)
        old_time = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
        session = _create_session(session_store, session_id="session-abc")
        session_store.update_session("session-abc", last_active=old_time)
        entry = _make_entry(on_complete={"action": "reply"}, originating_session_id="session-abc")

        await sa._handle_on_complete(entry, "result")

        mock_session_runner.reply_to_session.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_injects_history_when_mid_turn(self, session_store, mock_session_runner):
        sa, _ = _make_sa(mock_session_runner)
        _create_session(session_store, session_id="session-abc")
        mock_session_runner.is_session_running.return_value = True
        entry = _make_entry(on_complete={"action": "reply"}, originating_session_id="session-abc")

        with patch.object(sa, "_inject_completion_into_history", new_callable=AsyncMock) as mock_inject:
            await sa._handle_on_complete(entry, "task output")

        mock_session_runner.reply_to_session.assert_not_awaited()
        mock_inject.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_message_contains_structured_xml(self, session_store, mock_session_runner):
        sa, _ = _make_sa(mock_session_runner)
        _create_session(session_store, session_id="session-abc")
        entry = _make_entry(
            on_complete={"action": "reply"}, originating_session_id="session-abc",
            prompt="analyze the logs", chain_depth=1,
        )

        await sa._handle_on_complete(entry, "found 3 errors")

        msg = mock_session_runner.reply_to_session.call_args[0][1]
        assert "<prompt>analyze the logs</prompt>" in msg
        assert "<result>" in msg
        assert "found 3 errors" in msg
        assert 'chain_depth="1"' in msg

    @pytest.mark.asyncio
    async def test_prompt_truncated_in_message(self, session_store, mock_session_runner):
        sa, _ = _make_sa(mock_session_runner)
        _create_session(session_store, session_id="session-abc")
        long_prompt = "x" * 300
        entry = _make_entry(
            on_complete={"action": "reply"}, originating_session_id="session-abc",
            prompt=long_prompt,
        )

        await sa._handle_on_complete(entry, "done")

        msg = mock_session_runner.reply_to_session.call_args[0][1]
        assert len(long_prompt) > 200
        assert "…" in msg


# --- Chain depth ---


class TestChainDepth:
    @pytest.mark.asyncio
    async def test_skips_at_max_depth(self, session_store, mock_session_runner):
        sa, _ = _make_sa(mock_session_runner)
        entry = _make_entry(
            on_complete={"action": "reply"}, originating_session_id="session-abc",
            chain_depth=MAX_CHAIN_DEPTH,
        )
        await sa._handle_on_complete(entry, "result")
        mock_session_runner.reply_to_session.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_allows_below_max_depth(self, session_store, mock_session_runner):
        sa, _ = _make_sa(mock_session_runner)
        _create_session(session_store, session_id="session-abc")
        entry = _make_entry(
            on_complete={"action": "reply"}, originating_session_id="session-abc",
            chain_depth=MAX_CHAIN_DEPTH - 1,
        )
        await sa._handle_on_complete(entry, "result")
        mock_session_runner.reply_to_session.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_sets_context_var_during_reply(self, session_store, mock_session_runner):
        sa, _ = _make_sa(mock_session_runner)
        _create_session(session_store, session_id="session-abc")

        captured = None

        async def capture(session_id, message, **kw):
            nonlocal captured
            captured = get_current_chain_depth()
            return "ok"

        mock_session_runner.reply_to_session = capture
        entry = _make_entry(
            on_complete={"action": "reply"}, originating_session_id="session-abc", chain_depth=2,
        )
        await sa._handle_on_complete(entry, "result")
        assert captured == 3

    @pytest.mark.asyncio
    async def test_resets_context_var_after_reply(self, session_store, mock_session_runner):
        sa, _ = _make_sa(mock_session_runner)
        _create_session(session_store, session_id="session-abc")
        entry = _make_entry(
            on_complete={"action": "reply"}, originating_session_id="session-abc", chain_depth=3,
        )
        await sa._handle_on_complete(entry, "result")
        assert get_current_chain_depth() == 0

    @pytest.mark.asyncio
    async def test_resets_context_var_on_error(self, session_store, mock_session_runner):
        sa, _ = _make_sa(mock_session_runner)
        _create_session(session_store, session_id="session-abc")
        mock_session_runner.reply_to_session = AsyncMock(side_effect=RuntimeError("boom"))
        entry = _make_entry(
            on_complete={"action": "reply"}, originating_session_id="session-abc", chain_depth=3,
        )
        await sa._handle_on_complete(entry, "result")
        assert get_current_chain_depth() == 0


# --- _run_agent integration ---


class TestRunAgentOnComplete:
    @pytest.mark.asyncio
    async def test_calls_handle_on_complete(self, session_store, mock_session_runner):
        sa, _ = _make_sa(mock_session_runner)
        entry = _make_entry(on_complete={"action": "reply"}, originating_session_id="session-abc")

        with (
            patch("tsugite.interaction.set_interaction_backend"),
            patch.object(sa, "_handle_on_complete", new_callable=AsyncMock) as mock_handle,
        ):
            await sa._run_agent(entry)

        mock_handle.assert_awaited_once()
        assert mock_handle.call_args[0][0] is entry


# --- background_task() tool ---


class TestBackgroundTaskTool:
    @pytest.fixture
    def tool_loop(self):
        from tsugite.tools.schedule import set_scheduler

        loop = asyncio.new_event_loop()
        t = Thread(target=loop.run_forever, daemon=True)
        t.start()
        yield loop
        loop.call_soon_threadsafe(loop.stop)
        t.join(timeout=2)
        loop.close()
        set_scheduler(None)

    def test_passes_on_complete_and_session(self, tool_loop):
        from tsugite.tools.schedule import background_task, set_scheduler

        mock_sched = MagicMock()
        set_scheduler(mock_sched, tool_loop, agent_names={"bot"})

        with (
            patch("tsugite.agent_runner.helpers.get_current_agent", return_value="bot"),
            patch("tsugite.daemon.session_runner.get_current_session_id", return_value="session-123"),
        ):
            result = background_task(prompt="test", on_complete={"action": "reply"})

        assert result["status"] == "started"
        entry = mock_sched.add.call_args[0][0]
        assert entry.on_complete == {"action": "reply"}
        assert entry.originating_session_id == "session-123"
        assert entry.chain_depth == 0

    def test_captures_chain_depth(self, tool_loop):
        from tsugite.tools.schedule import background_task, set_scheduler

        mock_sched = MagicMock()
        set_scheduler(mock_sched, tool_loop, agent_names={"bot"})

        with (
            patch("tsugite.agent_runner.helpers.get_current_agent", return_value="bot"),
            patch("tsugite.daemon.session_runner.get_current_session_id", return_value="s"),
            patch("tsugite.daemon.session_runner.get_current_chain_depth", return_value=3),
        ):
            background_task(prompt="test", on_complete={"action": "reply"})

        assert mock_sched.add.call_args[0][0].chain_depth == 3

    def test_rejects_invalid_on_complete(self, tool_loop):
        from tsugite.tools.schedule import background_task, set_scheduler

        mock_sched = MagicMock()
        set_scheduler(mock_sched, tool_loop, agent_names={"bot"})

        with (
            patch("tsugite.agent_runner.helpers.get_current_agent", return_value="bot"),
            pytest.raises(ValueError, match="on_complete must be"),
        ):
            background_task(prompt="test", on_complete={"action": "webhook"})

    def test_rejects_without_session_context(self, tool_loop):
        from tsugite.tools.schedule import background_task, set_scheduler

        mock_sched = MagicMock()
        set_scheduler(mock_sched, tool_loop, agent_names={"bot"})

        with (
            patch("tsugite.agent_runner.helpers.get_current_agent", return_value="bot"),
            patch("tsugite.daemon.session_runner.get_current_session_id", return_value=None),
            pytest.raises(ValueError, match="session context"),
        ):
            background_task(prompt="test", on_complete={"action": "reply"})

    def test_skips_session_capture_without_on_complete(self, tool_loop):
        from tsugite.tools.schedule import background_task, set_scheduler

        mock_sched = MagicMock()
        set_scheduler(mock_sched, tool_loop, agent_names={"bot"})

        with patch("tsugite.agent_runner.helpers.get_current_agent", return_value="bot"):
            background_task(prompt="test")

        entry = mock_sched.add.call_args[0][0]
        assert entry.on_complete is None
        assert entry.originating_session_id is None


# --- SessionRunner.is_session_running ---


class TestIsSessionRunning:
    def test_true_when_active(self):
        runner = SessionRunner(MagicMock(), {})
        task = MagicMock(done=MagicMock(return_value=False))
        runner._active_tasks["s1"] = task
        assert runner.is_session_running("s1") is True

    def test_false_when_done(self):
        runner = SessionRunner(MagicMock(), {})
        task = MagicMock(done=MagicMock(return_value=True))
        runner._active_tasks["s1"] = task
        assert runner.is_session_running("s1") is False

    def test_false_when_absent(self):
        assert SessionRunner(MagicMock(), {}).is_session_running("s1") is False


# --- Context var ---


class TestChainDepthContextVar:
    def test_default_zero(self):
        assert get_current_chain_depth() == 0

    def test_roundtrip(self):
        set_current_chain_depth(3)
        assert get_current_chain_depth() == 3
        set_current_chain_depth(0)


class TestSessionIdContextVar:
    def test_default_none(self):
        assert get_current_session_id() is None

    def test_roundtrip(self):
        set_current_session_id("session-xyz")
        assert get_current_session_id() == "session-xyz"
        _current_session_id.set(None)


# --- handle_message sets session_id contextvar ---


class _StubAdapter(BaseAdapter):
    async def start(self):
        pass

    async def stop(self):
        pass


class TestHandleMessageSetsSessionId:
    """Verify handle_message populates _current_session_id for tools like background_task."""

    @pytest.fixture(autouse=True)
    def _reset_session_id(self):
        yield
        _current_session_id.set(None)

    @pytest.fixture
    def adapter(self, tmp_path):
        store = SessionStore(tmp_path / "sessions.json", context_limits={"bot": 128000})
        config = AgentConfig(workspace_dir=tmp_path, agent_file="default")
        return _StubAdapter("bot", config, store)

    @pytest.mark.asyncio
    async def test_sets_session_id_when_not_preset(self, adapter, tmp_path):
        """HTTP/Discord path: handle_message should set _current_session_id to conv_id."""
        captured_session_id = None

        def fake_run_agent(**kwargs):
            nonlocal captured_session_id
            captured_session_id = get_current_session_id()
            return AgentExecutionResult(response="ok")

        ctx = ChannelContext(source="http", channel_id=None, user_id="user1", reply_to="http:user1")

        with patch("tsugite.daemon.adapters.base.run_agent", side_effect=fake_run_agent):
            await adapter.handle_message("user1", "hello", ctx)

        assert captured_session_id is not None, "_current_session_id was not set inside run_agent"

    @pytest.mark.asyncio
    async def test_does_not_overwrite_preset_session_id(self, adapter, tmp_path):
        """Session runner path: pre-set session_id must not be overwritten."""
        captured_session_id = None

        def fake_run_agent(**kwargs):
            nonlocal captured_session_id
            captured_session_id = get_current_session_id()
            return AgentExecutionResult(response="ok")

        set_current_session_id("scheduler-session-99")

        ctx = ChannelContext(source="session", channel_id=None, user_id="user1", reply_to="session:user1")

        with patch("tsugite.daemon.adapters.base.run_agent", side_effect=fake_run_agent):
            await adapter.handle_message("user1", "hello", ctx)

        assert captured_session_id == "scheduler-session-99"
