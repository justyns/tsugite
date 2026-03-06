"""Tests for Claude Code attachment and session ID fixes.

Fix 2: Attachments included in _build_claude_code_first_message (fresh sessions only)
Fix 3: claude_code_session_id passed to save_run_to_history from daemon
"""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tsugite.attachments.base import Attachment, AttachmentContentType
from tsugite.core.agent import TsugiteAgent
from tsugite.skill_discovery import Skill


# ── Fix 2: _build_claude_code_first_message attachments ──


class TestClaudeCodeFirstMessageAttachments:
    def _make_agent(self, attachments=None, skills=None, resume_session=None):
        return TsugiteAgent(
            model_string="claude_code:sonnet",
            tools=[],
            instructions="test",
            attachments=attachments,
            skills=skills,
            resume_session=resume_session,
        )

    def _att(self, name="test.md", content="test", content_type=AttachmentContentType.TEXT):
        return Attachment(name=name, content=content, content_type=content_type, mime_type="text/plain")

    def test_fresh_session_includes_attachments(self):
        att = self._att(name="MEMORY.md", content="memory content")
        agent = self._make_agent(attachments=[att])
        agent.memory.task = "do something"

        msg = agent._build_claude_code_first_message()

        assert '<attachment name="MEMORY.md">' in msg
        assert "memory content" in msg
        assert "<context>" in msg

    def test_fresh_session_includes_skills(self):
        skill = Skill(name="my-skill", content="skill instructions")
        agent = self._make_agent(skills=[skill])
        agent.memory.task = "do something"

        msg = agent._build_claude_code_first_message()

        assert '<skill name="my-skill">' in msg
        assert "skill instructions" in msg

    def test_fresh_session_includes_both(self):
        att = self._att(name="USER.md", content="user prefs")
        skill = Skill(name="helper", content="help text")
        agent = self._make_agent(attachments=[att], skills=[skill])
        agent.memory.task = "task"

        msg = agent._build_claude_code_first_message()

        assert '<attachment name="USER.md">' in msg
        assert '<skill name="helper">' in msg
        assert "<context>" in msg

    def test_resumed_session_excludes_attachments(self):
        att = self._att(name="MEMORY.md", content="memory content")
        agent = self._make_agent(attachments=[att], resume_session="old-session-id")
        agent.memory.task = "do something"

        msg = agent._build_claude_code_first_message()

        assert "<attachment" not in msg
        assert "<context>" not in msg
        assert "do something" in msg

    def test_resumed_session_excludes_skills(self):
        skill = Skill(name="my-skill", content="skill instructions")
        agent = self._make_agent(skills=[skill], resume_session="old-session-id")
        agent.memory.task = "do something"

        msg = agent._build_claude_code_first_message()

        assert "<skill" not in msg
        assert "<context>" not in msg

    def test_no_attachments_no_context_block(self):
        agent = self._make_agent()
        agent.memory.task = "task"

        msg = agent._build_claude_code_first_message()

        assert "<context>" not in msg
        assert "task" in msg

    def test_non_text_attachments_excluded(self):
        att = self._att(name="image.png", content="base64data", content_type=AttachmentContentType.IMAGE)
        agent = self._make_agent(attachments=[att])
        agent.memory.task = "task"

        msg = agent._build_claude_code_first_message()

        assert "<context>" not in msg

    def test_task_always_present(self):
        att = self._att(name="doc.md", content="doc")
        agent = self._make_agent(attachments=[att], resume_session="sess")
        agent.memory.task = "my actual task"

        msg = agent._build_claude_code_first_message()

        assert "my actual task" in msg


# ── Fix 3: daemon passes claude_code_session_id to history ──


class TestDaemonSessionIdPassthrough:
    @pytest.mark.asyncio
    async def test_handle_message_passes_session_id_to_history(self):
        from tsugite.daemon.adapters.base import BaseAdapter, ChannelContext
        from tsugite.daemon.config import AgentConfig
        from tsugite.daemon.session import SessionManager

        agent_config = AgentConfig(
            agent_file="default",
            workspace_dir="/tmp/test-workspace",
        )
        session_mgr = SessionManager(agent_name="test-agent", workspace_dir=Path("/tmp/test-workspace"))

        # Create a concrete subclass since BaseAdapter is abstract
        class TestAdapter(BaseAdapter):
            async def start(self):
                pass

            async def stop(self):
                pass

        adapter = TestAdapter(
            agent_name="test-agent",
            agent_config=agent_config,
            session_manager=session_mgr,
        )

        channel_ctx = ChannelContext(source="test", channel_id="ch1", user_id="user1", reply_to="test:ch1")

        mock_result = MagicMock()
        mock_result.__str__ = MagicMock(return_value="response text")
        mock_result.token_count = 500
        mock_result.cost = 0.01
        mock_result.execution_steps = []
        mock_result.system_message = "system"
        mock_result.attachments = []
        mock_result.claude_code_session_id = "cc-session-abc"
        mock_result.context_window = None

        with (
            patch.object(adapter, "_get_workspace_attachments", return_value=[]),
            patch.object(adapter, "_resolve_agent_path", return_value=MagicMock()),
            patch("tsugite.daemon.adapters.base.run_agent", return_value=mock_result),
            patch("tsugite.daemon.adapters.base.os.getcwd", return_value="/tmp"),
            patch("tsugite.daemon.adapters.base.os.chdir"),
            patch("tsugite.agent_runner.history_integration.save_run_to_history") as mock_save,
            patch("tsugite.agent_runner.validation.get_agent_info", return_value={"model": "test"}),
        ):
            await adapter.handle_message("user1", "hello", channel_ctx)

            mock_save.assert_called_once()
            call_kwargs = mock_save.call_args[1]
            assert call_kwargs["claude_code_session_id"] == "cc-session-abc"

    @pytest.mark.asyncio
    async def test_handle_message_passes_none_session_id_for_litellm(self):
        from tsugite.daemon.adapters.base import BaseAdapter, ChannelContext
        from tsugite.daemon.config import AgentConfig
        from tsugite.daemon.session import SessionManager

        agent_config = AgentConfig(
            agent_file="default",
            workspace_dir="/tmp/test-workspace",
        )
        session_mgr = SessionManager(agent_name="test-agent", workspace_dir=Path("/tmp/test-workspace"))

        class TestAdapter(BaseAdapter):
            async def start(self):
                pass

            async def stop(self):
                pass

        adapter = TestAdapter(
            agent_name="test-agent",
            agent_config=agent_config,
            session_manager=session_mgr,
        )

        channel_ctx = ChannelContext(source="test", channel_id="ch1", user_id="user1", reply_to="test:ch1")

        mock_result = MagicMock()
        mock_result.__str__ = MagicMock(return_value="response text")
        mock_result.token_count = 500
        mock_result.cost = 0.01
        mock_result.execution_steps = []
        mock_result.system_message = "system"
        mock_result.attachments = []
        mock_result.context_window = None
        # LiteLLM result won't have claude_code_session_id attribute
        del mock_result.claude_code_session_id

        with (
            patch.object(adapter, "_get_workspace_attachments", return_value=[]),
            patch.object(adapter, "_resolve_agent_path", return_value=MagicMock()),
            patch("tsugite.daemon.adapters.base.run_agent", return_value=mock_result),
            patch("tsugite.daemon.adapters.base.os.getcwd", return_value="/tmp"),
            patch("tsugite.daemon.adapters.base.os.chdir"),
            patch("tsugite.agent_runner.history_integration.save_run_to_history") as mock_save,
            patch("tsugite.agent_runner.validation.get_agent_info", return_value={"model": "test"}),
        ):
            await adapter.handle_message("user1", "hello", channel_ctx)

            mock_save.assert_called_once()
            call_kwargs = mock_save.call_args[1]
            assert call_kwargs["claude_code_session_id"] is None


# ── Fix 1: workspace attachments built fresh per message ──


class TestWorkspaceAttachmentsFresh:
    def test_get_workspace_attachments_called_per_message(self):
        """Verify _get_workspace_attachments is a method call, not a cached attribute."""
        from tsugite.daemon.adapters.base import BaseAdapter

        assert callable(getattr(BaseAdapter, "_get_workspace_attachments", None))
        assert not hasattr(BaseAdapter, "workspace_attachments")
