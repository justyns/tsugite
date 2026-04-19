"""Tests for Claude Code attachment handling and session ID fixes."""

import pytest

from tsugite.attachments.base import Attachment, AttachmentContentType
from tsugite.providers.claude_code import ClaudeCodeProvider
from tsugite.skill_discovery import Skill


class TestClaudeCodeFirstMessageAttachments:
    def _build(self, attachments=None, skills=None, task="do something"):
        provider = ClaudeCodeProvider()
        provider.set_context(attachments=attachments or [], skills=skills or [])
        messages = [{"role": "user", "content": task}]
        return provider._build_first_message(messages)

    def _att(self, name="test.md", content="test", content_type=AttachmentContentType.TEXT):
        return Attachment(name=name, content=content, content_type=content_type, mime_type="text/plain")

    def test_fresh_session_includes_attachments(self):
        att = self._att(name="MEMORY.md", content="memory content")
        msg = self._build(attachments=[att])

        assert '<attachment name="MEMORY.md">' in msg
        assert "memory content" in msg
        assert "<context>" in msg

    def test_fresh_session_includes_skills(self):
        skill = Skill(name="my-skill", content="skill instructions")
        msg = self._build(skills=[skill])

        assert '<skill_content name="my-skill">' in msg
        assert "skill instructions" in msg

    def test_fresh_session_includes_both(self):
        att = self._att(name="USER.md", content="user prefs")
        skill = Skill(name="helper", content="help text")
        msg = self._build(attachments=[att], skills=[skill])

        assert '<attachment name="USER.md">' in msg
        assert '<skill_content name="helper">' in msg
        assert "<context>" in msg

    def test_no_attachments_no_context_block(self):
        msg = self._build(task="task")
        assert "<context>" not in msg
        assert "task" in msg

    def test_non_text_attachments_excluded(self):
        att = self._att(name="image.png", content="base64data", content_type=AttachmentContentType.IMAGE)
        msg = self._build(attachments=[att])
        assert "<context>" not in msg

    def test_large_attachments_not_truncated(self):
        large_content = "x" * 5000
        att = self._att(name="BIG.md", content=large_content)
        msg = self._build(attachments=[att])

        assert "x" * 5000 in msg
        assert "truncated" not in msg

    def test_small_attachments_not_truncated(self):
        att = self._att(name="small.md", content="short content")
        msg = self._build(attachments=[att])

        assert "short content" in msg
        assert "truncated" not in msg


class TestClaudeCodeSessionId:
    @pytest.mark.asyncio
    async def test_session_id_captured_from_result(self):
        from tsugite.core.agent import TsugiteAgent

        agent = TsugiteAgent(
            model_string="claude_code:sonnet",
            tools=[],
            instructions="test",
            max_turns=1,
        )
        state = agent._provider.get_state()
        assert state is not None
        assert state["session_id"] is None
