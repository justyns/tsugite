"""Per-session topic injection into agent context.

Topic is rendered as a dedicated <session_topic> block in both the live
per-message context (BaseAdapter._build_message_context) and the compaction
prompt preamble. It is carved out of the generic <session_metadata> block so
the LLM treats it as info, not instructions.
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest

from tsugite.daemon.adapters.base import BaseAdapter, ChannelContext
from tsugite.daemon.config import AgentConfig
from tsugite.daemon.session_store import Session, SessionSource, SessionStore
from tsugite.history.models import Event


class _StubAdapter(BaseAdapter):
    def get_platform_name(self) -> str:
        return "test"

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass


@pytest.fixture
def adapter(tmp_path):
    ws = tmp_path / "workspace"
    ws.mkdir()
    (ws / "agent.md").write_text("---\nname: test-agent\n---\n\nHi.\n")

    store = SessionStore(tmp_path / "store.json")
    config = AgentConfig(workspace_dir=ws, agent_file=str(ws / "agent.md"))
    return _StubAdapter("test-agent", config, store)


@pytest.fixture
def channel_context():
    return ChannelContext(source="http", channel_id=None, user_id="alice", reply_to="http:alice")


def _seed_session(adapter, **metadata) -> str:
    session = adapter.session_store.get_or_create_interactive("alice", "test-agent")
    if metadata:
        adapter.session_store.set_metadata_bulk(session.id, metadata)
    return session.id


class TestMessageContextTopic:
    def test_topic_emits_dedicated_block(self, adapter, channel_context):
        _seed_session(adapter, topic="focus is plugin sandboxing")
        rendered = adapter._build_message_context("hello", channel_context, "alice")
        assert "<session_topic>" in rendered
        assert "focus is plugin sandboxing" in rendered
        assert "</session_topic>" in rendered

    def test_topic_not_in_generic_metadata_block(self, adapter, channel_context):
        _seed_session(adapter, topic="focus is plugin sandboxing", task="https://x/1")
        rendered = adapter._build_message_context("hello", channel_context, "alice")

        topic_block_start = rendered.index("<session_topic>")
        topic_block_end = rendered.index("</session_topic>") + len("</session_topic>")
        outside_topic = rendered[:topic_block_start] + rendered[topic_block_end:]

        assert "topic=" not in outside_topic
        assert "focus is plugin sandboxing" not in outside_topic
        assert "<session_metadata>" in outside_topic
        assert "task=https://x/1" in outside_topic

    def test_no_topic_no_block(self, adapter, channel_context):
        _seed_session(adapter, task="https://x/1")
        rendered = adapter._build_message_context("hello", channel_context, "alice")
        assert "<session_topic>" not in rendered
        assert "<session_metadata>" in rendered

    def test_only_topic_no_generic_metadata_block(self, adapter, channel_context):
        _seed_session(adapter, topic="reading mode: summarize don't act")
        rendered = adapter._build_message_context("hello", channel_context, "alice")
        assert "<session_topic>" in rendered
        assert "<session_metadata>" not in rendered

    def test_empty_metadata_no_blocks(self, adapter, channel_context):
        _seed_session(adapter)
        rendered = adapter._build_message_context("hello", channel_context, "alice")
        assert "<session_topic>" not in rendered
        assert "<session_metadata>" not in rendered

    def test_topic_block_includes_editable_hint(self, adapter, channel_context):
        _seed_session(adapter, topic="testing")
        rendered = adapter._build_message_context("hello", channel_context, "alice")
        assert "session_metadata(key='topic'" in rendered

    def test_topic_block_inside_message_context(self, adapter, channel_context):
        _seed_session(adapter, topic="testing")
        rendered = adapter._build_message_context("hello", channel_context, "alice")
        mc_start = rendered.index("<message_context>")
        mc_end = rendered.index("</message_context>")
        topic_pos = rendered.index("<session_topic>")
        assert mc_start < topic_pos < mc_end


class TestCompactionTopic:
    """Topic must survive compaction so the post-compaction agent still knows it."""

    @pytest.mark.asyncio
    async def test_topic_appears_in_compaction_prompt(self, adapter, tmp_path):
        from tsugite.history import SessionStorage

        history_dir = tmp_path / "history"
        history_dir.mkdir()

        session = adapter.session_store.get_or_create_interactive("alice", "test-agent")
        adapter.session_store.set_metadata(session.id, "topic", "focus this week is plugin sandboxing")

        session_path = history_dir / f"{session.id}.jsonl"
        storage = SessionStorage.create(
            agent_name="test-agent",
            model="anthropic:claude-sonnet-4-5",
            session_path=session_path,
        )
        for i in range(4):
            storage.record("user_input", text=f"message {i}")
            storage.record("model_response", raw_content=f"reply {i}")

        old_events = [
            Event(type="user_input", ts=datetime.now(timezone.utc), data={"text": f"old {i}"}) for i in range(2)
        ]
        recent_events = [
            Event(type="user_input", ts=datetime.now(timezone.utc), data={"text": f"recent {i}"}) for i in range(2)
        ]

        captured = {}

        async def fake_summarize(messages, model=None, max_context_tokens=None, progress_callback=None):
            captured["messages"] = messages
            return "Summary"

        with (
            patch("tsugite.daemon.memory.get_context_limit", return_value=200_000),
            patch("tsugite.daemon.memory.infer_compaction_model", return_value="anthropic:claude-3-haiku-20240307"),
            patch(
                "tsugite.daemon.memory.split_events_for_compaction",
                return_value=(old_events, recent_events),
            ),
            patch("tsugite.daemon.memory.summarize_session", new=fake_summarize),
            patch("tsugite.history.get_history_dir", return_value=history_dir),
            patch("tsugite.history.storage.get_history_dir", return_value=history_dir),
            patch("tsugite.history.storage.get_machine_name", return_value="test"),
            patch("tsugite.hooks.fire_compact_hooks", new_callable=AsyncMock, return_value=[]),
        ):
            await adapter._compact_session(session.id)

        prompt_text = "\n".join(m.get("content", "") for m in captured["messages"] if isinstance(m, dict))
        assert "<session_topic>" in prompt_text
        assert "focus this week is plugin sandboxing" in prompt_text

    @pytest.mark.asyncio
    async def test_no_topic_no_block_in_compaction(self, adapter, tmp_path):
        from tsugite.history import SessionStorage

        history_dir = tmp_path / "history"
        history_dir.mkdir()

        session = adapter.session_store.get_or_create_interactive("alice", "test-agent")
        # No topic set.

        session_path = history_dir / f"{session.id}.jsonl"
        storage = SessionStorage.create(
            agent_name="test-agent",
            model="anthropic:claude-sonnet-4-5",
            session_path=session_path,
        )
        for i in range(2):
            storage.record("user_input", text=f"m {i}")
            storage.record("model_response", raw_content=f"r {i}")

        old_events = [Event(type="user_input", ts=datetime.now(timezone.utc), data={"text": "old"})]
        recent_events = [Event(type="user_input", ts=datetime.now(timezone.utc), data={"text": "recent"})]

        captured = {}

        async def fake_summarize(messages, model=None, max_context_tokens=None, progress_callback=None):
            captured["messages"] = messages
            return "Summary"

        with (
            patch("tsugite.daemon.memory.get_context_limit", return_value=200_000),
            patch("tsugite.daemon.memory.infer_compaction_model", return_value="anthropic:claude-3-haiku-20240307"),
            patch(
                "tsugite.daemon.memory.split_events_for_compaction",
                return_value=(old_events, recent_events),
            ),
            patch("tsugite.daemon.memory.summarize_session", new=fake_summarize),
            patch("tsugite.history.get_history_dir", return_value=history_dir),
            patch("tsugite.history.storage.get_history_dir", return_value=history_dir),
            patch("tsugite.history.storage.get_machine_name", return_value="test"),
            patch("tsugite.hooks.fire_compact_hooks", new_callable=AsyncMock, return_value=[]),
        ):
            await adapter._compact_session(session.id)

        prompt_text = "\n".join(m.get("content", "") for m in captured["messages"] if isinstance(m, dict))
        assert "<session_topic>" not in prompt_text
