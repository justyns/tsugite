"""Slices 2-3 + later: ACPProvider contract + lifecycle."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest


class TestProtocolContract:
    def test_implements_protocol(self):
        from tsugite_acp.provider import ACPProvider

        from tsugite.providers.base import Provider

        p = ACPProvider()
        assert isinstance(p, Provider)
        assert p.name == "acp"
        assert p.cacheable is False

    def test_create_provider_factory(self):
        from tsugite_acp import create_provider

        p = create_provider(name="acp")
        assert p.name == "acp"
        assert p.cacheable is False

    def test_get_model_info_via_alias(self):
        from tsugite_acp.provider import ACPProvider

        p = ACPProvider()
        info = p.get_model_info("sonnet")
        assert info is not None
        assert info.supports_vision is True

    def test_get_model_info_via_full_id(self):
        from tsugite_acp.provider import ACPProvider

        p = ACPProvider()
        info = p.get_model_info("claude-sonnet-4-6")
        assert info is not None

    @pytest.mark.asyncio
    async def test_list_models_returns_aliases(self):
        from tsugite_acp.provider import ACPProvider

        p = ACPProvider()
        models = await p.list_models()
        assert "opus" in models
        assert "sonnet" in models
        assert "haiku" in models

    def test_count_tokens_uses_default(self):
        from tsugite_acp.provider import ACPProvider

        p = ACPProvider()
        n = p.count_tokens("hello world", "claude-sonnet-4-6")
        assert n > 0


class TestContextRoundTrip:
    def test_set_context_stores_attachments_and_skills(self):
        from tsugite_acp.provider import ACPProvider

        p = ACPProvider()
        p.set_context(
            attachments=["a"],
            skills=["b"],
            previous_messages=[{"role": "user", "content": "hi"}],
            resume_session="sess-abc",
            resume_after_compaction=True,
        )
        assert p._attachments == ["a"]
        assert p._skills == ["b"]
        assert p._previous_messages == [{"role": "user", "content": "hi"}]
        assert p._resume_session == "sess-abc"
        assert p._resume_after_compaction is True

    def test_set_context_defaults_when_keys_absent(self):
        from tsugite_acp.provider import ACPProvider

        p = ACPProvider()
        p.set_context()
        assert p._attachments == []
        assert p._skills == []
        assert p._previous_messages == []
        assert p._resume_session is None
        assert p._resume_after_compaction is False

    def test_get_state_keys(self):
        from tsugite_acp.provider import ACPProvider

        p = ACPProvider()
        p._session_id = "sess-xyz"
        p._cache_creation_tokens = 10
        p._cache_read_tokens = 20
        p._context_window = 200_000

        state = p.get_state()
        assert state["session_id"] == "sess-xyz"
        assert state["cache_creation_tokens"] == 10
        assert state["cache_read_tokens"] == 20
        assert state["context_window"] == 200_000

    @pytest.mark.asyncio
    async def test_stop_is_idempotent_with_no_session(self):
        from tsugite_acp.provider import ACPProvider

        p = ACPProvider()
        await p.stop()  # no session yet - must not raise
        await p.stop()


# ── Slice 11: acompletion end-to-end ──


class TestAcompletion:
    @pytest.fixture
    def patched_provider(self, mock_conn):
        from tsugite_acp.client import ACPClientHandler, ACPClientSession
        from tsugite_acp.provider import ACPProvider

        p = ACPProvider()
        p._session_factory = lambda: ACPClientSession(handler=ACPClientHandler(), conn=mock_conn)
        return p, mock_conn

    @pytest.mark.asyncio
    async def test_collect_concatenates_text_chunks(self, patched_provider):
        from acp.schema import AgentMessageChunk, PromptResponse, TextContentBlock

        p, conn = patched_provider

        async def fake_prompt(*, prompt, session_id, **_):
            await p._session._handler.session_update(
                session_id=session_id,
                update=AgentMessageChunk(
                    content=TextContentBlock(type="text", text="hello "),
                    session_update="agent_message_chunk",
                ),
            )
            await p._session._handler.session_update(
                session_id=session_id,
                update=AgentMessageChunk(
                    content=TextContentBlock(type="text", text="world"),
                    session_update="agent_message_chunk",
                ),
            )
            return PromptResponse(stop_reason="end_turn")

        conn.prompt.side_effect = fake_prompt

        resp = await p.acompletion(
            messages=[{"role": "user", "content": "hi"}],
            model="sonnet",
            stream=False,
        )
        assert resp.content == "hello world"

    @pytest.mark.asyncio
    async def test_stream_yields_chunks_then_done(self, patched_provider):
        from acp.schema import AgentMessageChunk, AgentThoughtChunk, PromptResponse, TextContentBlock

        p, conn = patched_provider

        async def fake_prompt(*, prompt, session_id, **_):
            await p._session._handler.session_update(
                session_id=session_id,
                update=AgentThoughtChunk(
                    content=TextContentBlock(type="text", text="planning..."),
                    session_update="agent_thought_chunk",
                ),
            )
            await p._session._handler.session_update(
                session_id=session_id,
                update=AgentMessageChunk(
                    content=TextContentBlock(type="text", text="done!"),
                    session_update="agent_message_chunk",
                ),
            )
            return PromptResponse(stop_reason="end_turn")

        conn.prompt.side_effect = fake_prompt

        chunks = []
        result = await p.acompletion(
            messages=[{"role": "user", "content": "hi"}],
            model="sonnet",
            stream=True,
        )
        async for c in result:
            chunks.append(c)

        text = "".join(c.content for c in chunks)
        reasoning = "".join(c.reasoning_content for c in chunks)
        assert "done!" in text
        assert "planning..." in reasoning
        assert chunks[-1].done is True

    @pytest.mark.asyncio
    async def test_first_call_uses_new_session(self, patched_provider):
        from acp.schema import PromptResponse

        p, conn = patched_provider
        conn.prompt = AsyncMock(return_value=PromptResponse(stop_reason="end_turn"))

        await p.acompletion(messages=[{"role": "user", "content": "first"}], model="sonnet", stream=False)
        conn.new_session.assert_awaited_once()
        conn.load_session.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_collect_includes_tool_call_activity(self, patched_provider):
        """Tool-call activity must land in the completion content so it's recorded to
        history - otherwise a turn that only ran tools renders empty."""
        from acp.schema import AgentMessageChunk, PromptResponse, TextContentBlock, ToolCallStart

        p, conn = patched_provider

        async def fake_prompt(*, prompt, session_id, **_):
            await p._session._handler.session_update(
                session_id=session_id,
                update=ToolCallStart(title="Launch job: research", tool_call_id="t1", session_update="tool_call"),
            )
            await p._session._handler.session_update(
                session_id=session_id,
                update=AgentMessageChunk(
                    content=TextContentBlock(type="text", text="launched"),
                    session_update="agent_message_chunk",
                ),
            )
            return PromptResponse(stop_reason="end_turn")

        conn.prompt.side_effect = fake_prompt

        resp = await p.acompletion(messages=[{"role": "user", "content": "hi"}], model="sonnet", stream=False)
        assert "Launch job: research" in resp.content
        assert "launched" in resp.content

    @pytest.mark.asyncio
    async def test_collect_preserves_content_on_max_tokens(self, patched_provider):
        """A max_tokens stop must not discard the turn's work - the content collected so
        far is still returned (not raised away)."""
        from acp.schema import AgentMessageChunk, PromptResponse, TextContentBlock

        p, conn = patched_provider

        async def fake_prompt(*, prompt, session_id, **_):
            await p._session._handler.session_update(
                session_id=session_id,
                update=AgentMessageChunk(
                    content=TextContentBlock(type="text", text="did a lot of work"),
                    session_update="agent_message_chunk",
                ),
            )
            return PromptResponse(stop_reason="max_tokens")

        conn.prompt.side_effect = fake_prompt

        resp = await p.acompletion(messages=[{"role": "user", "content": "hi"}], model="sonnet", stream=False)
        assert "did a lot of work" in resp.content

    @pytest.mark.asyncio
    async def test_resume_session_uses_load_session(self, mock_conn):
        from acp.schema import PromptResponse
        from tsugite_acp.client import ACPClientHandler, ACPClientSession
        from tsugite_acp.provider import ACPProvider

        p = ACPProvider()
        p._session_factory = lambda: ACPClientSession(handler=ACPClientHandler(), conn=mock_conn)
        p.set_context(resume_session="sess-prev-7")
        mock_conn.prompt = AsyncMock(return_value=PromptResponse(stop_reason="end_turn"))

        await p.acompletion(messages=[{"role": "user", "content": "go"}], model="sonnet", stream=False)
        mock_conn.load_session.assert_awaited_once()
        mock_conn.new_session.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_session_id_recorded_in_state_after_turn(self, patched_provider):
        from acp.schema import PromptResponse

        p, conn = patched_provider
        conn.prompt = AsyncMock(return_value=PromptResponse(stop_reason="end_turn"))

        await p.acompletion(messages=[{"role": "user", "content": "go"}], model="sonnet", stream=False)
        state = p.get_state()
        assert state["session_id"] == "sess-test-1"
