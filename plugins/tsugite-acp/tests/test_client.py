"""Slices 4-8: ACPClientSession handshake, turn streaming, cancellation."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

# ── Slice 4: handshake ──


class TestHandshake:
    @pytest.mark.asyncio
    async def test_initialize_then_new_session(self, mock_conn):
        from tsugite_acp.client import ACPClientHandler, ACPClientSession

        session = ACPClientSession(handler=ACPClientHandler(), conn=mock_conn)
        sid = await session.start(cwd="/tmp")

        assert sid == "sess-test-1"
        mock_conn.initialize.assert_awaited_once()
        mock_conn.new_session.assert_awaited_once()
        mock_conn.load_session.assert_not_awaited()

        # client capabilities: fs/terminal must be unsupported (False)
        init_kwargs = mock_conn.initialize.call_args.kwargs
        caps = init_kwargs["client_capabilities"]
        assert caps.fs.read_text_file is False
        assert caps.fs.write_text_file is False
        assert caps.terminal is False

    @pytest.mark.asyncio
    async def test_initialize_then_load_session_when_resuming(self, mock_conn):
        from tsugite_acp.client import ACPClientHandler, ACPClientSession

        session = ACPClientSession(handler=ACPClientHandler(), conn=mock_conn)
        sid = await session.start(cwd="/tmp", resume_session_id="sess-prev-9")

        assert sid == "sess-prev-9"
        mock_conn.initialize.assert_awaited_once()
        mock_conn.load_session.assert_awaited_once()
        mock_conn.new_session.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_session_capabilities_cached(self, mock_conn):
        from tsugite_acp.client import ACPClientHandler, ACPClientSession

        session = ACPClientSession(handler=ACPClientHandler(), conn=mock_conn)
        await session.start(cwd="/tmp")
        assert session.agent_capabilities is not None
        assert session.agent_capabilities.session_capabilities.close is not None


# ── Slice 5: single-turn streaming ──


def _make_message_chunk(text: str):
    from acp.schema import AgentMessageChunk, TextContentBlock

    return AgentMessageChunk(
        content=TextContentBlock(type="text", text=text),
        session_update="agent_message_chunk",
    )


def _make_thought_chunk(text: str):
    from acp.schema import AgentThoughtChunk, TextContentBlock

    return AgentThoughtChunk(
        content=TextContentBlock(type="text", text=text),
        session_update="agent_thought_chunk",
    )


class TestPromptTurn:
    @pytest.mark.asyncio
    async def test_single_turn_streams_text_then_done(self, mock_conn):
        """Handler receives session_updates concurrently with the awaited prompt response."""
        from acp.schema import PromptResponse
        from tsugite_acp.client import ACPClientHandler, ACPClientSession

        handler = ACPClientHandler()
        session = ACPClientSession(handler=handler, conn=mock_conn)
        await session.start(cwd="/tmp")

        async def fake_prompt(*, prompt, session_id, **_):
            await handler.session_update(session_id=session_id, update=_make_message_chunk("hello "))
            await handler.session_update(session_id=session_id, update=_make_message_chunk("world"))
            return PromptResponse(stop_reason="end_turn")

        mock_conn.prompt.side_effect = fake_prompt

        chunks = [c async for c in session.prompt(blocks=[{"type": "text", "text": "hi"}])]
        text_chunks = [c for c in chunks if c.kind == "text"]
        done = [c for c in chunks if c.kind == "done"]

        assert "".join(c.text for c in text_chunks) == "hello world"
        assert len(done) == 1
        assert done[0].stop_reason == "end_turn"

    # ── Slice 6: thought chunks ──
    @pytest.mark.asyncio
    async def test_thought_chunks_become_reasoning(self, mock_conn):
        from acp.schema import PromptResponse
        from tsugite_acp.client import ACPClientHandler, ACPClientSession

        handler = ACPClientHandler()
        session = ACPClientSession(handler=handler, conn=mock_conn)
        await session.start(cwd="/tmp")

        async def fake_prompt(*, prompt, session_id, **_):
            await handler.session_update(session_id=session_id, update=_make_thought_chunk("thinking..."))
            await handler.session_update(session_id=session_id, update=_make_message_chunk("answer"))
            return PromptResponse(stop_reason="end_turn")

        mock_conn.prompt.side_effect = fake_prompt

        chunks = [c async for c in session.prompt(blocks=[{"type": "text", "text": "hi"}])]
        thoughts = [c for c in chunks if c.kind == "thought"]
        texts = [c for c in chunks if c.kind == "text"]

        assert "".join(t.text for t in thoughts) == "thinking..."
        assert "".join(t.text for t in texts) == "answer"


# ── Slice 7: stop-reason error paths ──


class TestStopReasonErrors:
    @pytest.mark.asyncio
    async def test_max_tokens_raises(self, mock_conn):
        from acp.schema import PromptResponse
        from tsugite_acp.client import ACPClientHandler, ACPClientSession

        from tsugite.exceptions import AgentExecutionError

        handler = ACPClientHandler()
        session = ACPClientSession(handler=handler, conn=mock_conn)
        await session.start(cwd="/tmp")
        mock_conn.prompt = AsyncMock(return_value=PromptResponse(stop_reason="max_tokens"))

        with pytest.raises(AgentExecutionError) as ei:
            async for _ in session.prompt(blocks=[{"type": "text", "text": "hi"}]):
                pass
        assert "max_tokens" in str(ei.value)

    @pytest.mark.asyncio
    async def test_refusal_raises(self, mock_conn):
        from acp.schema import PromptResponse
        from tsugite_acp.client import ACPClientHandler, ACPClientSession

        from tsugite.exceptions import AgentExecutionError

        handler = ACPClientHandler()
        session = ACPClientSession(handler=handler, conn=mock_conn)
        await session.start(cwd="/tmp")
        mock_conn.prompt = AsyncMock(return_value=PromptResponse(stop_reason="refusal"))

        with pytest.raises(AgentExecutionError):
            async for _ in session.prompt(blocks=[{"type": "text", "text": "hi"}]):
                pass


# ── Slice 8: cancellation ──


class TestCancellation:
    @pytest.mark.asyncio
    async def test_cancel_calls_session_cancel(self, mock_conn):
        from tsugite_acp.client import ACPClientHandler, ACPClientSession

        session = ACPClientSession(handler=ACPClientHandler(), conn=mock_conn)
        await session.start(cwd="/tmp")
        await session.cancel()
        mock_conn.cancel.assert_awaited_once_with(session_id="sess-test-1")

    @pytest.mark.asyncio
    async def test_close_calls_close_session_when_capability_present(self, mock_conn):
        from tsugite_acp.client import ACPClientHandler, ACPClientSession

        session = ACPClientSession(handler=ACPClientHandler(), conn=mock_conn)
        await session.start(cwd="/tmp")
        await session.close()
        mock_conn.close_session.assert_awaited_once_with(session_id="sess-test-1")
        mock_conn.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_close_skips_close_session_when_capability_absent(self, mock_conn):
        from acp.schema import AgentCapabilities, InitializeResponse, PromptCapabilities, SessionCapabilities
        from tsugite_acp.client import ACPClientHandler, ACPClientSession

        mock_conn.initialize = AsyncMock(
            return_value=InitializeResponse(
                protocol_version=1,
                agent_capabilities=AgentCapabilities(
                    load_session=True,
                    prompt_capabilities=PromptCapabilities(image=True, audio=False, embedded_context=True),
                    session_capabilities=SessionCapabilities(close=None),
                ),
                auth_methods=[],
            )
        )
        session = ACPClientSession(handler=ACPClientHandler(), conn=mock_conn)
        await session.start(cwd="/tmp")
        await session.close()
        mock_conn.close_session.assert_not_awaited()
        mock_conn.close.assert_awaited_once()
