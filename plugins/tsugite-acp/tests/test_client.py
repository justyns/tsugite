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


def _make_tool_call_start(title: str, tool_call_id: str = "t1"):
    from acp.schema import ToolCallStart

    return ToolCallStart(title=title, tool_call_id=tool_call_id, session_update="tool_call")


def _make_tool_progress(status: str, title: str | None = None, tool_call_id: str = "t1"):
    from acp.schema import ToolCallProgress

    return ToolCallProgress(title=title, tool_call_id=tool_call_id, status=status, session_update="tool_call_update")


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


# ── Tool-call surfacing ──


class TestToolCalls:
    """Tool-call notifications must surface as `tool` events so the agent's executed
    tool/code activity reaches history instead of being silently dropped."""

    @pytest.mark.asyncio
    async def test_tool_call_start_surfaces_as_event(self, mock_conn):
        from acp.schema import PromptResponse
        from tsugite_acp.client import ACPClientHandler, ACPClientSession

        handler = ACPClientHandler()
        session = ACPClientSession(handler=handler, conn=mock_conn)
        await session.start(cwd="/tmp")

        async def fake_prompt(*, prompt, session_id, **_):
            await handler.session_update(session_id=session_id, update=_make_tool_call_start("Bash git status"))
            await handler.session_update(session_id=session_id, update=_make_message_chunk("done"))
            return PromptResponse(stop_reason="end_turn")

        mock_conn.prompt.side_effect = fake_prompt

        events = [c async for c in session.prompt(blocks=[{"type": "text", "text": "hi"}])]
        tool_events = [c for c in events if c.kind == "tool"]
        assert len(tool_events) == 1
        assert "Bash git status" in tool_events[0].text

    @pytest.mark.asyncio
    async def test_in_progress_tool_update_is_dropped(self, mock_conn):
        """Non-terminal progress ticks are noise; only the call and its terminal status
        surface."""
        from acp.schema import PromptResponse
        from tsugite_acp.client import ACPClientHandler, ACPClientSession

        handler = ACPClientHandler()
        session = ACPClientSession(handler=handler, conn=mock_conn)
        await session.start(cwd="/tmp")

        async def fake_prompt(*, prompt, session_id, **_):
            await handler.session_update(session_id=session_id, update=_make_tool_progress("in_progress"))
            return PromptResponse(stop_reason="end_turn")

        mock_conn.prompt.side_effect = fake_prompt

        events = [c async for c in session.prompt(blocks=[{"type": "text", "text": "hi"}])]
        assert [c for c in events if c.kind == "tool"] == []

    @pytest.mark.asyncio
    async def test_completed_tool_update_surfaces(self, mock_conn):
        from acp.schema import PromptResponse
        from tsugite_acp.client import ACPClientHandler, ACPClientSession

        handler = ACPClientHandler()
        session = ACPClientSession(handler=handler, conn=mock_conn)
        await session.start(cwd="/tmp")

        async def fake_prompt(*, prompt, session_id, **_):
            await handler.session_update(session_id=session_id, update=_make_tool_progress("failed", title="Bash"))
            return PromptResponse(stop_reason="end_turn")

        mock_conn.prompt.side_effect = fake_prompt

        events = [c async for c in session.prompt(blocks=[{"type": "text", "text": "hi"}])]
        tool_events = [c for c in events if c.kind == "tool"]
        assert len(tool_events) == 1
        assert "failed" in tool_events[0].text


# ── Stop-reason handling: preserve content, don't discard the turn ──


class TestStopReasonHandling:
    """A turn that stops for max_tokens / max_turn_requests / refusal did real work;
    its content must be preserved (recorded to history) instead of raising and
    discarding everything."""

    @pytest.mark.asyncio
    async def test_max_tokens_preserves_content(self, mock_conn):
        from acp.schema import PromptResponse
        from tsugite_acp.client import ACPClientHandler, ACPClientSession

        handler = ACPClientHandler()
        session = ACPClientSession(handler=handler, conn=mock_conn)
        await session.start(cwd="/tmp")

        async def fake_prompt(*, prompt, session_id, **_):
            await handler.session_update(session_id=session_id, update=_make_message_chunk("partial answer"))
            return PromptResponse(stop_reason="max_tokens")

        mock_conn.prompt.side_effect = fake_prompt

        events = [c async for c in session.prompt(blocks=[{"type": "text", "text": "hi"}])]
        text = "".join(c.text for c in events if c.kind == "text")
        done = [c for c in events if c.kind == "done"]
        assert "partial answer" in text
        assert len(done) == 1
        assert done[0].stop_reason == "max_tokens"

    @pytest.mark.asyncio
    async def test_refusal_preserves_content(self, mock_conn):
        from acp.schema import PromptResponse
        from tsugite_acp.client import ACPClientHandler, ACPClientSession

        handler = ACPClientHandler()
        session = ACPClientSession(handler=handler, conn=mock_conn)
        await session.start(cwd="/tmp")

        async def fake_prompt(*, prompt, session_id, **_):
            await handler.session_update(session_id=session_id, update=_make_message_chunk("I can't help with that."))
            return PromptResponse(stop_reason="refusal")

        mock_conn.prompt.side_effect = fake_prompt

        events = [c async for c in session.prompt(blocks=[{"type": "text", "text": "hi"}])]
        assert "I can't help with that." in "".join(c.text for c in events if c.kind == "text")
        assert [c for c in events if c.kind == "done"][0].stop_reason == "refusal"

    @pytest.mark.asyncio
    async def test_prompt_rpc_error_still_raises(self, mock_conn):
        """A genuine transport/protocol error (the prompt RPC itself failing) must still
        propagate - that's distinct from a stop reason."""
        from tsugite_acp.client import ACPClientHandler, ACPClientSession

        handler = ACPClientHandler()
        session = ACPClientSession(handler=handler, conn=mock_conn)
        await session.start(cwd="/tmp")
        mock_conn.prompt = AsyncMock(side_effect=RuntimeError("connection dropped"))

        with pytest.raises(RuntimeError, match="connection dropped"):
            async for _ in session.prompt(blocks=[{"type": "text", "text": "hi"}]):
                pass


# ── Large-message stdio buffer limit ──


class TestStdioBufferLimit:
    """A single ACP JSON-RPC line can exceed asyncio's default 64KB StreamReader limit
    (a big tool result / content block), which surfaced as "Separator is found, but
    chunk is longer than limit". spawn_acp_session must raise the subprocess stream
    limit so large lines are read whole."""

    @pytest.mark.asyncio
    async def test_large_line_overflows_default_but_fits_configured_limit(self):
        import asyncio

        from tsugite_acp.client import _STDIO_BUFFER_LIMIT

        line = b'{"jsonrpc":"2.0","result":"' + b"a" * (256 * 1024) + b'"}\n'  # ~256KB > 64KB default

        # The default StreamReader limit (what the subprocess used) rejects the line -
        # this is the exact failure the user saw.
        default_reader = asyncio.StreamReader()
        default_reader.feed_data(line)
        default_reader.feed_eof()
        with pytest.raises(ValueError, match="chunk is longer than limit"):
            await default_reader.readline()

        # The configured limit reads the same line whole.
        big_reader = asyncio.StreamReader(limit=_STDIO_BUFFER_LIMIT)
        big_reader.feed_data(line)
        big_reader.feed_eof()
        assert await big_reader.readline() == line

    @pytest.mark.asyncio
    async def test_spawn_passes_configured_stdio_limit(self, monkeypatch):
        import asyncio
        from unittest.mock import MagicMock

        import tsugite_acp.client as client_mod

        captured: dict = {}

        async def fake_exec(*_args, **kwargs):
            captured.update(kwargs)
            proc = MagicMock()
            proc.stdin, proc.stdout, proc.stderr = MagicMock(), MagicMock(), MagicMock()
            return proc

        async def noop_drain(*_a, **_k):
            return None

        monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
        monkeypatch.setattr(client_mod, "connect_to_agent", lambda *a, **k: MagicMock())
        monkeypatch.setattr(client_mod, "_drain_stream", noop_drain)

        await client_mod.spawn_acp_session(command="/bin/true")  # "/" skips the PATH check
        assert captured.get("limit") == client_mod._STDIO_BUFFER_LIMIT


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
