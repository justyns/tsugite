"""Tests for Anthropic SDK extended thinking (reasoning_effort)."""

from __future__ import annotations

import json

import httpx
import pytest


def _mock_response(payload: dict) -> httpx.Response:
    resp = httpx.Response(200, content=json.dumps(payload).encode())
    resp._request = httpx.Request("POST", "http://mock")
    return resp


class TestAnthropicEffortModelInfo:
    def test_claude_4_models_declare_effort_levels(self):
        from tsugite.providers.anthropic import _ANTHROPIC_MODELS

        for key in (
            "anthropic/claude-opus-4-6",
            "anthropic/claude-sonnet-4-6",
            "anthropic/claude-haiku-4-5",
            "anthropic/claude-opus-4-5",
            "anthropic/claude-sonnet-4-5",
        ):
            info = _ANTHROPIC_MODELS.get(key)
            assert info is not None, f"missing: {key}"
            assert info.supported_effort_levels == ["low", "medium", "high", "max"], key

    def test_claude_3_models_have_no_effort_levels(self):
        from tsugite.providers.anthropic import _ANTHROPIC_MODELS

        for key in (
            "anthropic/claude-3-7-sonnet-20250219",
            "anthropic/claude-3-opus-20240229",
            "anthropic/claude-3-haiku-20240307",
        ):
            info = _ANTHROPIC_MODELS.get(key)
            assert info is not None, f"missing: {key}"
            assert info.supported_effort_levels is None, key


class TestAnthropicThinkingRequestBody:
    """Verify reasoning_effort is translated to the `thinking` request parameter."""

    @pytest.mark.asyncio
    async def test_reasoning_effort_sets_thinking_budget(self, monkeypatch):
        from tsugite.providers.anthropic import AnthropicProvider

        provider = AnthropicProvider(api_key="sk-test")

        captured = {}

        async def fake_post(self, url, json=None, headers=None):  # noqa: A002
            captured["body"] = json
            return _mock_response(
                {
                    "content": [{"type": "text", "text": "ok"}],
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                }
            )

        monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

        await provider.acompletion(
            messages=[{"role": "user", "content": "hi"}],
            model="claude-opus-4-6",
            reasoning_effort="high",
        )
        body = captured["body"]
        assert "thinking" in body
        assert body["thinking"]["type"] == "enabled"
        assert body["thinking"]["budget_tokens"] == 16384

    @pytest.mark.asyncio
    async def test_each_effort_level_maps_to_expected_budget(self, monkeypatch):
        from tsugite.providers.anthropic import AnthropicProvider

        provider = AnthropicProvider(api_key="sk-test")

        captured = {}

        async def fake_post(self, url, json=None, headers=None):  # noqa: A002
            captured["body"] = json
            return _mock_response(
                {
                    "content": [{"type": "text", "text": "ok"}],
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                }
            )

        monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

        mapping = {"low": 2048, "medium": 8192, "high": 16384, "max": 32768}
        for level, expected in mapping.items():
            await provider.acompletion(
                messages=[{"role": "user", "content": "hi"}],
                model="claude-opus-4-6",
                reasoning_effort=level,
            )
            assert captured["body"]["thinking"]["budget_tokens"] == expected, level

    @pytest.mark.asyncio
    async def test_no_reasoning_effort_omits_thinking_block(self, monkeypatch):
        from tsugite.providers.anthropic import AnthropicProvider

        provider = AnthropicProvider(api_key="sk-test")

        captured = {}

        async def fake_post(self, url, json=None, headers=None):  # noqa: A002
            captured["body"] = json
            return _mock_response(
                {
                    "content": [{"type": "text", "text": "ok"}],
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                }
            )

        monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

        await provider.acompletion(
            messages=[{"role": "user", "content": "hi"}],
            model="claude-opus-4-6",
        )
        assert "thinking" not in captured["body"]


class TestAnthropicThinkingStreaming:
    """Streaming path yields thinking text as it arrives."""

    @pytest.mark.asyncio
    async def test_thinking_delta_yields_reasoning_chunk(self, monkeypatch):
        from tsugite.providers.anthropic import AnthropicProvider
        from tsugite.providers.base import StreamChunk

        provider = AnthropicProvider(api_key="sk-test")

        sse_lines = [
            "event: message_start",
            'data: {"type":"message_start","message":{"usage":{"input_tokens":5}}}',
            "event: content_block_start",
            'data: {"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}',
            "event: content_block_delta",
            'data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"reasoning A "}}',
            "event: content_block_delta",
            'data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"reasoning B"}}',
            "event: content_block_stop",
            'data: {"type":"content_block_stop","index":0}',
            "event: content_block_start",
            'data: {"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}',
            "event: content_block_delta",
            'data: {"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"answer"}}',
            "event: message_delta",
            'data: {"type":"message_delta","usage":{"output_tokens":10}}',
            "event: message_stop",
            'data: {"type":"message_stop"}',
        ]

        class FakeStream:
            def __init__(self, *a, **kw):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            def raise_for_status(self):
                pass

            async def aiter_lines(self):
                for line in sse_lines:
                    yield line

        def fake_stream(self, method, url, **kw):
            return FakeStream()

        monkeypatch.setattr(httpx.AsyncClient, "stream", fake_stream)

        chunks: list[StreamChunk] = []
        result = await provider.acompletion(
            messages=[{"role": "user", "content": "hi"}],
            model="claude-opus-4-6",
            reasoning_effort="medium",
            stream=True,
        )
        async for chunk in result:
            chunks.append(chunk)

        reasoning_text = "".join(c.reasoning_content or "" for c in chunks)
        text = "".join(c.content for c in chunks)

        assert "reasoning A reasoning B" in reasoning_text
        assert "answer" in text
