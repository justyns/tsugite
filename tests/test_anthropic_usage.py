"""Anthropic token accounting: cache tokens must be in the totals (non-streaming + streaming)."""

import pytest

from tsugite.providers.anthropic import AnthropicProvider


class _FakeResp:
    def __init__(self, lines):
        self._lines = lines

    def raise_for_status(self):
        pass

    async def aiter_lines(self):
        for line in self._lines:
            yield line


class _FakeStreamCtx:
    def __init__(self, lines):
        self._lines = lines

    async def __aenter__(self):
        return _FakeResp(self._lines)

    async def __aexit__(self, *exc):
        return False


class _FakeClient:
    def __init__(self, lines):
        self._lines = lines

    def stream(self, *args, **kwargs):
        return _FakeStreamCtx(self._lines)


def test_parse_response_total_includes_cache_tokens():
    provider = AnthropicProvider()
    data = {
        "content": [{"type": "text", "text": "hi"}],
        "usage": {
            "input_tokens": 100,
            "output_tokens": 50,
            "cache_creation_input_tokens": 300,
            "cache_read_input_tokens": 200,
        },
    }
    resp = provider._parse_response(data, "claude-opus-4-8")
    # total must include cache (Anthropic input_tokens is the uncached remainder; cache is additive)
    assert resp.usage.total_tokens == 650
    assert resp.usage.cache_read_input_tokens == 200


@pytest.mark.asyncio
async def test_stream_usage_includes_cache_tokens(monkeypatch):
    lines = [
        'data: {"type": "message_start", "message": {"usage": '
        '{"input_tokens": 100, "cache_creation_input_tokens": 300, "cache_read_input_tokens": 200}}}',
        'data: {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "hi"}}',
        'data: {"type": "message_delta", "usage": {"output_tokens": 50}}',
        'data: {"type": "message_stop"}',
    ]
    provider = AnthropicProvider()
    monkeypatch.setattr(provider, "_get_client", lambda: _FakeClient(lines))
    chunks = [c async for c in provider._stream("http://x", {}, {"model": "claude-opus-4-8"})]
    done = chunks[-1]
    assert done.done is True
    assert done.usage.total_tokens == 650
    assert done.usage.cache_read_input_tokens == 200
    assert done.usage.cache_creation_input_tokens == 300
