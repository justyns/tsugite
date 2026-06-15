"""Streaming usage-accounting tests for the OpenAI-compatible provider."""

import pytest

from tsugite.providers.openai_compat import OpenAICompatProvider


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


@pytest.mark.asyncio
async def test_streaming_usage_captured_when_on_finish_chunk_with_choices(monkeypatch):
    """vLLM/Azure (and some OpenAI-compat servers) attach usage to the finish chunk that
    STILL carries a choices entry. Usage must be captured anyway — it was previously
    dropped (guard required empty choices), recording a zero-token/zero-cost turn."""
    lines = [
        'data: {"choices": [{"delta": {"content": "hi"}}]}',
        'data: {"choices": [{"delta": {}, "finish_reason": "stop"}], '
        '"usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}}',
        "data: [DONE]",
    ]
    provider = OpenAICompatProvider("openai", "http://test")
    monkeypatch.setattr(provider, "_get_client", lambda: _FakeClient(lines))

    chunks = [c async for c in provider._stream("http://test/chat", {}, {"model": "gpt-4o"})]

    content = "".join(c.content for c in chunks if c.content)
    assert content == "hi"
    done = chunks[-1]
    assert done.done is True
    assert done.usage is not None, "usage on a finish chunk with choices must not be dropped"
    assert done.usage.total_tokens == 120
