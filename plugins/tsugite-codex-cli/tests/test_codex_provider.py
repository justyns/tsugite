"""Tests for CodexCliProvider (cases P1-P5 from the plan)."""

from __future__ import annotations

import json
from contextlib import contextmanager
from typing import Any
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from tsugite_codex_cli.auth import CodexAuthError
from tsugite_codex_cli.provider import CodexCliProvider, CodexResponsesError


@contextmanager
def _patched_auth(provider: CodexCliProvider, access: str = "AT", account: str = "acct-42"):
    async def fake_get_access_token():
        return access, account

    with patch.object(provider._auth, "get_access_token", side_effect=fake_get_access_token):
        yield


@contextmanager
def _patched_auth_error(provider: CodexCliProvider, message: str = "Run codex login"):
    async def raises():
        raise CodexAuthError(message)

    with patch.object(provider._auth, "get_access_token", side_effect=raises):
        yield


class _FakeResp:
    def __init__(self, status_code: int = 200, json_body: Any = None, text: str = ""):
        self.status_code = status_code
        self._json = json_body
        self.text = text

    def json(self):
        if self._json is None:
            raise ValueError("no json body")
        return self._json

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("err", request=None, response=None)


class _FakeStreamCtx:
    """Async context manager that mimics httpx.AsyncClient.stream() responses.

    Matches httpx's lazy-body semantics: .text and .json() raise ResponseNotRead
    until aread() has been awaited.
    """

    def __init__(
        self, *, status_code: int = 200, events: list[str] | None = None, json_body: Any = None, text: str = ""
    ):
        self.status_code = status_code
        self._events = events or []
        self._json = json_body
        self._text = text
        self._read = False

    async def aread(self):
        self._read = True

    def json(self):
        if not self._read:
            raise httpx.ResponseNotRead()
        if self._json is None:
            raise ValueError("no json body")
        return self._json

    @property
    def text(self):
        if not self._read:
            raise httpx.ResponseNotRead()
        return self._text

    async def aiter_lines(self):
        for ev in self._events:
            yield ev

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False


def _stream_patch(captured: dict | None = None, **ctx_kwargs):
    """Build a patch.object replacement that captures json/headers and returns _FakeStreamCtx."""

    def fake_stream(self_, method, url, json=None, headers=None, **kwargs):
        if captured is not None:
            captured["url"] = url
            captured["json"] = json
            captured["headers"] = headers
        return _FakeStreamCtx(**ctx_kwargs)

    return patch.object(httpx.AsyncClient, "stream", new=fake_stream)


# ── P1: request body translation ──
@pytest.mark.asyncio
async def test_request_body_user_text_and_image_translated():
    provider = CodexCliProvider()
    captured: dict = {}

    with _patched_auth(provider), _stream_patch(captured):
        await provider.acompletion(
            messages=[
                {"role": "system", "content": "You are helpful."},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "describe this"},
                        {"type": "image_url", "image_url": {"url": "https://example/x.png", "format": "image/png"}},
                    ],
                },
            ],
            model="gpt-5.4",
            stream=False,
            max_tokens=1024,
            reasoning_effort="high",
        )

    body = captured["json"]
    assert body["model"] == "gpt-5.4"
    assert body["instructions"] == "You are helpful."
    assert isinstance(body["input"], list)
    assert len(body["input"]) == 1
    user_msg = body["input"][0]
    assert user_msg["role"] == "user"
    assert {b["type"] for b in user_msg["content"]} == {"input_text", "input_image"}
    # kwargs normalised
    assert "max_tokens" not in body
    assert body["max_output_tokens"] == 1024
    assert body["reasoning"] == {"effort": "high"}
    assert "reasoning_effort" not in body
    # headers
    headers = captured["headers"]
    assert headers["Authorization"] == "Bearer AT"
    assert headers["ChatGPT-Account-ID"] == "acct-42"


@pytest.mark.asyncio
async def test_request_body_assistant_history_uses_input_text_not_output_text():
    provider = CodexCliProvider()
    captured: dict = {}

    with _patched_auth(provider), _stream_patch(captured):
        await provider.acompletion(
            messages=[
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": [{"type": "text", "text": "hello back"}]},
                {"role": "user", "content": "follow up"},
            ],
            model="gpt-5.4",
        )

    assistant = next(m for m in captured["json"]["input"] if m["role"] == "assistant")
    assert assistant["content"] == [{"type": "input_text", "text": "hello back"}]
    # Critical: never emit output_text on request input
    for msg in captured["json"]["input"]:
        if isinstance(msg["content"], list):
            for block in msg["content"]:
                assert block["type"] != "output_text"


# ── Fix: Codex backend requires `instructions` even when there's no system message ──
@pytest.mark.asyncio
async def test_request_body_sets_instructions_when_no_system_message():
    """ChatGPT subscription backend 400s with `Instructions are required` if the
    field is omitted entirely. Default to an empty string when no system message exists."""
    provider = CodexCliProvider()
    captured: dict = {}

    with _patched_auth(provider), _stream_patch(captured):
        await provider.acompletion(messages=[{"role": "user", "content": "hi"}], model="gpt-5.4")

    assert "instructions" in captured["json"]
    assert captured["json"]["instructions"] == ""


# ── P1b: regression - Codex backend requires store=false ──
@pytest.mark.asyncio
async def test_request_body_sets_store_false():
    """ChatGPT subscription backend rejects requests without `store: false`."""
    provider = CodexCliProvider()
    captured: dict = {}

    with _patched_auth(provider), _stream_patch(captured):
        await provider.acompletion(messages=[{"role": "user", "content": "hi"}], model="gpt-5.4")

    assert captured["json"]["store"] is False


# ── P1c: regression - Codex backend requires stream=true even for non-stream callers ──
@pytest.mark.asyncio
async def test_non_stream_call_internally_streams_and_collects():
    """ChatGPT subscription backend rejects requests without `stream: true`.

    Tsugite callers can still pass stream=False; the provider must stream
    internally and assemble a CompletionResponse from the chunks.
    """
    provider = CodexCliProvider()
    captured: dict = {}

    events = [
        'data: {"type":"response.output_text.delta","delta":"Hello "}',
        'data: {"type":"response.output_text.delta","delta":"world"}',
        'data: {"type":"response.completed","response":{"usage":{"input_tokens":3,"output_tokens":2}}}',
    ]

    class _FakeStream:
        def __init__(self):
            self.status_code = 200

        async def aiter_lines(self):
            for ev in events:
                yield ev

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

    def fake_stream(self_, method, url, json=None, headers=None, **kwargs):
        captured["json"] = json
        return _FakeStream()

    with _patched_auth(provider), patch.object(httpx.AsyncClient, "stream", new=fake_stream):
        resp = await provider.acompletion(messages=[{"role": "user", "content": "hi"}], model="gpt-5.4", stream=False)

    assert captured["json"]["stream"] is True  # backend always sees stream=true
    # But the caller (stream=False) gets a CompletionResponse, not an async iterator
    assert resp.content == "Hello world"
    assert resp.usage.prompt_tokens == 3
    assert resp.usage.completion_tokens == 2
    assert resp.usage.total_tokens == 5


# ── P2: usage mapping (both stream and non-stream collect via _build_usage) ──
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "usage_in, expected_total",
    [
        ({"input_tokens": 100, "output_tokens": 50, "total_tokens": 200}, 200),  # upstream-provided
        ({"input_tokens": 100, "output_tokens": 50}, 150),  # fallback to input+output
    ],
)
async def test_collected_response_usage_and_total_tokens(usage_in, expected_total):
    provider = CodexCliProvider()
    usage_with_details = {**usage_in, "output_tokens_details": {"reasoning_tokens": 7}}
    events = [
        'data: {"type":"response.output_text.delta","delta":"Hello, "}',
        'data: {"type":"response.output_text.delta","delta":"world."}',
        'data: {"type":"response.completed","response":{"usage":' + json.dumps(usage_with_details) + "}}",
    ]

    class _FakeStream:
        def __init__(self):
            self.status_code = 200

        async def aiter_lines(self):
            for ev in events:
                yield ev

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

    def fake_stream(self_, method, url, **kwargs):
        return _FakeStream()

    with _patched_auth(provider), patch.object(httpx.AsyncClient, "stream", new=fake_stream):
        resp = await provider.acompletion(messages=[{"role": "user", "content": "hi"}], model="gpt-5.4")

    assert resp.content == "Hello, world."
    assert resp.usage.prompt_tokens == 100
    assert resp.usage.completion_tokens == 50
    assert resp.usage.total_tokens == expected_total
    assert resp.usage.reasoning_tokens == 7


# ── P3: streaming ──
@pytest.mark.asyncio
async def test_streaming_accumulates_deltas_and_emits_final_usage():
    provider = CodexCliProvider()

    events = [
        'data: {"type":"response.created"}',
        'data: {"type":"response.output_text.delta","delta":"Hel"}',
        'data: {"type":"response.in_progress"}',
        'data: {"type":"response.output_text.delta","delta":"lo"}',
        'data: {"type":"response.completed","response":{"usage":{"input_tokens":12,"output_tokens":4}}}',
    ]

    class _FakeStream:
        def __init__(self):
            self.status_code = 200

        async def aiter_lines(self):
            for ev in events:
                yield ev

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        def raise_for_status(self):
            return None

    def fake_stream(self_, method, url, **kwargs):
        return _FakeStream()

    with _patched_auth(provider), patch.object(httpx.AsyncClient, "stream", new=fake_stream):
        chunks = []
        result = await provider.acompletion(messages=[{"role": "user", "content": "hi"}], model="gpt-5.4", stream=True)
        async for chunk in result:
            chunks.append(chunk)

    deltas = [c.content for c in chunks if not c.done]
    assert deltas == ["Hel", "lo"]
    final = chunks[-1]
    assert final.done
    assert final.usage.prompt_tokens == 12
    assert final.usage.completion_tokens == 4
    assert final.usage.total_tokens == 16  # fallback path


# ── P4: error response parsing ──
@pytest.mark.asyncio
async def test_error_response_with_json_body_raises_typed_error():
    provider = CodexCliProvider()
    body = {"error": {"message": "context length exceeded", "type": "invalid_request_error"}}
    with _patched_auth(provider), _stream_patch(status_code=400, json_body=body):
        with pytest.raises(CodexResponsesError) as exc:
            await provider.acompletion(messages=[{"role": "user", "content": "x"}], model="gpt-5.4")

    assert "context length exceeded" in str(exc.value)
    assert exc.value.status_code == 400


@pytest.mark.asyncio
async def test_error_response_with_plain_text_body_raises_typed_error():
    provider = CodexCliProvider()
    with _patched_auth(provider), _stream_patch(status_code=503, json_body=None, text="upstream gateway error"):
        with pytest.raises(CodexResponsesError) as exc:
            await provider.acompletion(messages=[{"role": "user", "content": "x"}], model="gpt-5.4")

    assert exc.value.status_code == 503
    assert "upstream gateway error" in str(exc.value) or "503" in str(exc.value)


# ── P5: list_models semantics ──
@pytest.mark.asyncio
async def test_list_models_falls_back_on_auth_error():
    """Per the README, list_models returns the static fallback on any failure,
    including missing/bad auth (so model-picker UIs don't crash before login)."""
    provider = CodexCliProvider()
    with _patched_auth_error(provider):
        models = await provider.list_models()
    assert models == ["gpt-5.5", "gpt-5.4", "gpt-5.4-mini", "gpt-5.4-nano"]


@pytest.mark.asyncio
async def test_list_models_falls_back_on_network_error():
    provider = CodexCliProvider()

    async def boom(*a, **k):
        raise httpx.ConnectError("nope")

    with _patched_auth(provider), patch.object(httpx.AsyncClient, "get", new=AsyncMock(side_effect=boom)):
        models = await provider.list_models()

    assert models == ["gpt-5.5", "gpt-5.4", "gpt-5.4-mini", "gpt-5.4-nano"]


# ── Fix #2: stop() must not break the shared client when concurrent agents share it ──
@pytest.mark.asyncio
async def test_stop_does_not_close_shared_client():
    """cacheable=True means the registry hands the same provider to concurrent
    agents. One agent's cleanup must not aclose the client the other is mid-stream on."""
    provider = CodexCliProvider()
    client_before = provider._get_client()
    await provider.stop()
    assert not client_before.is_closed, "stop() must not close the shared httpx client"


# ── Fix #3: mid-stream response.failed → CodexResponsesError ──
@pytest.mark.asyncio
async def test_mid_stream_response_failed_raises_typed_error():
    provider = CodexCliProvider()
    events = [
        'data: {"type":"response.output_text.delta","delta":"part"}',
        'data: {"type":"response.failed","response":{"error":{"message":"content filter tripped","code":"safety"}}}',
    ]
    with _patched_auth(provider), _stream_patch(events=events):
        with pytest.raises(CodexResponsesError) as exc:
            await provider.acompletion(messages=[{"role": "user", "content": "x"}], model="gpt-5.4")
    assert "content filter" in str(exc.value).lower()


@pytest.mark.asyncio
async def test_mid_stream_error_event_raises_typed_error():
    """The Responses API also signals failure with a bare `error` event/type."""
    provider = CodexCliProvider()
    events = [
        'data: {"type":"response.output_text.delta","delta":"hi"}',
        'data: {"type":"error","message":"upstream blew up","code":"server_error"}',
    ]
    with _patched_auth(provider), _stream_patch(events=events):
        with pytest.raises(CodexResponsesError) as exc:
            await provider.acompletion(messages=[{"role": "user", "content": "x"}], model="gpt-5.4")
    assert "upstream blew up" in str(exc.value).lower()


# ── Fix #4: reasoning content extracted from reasoning_summary_text.delta ──
@pytest.mark.asyncio
async def test_reasoning_content_extracted_from_summary_text_deltas():
    provider = CodexCliProvider()
    events = [
        'data: {"type":"response.reasoning_summary_text.delta","delta":"Thinking "}',
        'data: {"type":"response.reasoning_summary_text.delta","delta":"about it..."}',
        'data: {"type":"response.output_text.delta","delta":"Answer."}',
        'data: {"type":"response.completed","response":{"usage":{"input_tokens":5,"output_tokens":3}}}',
    ]
    with _patched_auth(provider), _stream_patch(events=events):
        resp = await provider.acompletion(messages=[{"role": "user", "content": "x"}], model="gpt-5.4")
    assert resp.content == "Answer."
    assert resp.reasoning_content == "Thinking about it..."


@pytest.mark.asyncio
async def test_reasoning_content_streams_as_chunks():
    """Stream callers see reasoning_content on intermediate StreamChunks too."""
    provider = CodexCliProvider()
    events = [
        'data: {"type":"response.reasoning_summary_text.delta","delta":"Hmm "}',
        'data: {"type":"response.output_text.delta","delta":"OK"}',
        'data: {"type":"response.completed","response":{"usage":{"input_tokens":1,"output_tokens":1}}}',
    ]
    with _patched_auth(provider), _stream_patch(events=events):
        result = await provider.acompletion(messages=[{"role": "user", "content": "x"}], model="gpt-5.4", stream=True)
        reasoning_seen, content_seen = [], []
        async for chunk in result:
            if chunk.reasoning_content:
                reasoning_seen.append(chunk.reasoning_content)
            if chunk.content:
                content_seen.append(chunk.content)
    assert "".join(reasoning_seen) == "Hmm "
    assert "".join(content_seen) == "OK"


# ── Fix #8: count_tokens uses an o200k-compatible encoder for gpt-5.x ──
def test_count_tokens_uses_o200k_for_gpt5():
    """gpt-5.x falls outside base._O200K_PREFIXES so default_count_tokens
    silently uses cl100k_base. Confirm the provider overrides correctly."""
    try:
        import tiktoken
    except ImportError:
        pytest.skip("tiktoken not installed")

    provider = CodexCliProvider()
    text = "日本語のテキストを少し"  # o200k=9, cl100k=11
    o200k = tiktoken.get_encoding("o200k_base")
    cl100k = tiktoken.get_encoding("cl100k_base")
    expected_o200k = len(o200k.encode(text))

    # Sanity: encoders disagree on this input so the next assert is meaningful
    assert expected_o200k != len(cl100k.encode(text))

    assert provider.count_tokens(text, "gpt-5.4") == expected_o200k


# ── Registry integration ──
@pytest.mark.asyncio
async def test_provider_resolves_via_entry_points():
    from tsugite.providers import get_provider, list_all_providers

    assert "codex_cli" in list_all_providers()
    provider = get_provider("codex_cli")
    assert provider.name == "codex_cli"
    assert isinstance(provider, CodexCliProvider)
