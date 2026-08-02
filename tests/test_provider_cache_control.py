"""The `cache_control` cache-breakpoint hint contract.

`core/agent.py` marks context-turn messages with a provider-neutral
`cache_control` hint. Each provider translates it for its own API: Anthropic moves
it onto a content block (where the Messages API takes it), OpenAI-schema endpoints
cache automatically and drop it - Fireworks and other strict ones reject the
unknown field with HTTP 400.
"""

from __future__ import annotations

import json

import httpx
import pytest

EPHEMERAL = {"type": "ephemeral"}

ANTHROPIC_REPLY = {"content": [{"type": "text", "text": "ok"}], "usage": {"input_tokens": 1, "output_tokens": 1}}
OPENAI_REPLY = {"choices": [{"message": {"content": "ok"}}], "usage": {"prompt_tokens": 1, "completion_tokens": 1}}


def _mock_response(url: str, payload: dict) -> httpx.Response:
    resp = httpx.Response(200, content=json.dumps(payload).encode())
    resp._request = httpx.Request("POST", url)
    return resp


def _capture_post(monkeypatch, reply: dict) -> dict:
    """Intercept the provider's HTTP call and return a dict holding the sent body."""
    captured: dict = {}

    async def fake_post(self, url, json, headers):  # noqa: A002 - shadows the module, matches httpx's kwarg
        captured["body"] = json
        return _mock_response(url, reply)

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)
    return captured


async def _anthropic_body(monkeypatch, messages: list[dict]) -> dict:
    from tsugite.providers.anthropic import AnthropicProvider

    captured = _capture_post(monkeypatch, ANTHROPIC_REPLY)
    await AnthropicProvider(api_key="sk-test").acompletion(messages=messages, model="claude-sonnet-4-6")
    return captured["body"]


async def _openai_compat_body(monkeypatch, messages: list[dict]) -> dict:
    from tsugite.providers.openai_compat import OpenAICompatProvider

    captured = _capture_post(monkeypatch, OPENAI_REPLY)
    provider = OpenAICompatProvider("fireworks", "http://test")
    await provider.acompletion(messages, model="accounts/fireworks/models/gpt-oss-20b")
    return captured["body"]


class TestAnthropicHonorsHint:
    """Anthropic needs explicit breakpoints to cache at all, so the hint must survive."""

    @pytest.mark.asyncio
    async def test_hint_moves_onto_last_content_block(self, monkeypatch):
        body = await _anthropic_body(
            monkeypatch,
            [
                {"role": "user", "content": [{"type": "text", "text": "ctx"}], "cache_control": EPHEMERAL},
                {"role": "user", "content": "hi"},
            ],
        )

        ctx, task = body["messages"]
        assert ctx["content"][-1] == {"type": "text", "text": "ctx", "cache_control": EPHEMERAL}
        assert "cache_control" not in ctx, "the hint belongs on the block, not the message"
        assert task["content"] == "hi", "unmarked messages stay untouched"

    @pytest.mark.asyncio
    async def test_hint_marks_only_the_last_block_of_a_multi_block_message(self, monkeypatch):
        body = await _anthropic_body(
            monkeypatch,
            [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}],
                    "cache_control": EPHEMERAL,
                }
            ],
        )

        blocks = body["messages"][0]["content"]
        assert "cache_control" not in blocks[0]
        assert blocks[1]["cache_control"] == EPHEMERAL

    @pytest.mark.asyncio
    async def test_string_content_is_promoted_to_a_markable_block(self, monkeypatch):
        body = await _anthropic_body(monkeypatch, [{"role": "user", "content": "ctx", "cache_control": EPHEMERAL}])

        assert body["messages"][0]["content"] == [{"type": "text", "text": "ctx", "cache_control": EPHEMERAL}]

    @pytest.mark.asyncio
    async def test_breakpoints_are_capped_at_the_api_limit(self, monkeypatch):
        """More marked tiers than the API allows must not 400; the latest ones win
        because each caches the longest prefix."""
        from tsugite.providers.anthropic import MAX_CACHE_BREAKPOINTS

        marked = [{"role": "user", "content": f"t{i}", "cache_control": EPHEMERAL} for i in range(6)]
        body = await _anthropic_body(monkeypatch, marked)

        cached = [i for i, m in enumerate(body["messages"]) if "cache_control" in m["content"][-1]]
        assert len(cached) == MAX_CACHE_BREAKPOINTS
        assert cached == [2, 3, 4, 5]

    @pytest.mark.asyncio
    async def test_caller_messages_are_not_mutated(self, monkeypatch):
        messages = [{"role": "user", "content": [{"type": "text", "text": "ctx"}], "cache_control": EPHEMERAL}]

        await _anthropic_body(monkeypatch, messages)

        assert messages[0] == {
            "role": "user",
            "content": [{"type": "text", "text": "ctx"}],
            "cache_control": EPHEMERAL,
        }


class TestOpenAICompatDropsHint:
    """OpenAI-schema endpoints have no equivalent field; strict ones 400 on it."""

    @pytest.mark.asyncio
    async def test_hint_is_dropped_and_content_preserved(self, monkeypatch):
        body = await _openai_compat_body(
            monkeypatch,
            [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "ctx", "cache_control": EPHEMERAL},
                {"role": "assistant", "content": "ack", "cache_control": EPHEMERAL},
                {"role": "user", "content": "hi"},
            ],
        )

        sent = body["messages"]
        assert all("cache_control" not in m for m in sent)
        assert [m["content"] for m in sent] == ["sys", "ctx", "ack", "hi"]

    @pytest.mark.asyncio
    async def test_content_blocks_pass_through_unchanged(self, monkeypatch):
        blocks = [{"type": "text", "text": "hi"}, {"type": "image_url", "image_url": {"url": "http://x/y.png"}}]

        body = await _openai_compat_body(monkeypatch, [{"role": "user", "content": blocks, "cache_control": EPHEMERAL}])

        assert body["messages"][0] == {"role": "user", "content": blocks}

    @pytest.mark.asyncio
    async def test_caller_messages_are_not_mutated(self, monkeypatch):
        messages = [{"role": "user", "content": "ctx", "cache_control": EPHEMERAL}]

        await _openai_compat_body(monkeypatch, messages)

        assert messages[0]["cache_control"] == EPHEMERAL


@pytest.mark.asyncio
async def test_agent_marks_context_turns_for_the_providers_to_translate(monkeypatch):
    """The hint the providers translate is the one core/agent.py emits - keep the
    two ends of the contract pinned to each other."""
    from tsugite.core.agent import CONTEXT_ACK

    messages = [
        {"role": "user", "content": [{"type": "text", "text": "<context>...</context>"}], "cache_control": EPHEMERAL},
        {"role": "assistant", "content": CONTEXT_ACK},
        {"role": "user", "content": "task"},
    ]

    anthropic_body = await _anthropic_body(monkeypatch, messages)
    openai_body = await _openai_compat_body(monkeypatch, messages)

    assert anthropic_body["messages"][0]["content"][-1]["cache_control"] == EPHEMERAL
    assert "cache_control" not in openai_body["messages"][0]
