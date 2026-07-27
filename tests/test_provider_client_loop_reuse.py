"""Cached providers must survive being reused across successive event loops.

Providers are cached process-wide (tsugite/providers/__init__.py) while each agent
turn runs under a fresh asyncio.run() loop (tsugite/agent_runner/runner.py). So a
long-lived daemon process calls the same provider instance's _get_client() under a
new loop on every turn after the first, tripping the stale-loop rebuild branch.

Regression: httpx 0.28 AsyncClient exposes aclose(), not a sync close(). The
Anthropic provider's rebuild branch called the nonexistent close(), so every turn
after the first crashed with AttributeError and never self-healed.
"""

from __future__ import annotations

import asyncio

import httpx

from tsugite.providers.anthropic import AnthropicProvider
from tsugite.providers.openai_compat import OpenAICompatProvider


def _client_across_loops(provider, rounds: int) -> list[httpx.AsyncClient]:
    """Call provider._get_client() once inside each of `rounds` separate asyncio.run loops."""
    seen: list[httpx.AsyncClient] = []

    async def _round() -> None:
        client = provider._get_client()
        assert isinstance(client, httpx.AsyncClient)
        assert not client.is_closed
        seen.append(client)

    for _ in range(rounds):
        asyncio.run(_round())

    return seen


def test_anthropic_client_survives_new_event_loops():
    provider = AnthropicProvider(api_key="test")

    clients = _client_across_loops(provider, rounds=3)

    # A fresh loop each round means the stale-loop branch rebuilds every time.
    assert len(clients) == 3
    assert len({id(c) for c in clients}) == 3
    assert not clients[-1].is_closed


def test_openai_compat_client_survives_new_event_loops():
    provider = OpenAICompatProvider(name="openai", api_base="https://example.test/v1", api_key="test")

    clients = _client_across_loops(provider, rounds=3)

    assert len(clients) == 3
    assert not clients[-1].is_closed


def test_get_client_reuses_within_same_loop():
    provider = AnthropicProvider(api_key="test")

    async def _same_loop() -> None:
        first = provider._get_client()
        second = provider._get_client()
        # Same running loop: the cached client must be returned untouched.
        assert first is second

    asyncio.run(_same_loop())
