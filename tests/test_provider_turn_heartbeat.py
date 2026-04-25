"""Heartbeat events while waiting on the LLM provider during _provider_turn."""

import asyncio
import contextlib

import pytest

from tsugite.core.agent import TsugiteAgent
from tsugite.events import EventBus, LLMWaitProgressEvent
from tsugite.providers.base import CompletionResponse, StreamChunk, Usage


@pytest.fixture
def fast_heartbeat(monkeypatch):
    """Compress the heartbeat interval so tests don't sit on real-time wall clock."""
    monkeypatch.setattr("tsugite.core.agent._LLM_WAIT_HEARTBEAT_INTERVAL", 0.05)


def _make_agent(bus: EventBus) -> TsugiteAgent:
    return TsugiteAgent(model_string="openai:gpt-4o-mini", tools=[], event_bus=bus)


@pytest.mark.asyncio
async def test_provider_turn_emits_heartbeats_during_slow_completion(fast_heartbeat):
    """Non-streaming acompletion that takes a while should produce periodic heartbeats."""
    bus = EventBus()
    captured: list = []
    bus.subscribe(captured.append)

    agent = _make_agent(bus)

    async def slow_acompletion(messages, model, stream, **kwargs):
        await asyncio.sleep(0.18)
        return CompletionResponse(content="Final answer: hello", usage=Usage())

    agent._provider.acompletion = slow_acompletion

    await agent._provider_turn(messages=[{"role": "user", "content": "hi"}], turn_num=0, stream=False)

    progress_events = [e for e in captured if isinstance(e, LLMWaitProgressEvent)]
    assert len(progress_events) >= 2, f"expected ≥2 heartbeats during 0.18s wait, got {len(progress_events)}"
    assert all(e.elapsed_seconds >= 0 for e in progress_events)


@pytest.mark.asyncio
async def test_heartbeat_stops_after_completion(fast_heartbeat):
    """No heartbeat events fire after the LLM call returns."""
    bus = EventBus()
    captured: list = []
    bus.subscribe(captured.append)

    agent = _make_agent(bus)

    async def quick_acompletion(messages, model, stream, **kwargs):
        await asyncio.sleep(0.12)
        return CompletionResponse(content="Final answer: ok", usage=Usage())

    agent._provider.acompletion = quick_acompletion

    await agent._provider_turn(messages=[{"role": "user", "content": "hi"}], turn_num=0, stream=False)
    count_at_finish = sum(1 for e in captured if isinstance(e, LLMWaitProgressEvent))

    # Drain a few more intervals; no further heartbeats should land.
    await asyncio.sleep(0.15)
    final_count = sum(1 for e in captured if isinstance(e, LLMWaitProgressEvent))
    assert final_count == count_at_finish, "heartbeat should be cancelled once acompletion returns"


@pytest.mark.asyncio
async def test_heartbeat_runs_during_streaming_branch(fast_heartbeat):
    """Streaming providers also need heartbeats while waiting between chunks."""
    bus = EventBus()
    captured: list = []
    bus.subscribe(captured.append)

    agent = _make_agent(bus)

    async def slow_stream(messages, model, stream, **kwargs):
        async def gen():
            await asyncio.sleep(0.18)
            yield StreamChunk(content="Final answer: ok", done=True, usage=Usage())

        return gen()

    agent._provider.acompletion = slow_stream

    await agent._provider_turn(messages=[{"role": "user", "content": "hi"}], turn_num=0, stream=True)

    progress_events = [e for e in captured if isinstance(e, LLMWaitProgressEvent)]
    assert len(progress_events) >= 2, "streaming branch should still emit heartbeats while waiting on first chunk"


@pytest.mark.asyncio
async def test_heartbeat_cancelled_on_provider_exception(fast_heartbeat):
    """If acompletion raises, the heartbeat task must be cleanly cancelled."""
    bus = EventBus()
    captured: list = []
    bus.subscribe(captured.append)

    agent = _make_agent(bus)

    async def exploding_acompletion(messages, model, stream, **kwargs):
        await asyncio.sleep(0.08)
        raise RuntimeError("provider blew up")

    agent._provider.acompletion = exploding_acompletion

    with pytest.raises(RuntimeError, match="provider blew up"):
        await agent._provider_turn(messages=[{"role": "user", "content": "hi"}], turn_num=0, stream=False)

    # If the heartbeat leaked, we'd see new events arriving after the raise.
    count_at_raise = sum(1 for e in captured if isinstance(e, LLMWaitProgressEvent))
    await asyncio.sleep(0.15)
    final_count = sum(1 for e in captured if isinstance(e, LLMWaitProgressEvent))
    assert final_count == count_at_raise, "heartbeat must not survive past acompletion's exception"


@pytest.mark.asyncio
async def test_no_heartbeat_without_event_bus(fast_heartbeat):
    """Agents without an event_bus must not start a heartbeat task at all."""
    agent = TsugiteAgent(model_string="openai:gpt-4o-mini", tools=[], event_bus=None)

    async def slow_acompletion(messages, model, stream, **kwargs):
        await asyncio.sleep(0.12)
        return CompletionResponse(content="Final answer: ok", usage=Usage())

    agent._provider.acompletion = slow_acompletion

    # The call should simply succeed; nothing to assert on the bus.
    await agent._provider_turn(messages=[{"role": "user", "content": "hi"}], turn_num=0, stream=False)


@pytest.mark.asyncio
async def test_elapsed_seconds_increases_monotonically(fast_heartbeat):
    bus = EventBus()
    captured: list = []
    bus.subscribe(captured.append)

    agent = _make_agent(bus)

    async def slow_acompletion(messages, model, stream, **kwargs):
        await asyncio.sleep(0.25)
        return CompletionResponse(content="Final answer: ok", usage=Usage())

    agent._provider.acompletion = slow_acompletion

    await agent._provider_turn(messages=[{"role": "user", "content": "hi"}], turn_num=0, stream=False)

    elapsed_values = [e.elapsed_seconds for e in captured if isinstance(e, LLMWaitProgressEvent)]
    assert elapsed_values, "expected at least one heartbeat"
    assert elapsed_values == sorted(elapsed_values), f"elapsed_seconds should be monotonic: {elapsed_values}"


# Suppress unused-import warning; contextlib is reserved for the suppression of
# CancelledError in production code mirrored by these tests.
_ = contextlib
