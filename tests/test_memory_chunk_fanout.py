"""Concurrency and failure handling in summarize_session's chunk fan-out."""

import asyncio

import pytest
from tsugite_daemon import memory

MODEL = "openai:gpt-4o-mini"

CHUNK_COUNT = 32


@pytest.fixture
def one_message_per_chunk(monkeypatch):
    """Force _chunk_messages to emit one chunk per message."""
    monkeypatch.setattr(memory, "get_context_limit", lambda model, fallback=None: 100)
    monkeypatch.setattr(memory, "_count_tokens", lambda text, model: 40)


def _history(count: int = CHUNK_COUNT) -> list[dict]:
    return [{"role": "user", "content": f"message {i}"} for i in range(count)]


@pytest.mark.asyncio
async def test_chunk_summaries_respect_a_concurrency_bound(one_message_per_chunk, monkeypatch):
    active = 0
    peak = 0

    async def fake_complete(system_prompt, user_content, model):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0)
        active -= 1
        return "chunk summary"

    monkeypatch.setattr(memory, "_llm_complete", fake_complete)

    await memory.summarize_session(_history(), model=MODEL)

    assert peak <= memory.MAX_CONCURRENT_CHUNK_SUMMARIES


@pytest.mark.asyncio
async def test_failing_chunk_does_not_strand_its_siblings(one_message_per_chunk, monkeypatch):
    in_flight = 0
    combined = False

    async def fake_complete(system_prompt, user_content, model):
        nonlocal in_flight, combined
        if system_prompt == memory.COMBINE_SYSTEM_PROMPT:
            combined = True
            return "combined"
        if "message 0" in user_content:
            raise RuntimeError("chunk blew up")
        in_flight += 1
        await asyncio.sleep(0.01)
        in_flight -= 1
        return "chunk summary"

    monkeypatch.setattr(memory, "_llm_complete", fake_complete)

    with pytest.raises(RuntimeError, match="chunk blew up"):
        await memory.summarize_session(_history(), model=MODEL)

    assert in_flight == 0
    assert not combined


@pytest.mark.asyncio
async def test_ladder_fallback_waits_for_the_failed_attempt(one_message_per_chunk, monkeypatch):
    in_flight = 0
    in_flight_at_fallback = None

    async def fake_complete(system_prompt, user_content, model):
        nonlocal in_flight, in_flight_at_fallback
        if model == "fallback-model":
            if in_flight_at_fallback is None:
                in_flight_at_fallback = in_flight
            return "fallback summary"
        if "message 0" in user_content:
            raise RuntimeError("first model failed")
        in_flight += 1
        await asyncio.sleep(0.01)
        in_flight -= 1
        return "chunk summary"

    monkeypatch.setattr(memory, "_llm_complete", fake_complete)

    summary, used = await memory.walk_model_ladder(
        ["first-model", "fallback-model"],
        lambda m: memory.summarize_session(_history(), model=m),
    )

    assert used == "fallback-model"
    assert in_flight_at_fallback == 0


@pytest.mark.asyncio
async def test_progress_payloads_count_up_to_the_true_chunk_total(one_message_per_chunk, monkeypatch):
    async def fake_complete(system_prompt, user_content, model):
        await asyncio.sleep(0)
        return "chunk summary"

    monkeypatch.setattr(memory, "_llm_complete", fake_complete)

    payloads = []
    await memory.summarize_session(_history(), model=MODEL, progress_callback=payloads.append)

    assert payloads[0] == {"phase": "chunking"}
    assert payloads[-1] == {"phase": "combining"}
    summarizing = [p for p in payloads if p["phase"] == "summarizing"]
    assert [p["chunk_index"] for p in summarizing] == list(range(1, CHUNK_COUNT + 1))
    assert {p["chunk_total"] for p in summarizing} == {CHUNK_COUNT}
