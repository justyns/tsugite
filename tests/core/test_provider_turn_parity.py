"""Streaming and blocking turns differ only in how the response is obtained.

Everything after that - escaping runtime tags, parsing, accounting usage and
cost, recording reasoning, persisting the model response - is one shared tail.
Two parallel implementations had already drifted apart in ways that read as
oversights rather than decisions, so these tests pin the parts that must match
and the two that legitimately differ.
"""

import pytest

from tsugite.core.agent import TsugiteAgent
from tsugite.events import (
    EventBus,
    LLMMessageEvent,
    ReasoningContentEvent,
    ReasoningTokensEvent,
    StreamChunkEvent,
)
from tsugite.providers.base import CompletionResponse, StreamChunk, Usage

RESPONSE_TEXT = "Thinking about it.\n\n```python-exec\nprint('hi')\n```"


def _agent(bus: EventBus) -> TsugiteAgent:
    return TsugiteAgent(model_string="openai:gpt-4o-mini", tools=[], event_bus=bus)


def _stream_of(chunks):
    async def acompletion(messages, model, stream, **kwargs):
        async def gen():
            for chunk in chunks:
                yield chunk

        return gen()

    return acompletion


def _blocking(response):
    async def acompletion(messages, model, stream, **kwargs):
        return response

    return acompletion


@pytest.mark.asyncio
async def test_streaming_reports_reasoning_tokens_like_blocking():
    """Both paths must surface reasoning-token counts.

    Only the blocking path emitted ReasoningTokensEvent, so a streaming run on a
    reasoning model showed no reasoning usage in the UI at all.
    """
    usage = Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15, reasoning_tokens=99)

    bus = EventBus()
    captured: list = []
    bus.subscribe(captured.append)
    agent = _agent(bus)
    agent._provider.acompletion = _stream_of([StreamChunk(content=RESPONSE_TEXT), StreamChunk(done=True, usage=usage)])

    await agent._provider_turn(messages=[{"role": "user", "content": "hi"}], turn_num=0, stream=True)

    events = [e for e in captured if isinstance(e, ReasoningTokensEvent)]
    assert [e.tokens for e in events] == [99], "streaming turn did not report reasoning tokens"


@pytest.mark.asyncio
async def test_streaming_counts_cost_reported_without_usage():
    """A provider may report cost without a usage block (subscription models).

    Blocking handled that; streaming only looked at cost when `usage` was also
    present, silently dropping the charge.
    """
    bus = EventBus()
    agent = _agent(bus)
    agent._provider.acompletion = _stream_of([StreamChunk(content=RESPONSE_TEXT), StreamChunk(done=True, cost=0.25)])

    await agent._provider_turn(messages=[{"role": "user", "content": "hi"}], turn_num=0, stream=True)

    assert agent.reported_cost == 0.25, f"streaming turn lost the reported cost: {agent.reported_cost}"


@pytest.mark.asyncio
async def test_both_paths_produce_the_same_turn_result():
    """Same model output in, same parse and accounting out."""
    usage = Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15)

    stream_agent = _agent(EventBus())
    stream_agent._provider.acompletion = _stream_of(
        [StreamChunk(content=RESPONSE_TEXT), StreamChunk(done=True, usage=usage, cost=0.5)]
    )
    streamed = await stream_agent._provider_turn(messages=[{"role": "user", "content": "hi"}], turn_num=0, stream=True)

    block_agent = _agent(EventBus())
    block_agent._provider.acompletion = _blocking(CompletionResponse(content=RESPONSE_TEXT, usage=usage, cost=0.5))
    blocking = await block_agent._provider_turn(messages=[{"role": "user", "content": "hi"}], turn_num=0, stream=False)

    assert streamed.thought == blocking.thought
    assert streamed.code == blocking.code
    assert streamed.num_code_blocks == blocking.num_code_blocks
    assert streamed.content_blocks == blocking.content_blocks
    assert streamed.spoofed_runtime_tag == blocking.spoofed_runtime_tag
    assert streamed.has_bare_python == blocking.has_bare_python
    assert streamed.step_cost == blocking.step_cost
    assert stream_agent.total_tokens == block_agent.total_tokens
    assert stream_agent.reported_cost == block_agent.reported_cost

    # Absolute, not just equal to each other: a regression that zeroed step_cost
    # in the now-shared tail would zero it for both paths and still pass above.
    assert blocking.step_cost == 0.5
    assert block_agent.total_cost == 0.5
    assert block_agent.total_tokens == 15


@pytest.mark.asyncio
async def test_streaming_does_not_repeat_the_thought_as_a_message():
    """The one difference that is intentional.

    Streaming already delivered the prose chunk by chunk, so re-emitting it as a
    single LLMMessageEvent would render the same text twice.
    """
    bus = EventBus()
    captured: list = []
    bus.subscribe(captured.append)
    agent = _agent(bus)
    agent._provider.acompletion = _stream_of(
        [StreamChunk(content=RESPONSE_TEXT), StreamChunk(done=True, usage=Usage())]
    )

    await agent._provider_turn(messages=[{"role": "user", "content": "hi"}], turn_num=0, stream=True)

    assert any(isinstance(e, StreamChunkEvent) for e in captured), "chunks were never streamed"
    assert not [e for e in captured if isinstance(e, LLMMessageEvent)], "streamed thought was emitted twice"


@pytest.mark.asyncio
async def test_blocking_emits_the_thought_once():
    bus = EventBus()
    captured: list = []
    bus.subscribe(captured.append)
    agent = _agent(bus)
    agent._provider.acompletion = _blocking(CompletionResponse(content=RESPONSE_TEXT, usage=Usage()))

    await agent._provider_turn(messages=[{"role": "user", "content": "hi"}], turn_num=0, stream=False)

    messages = [e for e in captured if isinstance(e, LLMMessageEvent)]
    assert len(messages) == 1
    assert "Thinking about it." in messages[0].content


@pytest.mark.asyncio
async def test_streaming_does_not_replay_reasoning_as_an_aggregate():
    """The second emission the flag gates, previously untested.

    Reasoning is delivered chunk by chunk while streaming; emitting the
    accumulated text again would render it twice. Double-render is a recurring
    bug class in this UI, which is the whole reason the flag exists.
    """
    bus = EventBus()
    captured: list = []
    bus.subscribe(captured.append)
    agent = _agent(bus)
    agent._provider.acompletion = _stream_of(
        [
            StreamChunk(reasoning_content="think "),
            StreamChunk(reasoning_content="harder"),
            StreamChunk(content=RESPONSE_TEXT),
            StreamChunk(done=True, usage=Usage()),
        ]
    )

    await agent._provider_turn(messages=[{"role": "user", "content": "hi"}], turn_num=0, stream=True)

    reasoning = [e.content for e in captured if isinstance(e, ReasoningContentEvent)]
    assert reasoning == ["think ", "harder"], f"reasoning was replayed as an aggregate: {reasoning}"


@pytest.mark.asyncio
async def test_cost_is_counted_for_every_usage_and_cost_combination():
    """_accumulate_usage owns cost accounting for all four shapes.

    Collapsing the caller's usage/no-usage branch into it is only safe if the
    no-usage path still counts cost and still reports it.
    """
    for usage, cost, expected_cost, expected_tokens in [
        (Usage(total_tokens=7), 0.25, 0.25, 7),
        (Usage(total_tokens=7), None, 0.0, 7),
        (None, 0.25, 0.25, 0),
        (None, None, 0.0, 0),
    ]:
        agent = _agent(EventBus())
        step_cost = agent._accumulate_usage(usage, cost)

        assert step_cost == expected_cost
        assert agent.total_cost == expected_cost
        assert agent.total_tokens == expected_tokens
        assert agent.cost_reported is (cost is not None)
