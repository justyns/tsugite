"""A streamed response that degenerates into a whitespace run or a repeated chunk is aborted."""

import pytest

from tsugite.core.agent import TsugiteAgent, _find_runaway
from tsugite.events import EventBus, StreamChunkEvent, WarningEvent
from tsugite.providers.base import StreamChunk, Usage

FINAL = 'Thought: ok.\n\n```python-exec\nfinal_answer("done")\n```'


def _agent(bus=None, storage=None) -> TsugiteAgent:
    return TsugiteAgent(model_string="openai:gpt-4o-mini", tools=[], max_turns=3, event_bus=bus, storage=storage)


def _stream_turns(turns, consumed):
    """Provider whose Nth call streams turns[N]; `consumed` counts chunks actually pulled."""
    calls = iter(turns)

    async def acompletion(messages, model, stream, **kwargs):
        chunks = next(calls)

        async def gen():
            for chunk in chunks:
                consumed.append(chunk)
                yield chunk

        return gen()

    return acompletion


def _pieces(text, size=500):
    return [StreamChunk(content=text[i : i + size]) for i in range(0, len(text), size)]


def _done():
    return StreamChunk(done=True, usage=Usage(total_tokens=5), cost=0.0)


def test_find_runaway_ignores_long_legitimate_text():
    report = "".join(f"Section {i}: results were within tolerance.\n\n" for i in range(2000))
    assert len(report) > 80_000
    assert _find_runaway(report) is None
    assert _find_runaway("-" * 40 + "\n" + "| a | b |\n" * 20) is None


@pytest.mark.parametrize("text", ["\n" * 6000, " \n\t" * 2000, "ab " * 3000, "\n\nok " * 1500])
def test_find_runaway_flags_degenerate_text(text):
    assert _find_runaway("Thought: x\n" + text) is not None


def test_find_runaway_flags_a_long_repeated_tool_call_wrapper():
    wrapper = '<invoke name="python-exec">\n<parameter name="code">print(read_file("notes/projects/q4-roadmap.md"))</parameter>\n<parameter name="timeout">30000</parameter>\n</invoke>\n'
    assert 150 < len(wrapper) < 200
    assert _find_runaway("Thought: x\n" + wrapper * 40) is not None


@pytest.mark.asyncio
@pytest.mark.parametrize("runaway", ["\n" * 40_000, "I will check. " * 4000])
async def test_runaway_stream_is_aborted_and_reported_to_the_model(runaway):
    bus = EventBus()
    events: list = []
    bus.subscribe(events.append)
    agent = _agent(bus)
    consumed: list = []
    first = _pieces("Thought: starting.\n" + runaway)
    agent._provider.acompletion = _stream_turns([first, [StreamChunk(content=FINAL), _done()]], consumed)

    result = await agent.run("go", stream=True, return_full_result=True)

    assert result.output == "done"
    assert len([c for c in consumed if c in first]) < len(first) / 2, "stream was drained to the end"
    warnings = [e for e in events if isinstance(e, WarningEvent) and e.category == "runaway_output"]
    assert len(warnings) == 1
    step = agent.memory.steps[0]
    assert step.error and "5000" in step.error
    assert "<tsugite_runaway_output_notice>" in step.xml_observation
    assert len(step.raw_content) < 5000
    assert any(isinstance(e, StreamChunkEvent) for e in events)


@pytest.mark.asyncio
async def test_aborted_runaway_turn_records_a_model_response_with_unknown_usage():
    recorded: list = []

    class _Storage:
        def record(self, event_type, **fields):
            recorded.append((event_type, fields))

        def iter_events(self, types=None):
            return iter(())

    agent = _agent(storage=_Storage())
    first = _pieces("Thought: starting.\n" + "I will check. " * 4000)
    agent._provider.acompletion = _stream_turns([first, [StreamChunk(content=FINAL), _done()]], [])

    await agent.run("go", stream=True, return_full_result=True)

    responses = [f for t, f in recorded if t == "model_response"]
    aborted = responses[0]
    assert aborted["turn"] == 0
    assert aborted["raw_content"] == agent.memory.steps[0].raw_content
    assert aborted["usage"] is None and aborted["cost"] is None
    assert len(responses) == 2


@pytest.mark.asyncio
async def test_long_legitimate_stream_is_untouched():
    report = "".join(f"Finding {i}: nothing unusual here.\n\n" for i in range(1500))
    agent = _agent()
    consumed: list = []
    agent._provider.acompletion = _stream_turns([_pieces(report) + [_done()]], consumed)

    result = await agent.run("go", stream=True, return_full_result=True)

    assert result.output == report
    assert result.error is None
