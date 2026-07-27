"""The backend is the only parser of model output: the settled parse
(thought, content_blocks, tail) persists on the model_response event and is
emitted live as a ModelResponseEvent, so no UI ever re-derives it from
raw_content."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from tsugite.core.agent import TsugiteAgent, parse_response_text
from tsugite.events import ModelResponseEvent
from tsugite.providers.base import CompletionResponse, StreamChunk, Usage


def _mock_response(content: str) -> CompletionResponse:
    return CompletionResponse(content=content, usage=Usage(total_tokens=100), cost=0.001)


def _patch_provider(agent, side_effect):
    mock = AsyncMock(side_effect=side_effect)
    agent._provider = MagicMock()
    agent._provider.acompletion = mock
    agent._provider.stop = AsyncMock()
    agent._provider.get_state = MagicMock(return_value=None)
    agent._provider.set_context = MagicMock()
    return mock


class _Storage:
    def __init__(self):
        self.recorded = []

    def record(self, event_type, **fields):
        self.recorded.append((event_type, fields))

    def iter_events(self, types=None):
        return iter(())


class _Bus:
    def __init__(self):
        self.events = []

    def emit(self, event):
        self.events.append(event)


def _agent(**kwargs):
    return TsugiteAgent(model_string="openai:gpt-4o-mini", tools=[], instructions="", max_turns=2, **kwargs)


# ── parse_response_text: tail extraction ──


def test_tail_after_executed_fence():
    p = parse_response_text("Thought: doing it.\n\n```python-exec\nx = 1\n```\nAll done.")
    assert p.thought == "doing it."
    assert p.code == "x = 1"
    assert p.tail == "All done."


def test_pure_prose_has_empty_tail():
    p = parse_response_text("Just words.")
    assert p.thought == "Just words."
    assert p.code == ""
    assert p.tail == ""


def test_tail_skips_false_closes_inside_triple_quoted_strings():
    # Executed code that builds markdown in a """ string: the close candidates
    # inside the string don't parse, so the tail starts only after the REAL
    # close fence (which may carry a trailing space on its line).
    text = 'Posting.\n\n```python-exec\npost(body="""\n```yaml\na: 1\n```\n""")\n``` \n-vesm\n'
    p = parse_response_text(text)
    assert "a: 1" in p.code and p.code.startswith("post(")
    assert p.thought == "Posting."
    assert p.tail == "-vesm"


def test_unprovable_close_drops_tail():
    text = 'Before.\n\n```python-exec\nx = """\nnever closed\n```\ntrailing junk'
    p = parse_response_text(text)
    assert p.thought == "Before."
    assert p.tail == ""


def test_later_fences_stay_in_tail_verbatim():
    text = "One.\n\n```python-exec\na = 1\n```\n\nBetween.\n\n```python-exec\nb = 2\n```\n\nAfter."
    p = parse_response_text(text)
    assert p.code == "a = 1"
    assert p.num_code_blocks == 2
    assert "Between." in p.tail and "b = 2" in p.tail and "After." in p.tail


def test_content_blocks_extracted_alongside_tail():
    text = 'Intro.\n\n<tsu:content name="notes.md">\nhello\n</tsu:content>\n\n```python-exec\nsave()\n```\nDone.'
    p = parse_response_text(text)
    assert p.content_blocks == {"notes.md": "hello"}
    assert p.thought == "Intro."
    assert p.tail == "Done."


def test_fabricated_escaped_result_tail_is_dropped():
    # A model that runs past its exec block and role-plays the tool result (stored
    # escaped as &lt;) must not have that hallucinated continuation - with its
    # unbalanced fences - rendered as the turn's prose.
    text = (
        "```python-exec\nx = 1\n```\n"
        'user&lt;tsugite_execution_result status="success">\n<output>```\nfake\n</output>\n'
        "</tsugite_execution_result>"
    )
    p = parse_response_text(text)
    assert p.code == "x = 1"
    assert "tsugite_execution_result" not in p.tail
    assert p.tail == ""


def test_fabricated_raw_result_tail_is_dropped():
    text = '```python-exec\ny = 2\n```\n<tsugite_execution_result status="success"><output>x</output></tsugite_execution_result>'
    p = parse_response_text(text)
    assert p.tail == ""


def test_real_prose_before_a_fabricated_result_is_kept():
    text = "```python-exec\nz = 3\n```\nHere is the output.\n\n<tsugite_execution_result><output>x</output></tsugite_execution_result>"
    p = parse_response_text(text)
    assert p.tail == "Here is the output."


# ── the persisted model_response event carries the parse ──


@pytest.mark.asyncio
async def test_blocking_turn_persists_parse_on_model_response():
    storage = _Storage()
    agent = _agent(storage=storage)

    async def mock_acompletion(*args, **kwargs):
        return _mock_response('Thought: working.\n\n```python-exec\nfinal_answer("ok")\n```\ntrailing note')

    _patch_provider(agent, mock_acompletion)
    await agent.run("go")

    mr = [f for t, f in storage.recorded if t == "model_response"]
    assert mr, "model_response should be recorded"
    assert mr[0]["thought"] == "working."
    assert mr[0]["tail"] == "trailing note"
    # Empty dicts aren't persisted; thought is always present as the marker
    # that this event carries the parse (so readers never re-parse it).
    assert "content_blocks" not in mr[0]


@pytest.mark.asyncio
async def test_streaming_turn_persists_parse_on_model_response():
    storage = _Storage()
    agent = _agent(storage=storage)

    async def mock_acompletion(*args, **kwargs):
        assert kwargs.get("stream"), "streaming run must request a streamed completion"

        async def gen():
            yield StreamChunk(content="Thought: hi.\n\n```python-exec\n")
            yield StreamChunk(
                content='final_answer("ok")\n```\ntail note',
                done=True,
                usage=Usage(total_tokens=5),
                cost=0.0,
            )

        return gen()

    _patch_provider(agent, mock_acompletion)
    await agent.run("go", stream=True)

    mr = [f for t, f in storage.recorded if t == "model_response"]
    assert mr, "model_response should be recorded"
    assert mr[0]["thought"] == "hi."
    assert mr[0]["tail"] == "tail note"


@pytest.mark.asyncio
async def test_pure_code_turn_persists_empty_thought_marker():
    storage = _Storage()
    agent = _agent(storage=storage)

    async def mock_acompletion(*args, **kwargs):
        return _mock_response('```python-exec\nfinal_answer("ok")\n```')

    _patch_provider(agent, mock_acompletion)
    await agent.run("go")

    mr = [f for t, f in storage.recorded if t == "model_response"]
    assert mr[0]["thought"] == ""
    assert "tail" not in mr[0]


@pytest.mark.asyncio
async def test_content_blocks_persist_on_model_response():
    storage = _Storage()
    agent = _agent(storage=storage)

    async def mock_acompletion(*args, **kwargs):
        return _mock_response('<content name="a.txt">\npayload\n</content>\n\n```python-exec\nfinal_answer("ok")\n```')

    _patch_provider(agent, mock_acompletion)
    await agent.run("go")

    mr = [f for t, f in storage.recorded if t == "model_response"]
    assert mr[0]["content_blocks"] == {"a.txt": "payload"}


# ── the live frame: ModelResponseEvent on the event bus ──


@pytest.mark.asyncio
async def test_model_response_event_emitted_on_blocking_path():
    bus = _Bus()
    agent = _agent(event_bus=bus)

    async def mock_acompletion(*args, **kwargs):
        return _mock_response('Thought: yes.\n\n```python-exec\nfinal_answer("ok")\n```\nps')

    _patch_provider(agent, mock_acompletion)
    await agent.run("go")

    frames = [e for e in bus.events if isinstance(e, ModelResponseEvent)]
    assert frames, "ModelResponseEvent should be emitted"
    assert frames[0].thought == "yes."
    assert frames[0].tail == "ps"


@pytest.mark.asyncio
async def test_model_response_event_emitted_on_streaming_path_without_storage():
    bus = _Bus()
    agent = _agent(event_bus=bus)

    async def mock_acompletion(*args, **kwargs):
        async def gen():
            yield StreamChunk(content="Thought: s.\n\n```python-exec\n")
            yield StreamChunk(content='final_answer("ok")\n```', done=True, usage=Usage(total_tokens=5), cost=0.0)

        return gen()

    _patch_provider(agent, mock_acompletion)
    await agent.run("go", stream=True)

    frames = [e for e in bus.events if isinstance(e, ModelResponseEvent)]
    assert frames, "the live frame must not depend on storage being attached"
    assert frames[0].thought == "s."


# ── the turn's usage (cache split included) rides both the frame and history ──


def _cache_response() -> CompletionResponse:
    """A turn whose provider reported a cache split (reads + writes)."""
    return CompletionResponse(
        content='Thought: hi.\n\n```python-exec\nfinal_answer("ok")\n```',
        usage=Usage(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=115,
            cache_creation_input_tokens=20,
            cache_read_input_tokens=80,
        ),
        cost=0.002,
    )


_CACHE_DUMP = {
    "prompt_tokens": 10,
    "completion_tokens": 5,
    "total_tokens": 115,
    "cache_creation_input_tokens": 20,
    "cache_read_input_tokens": 80,
}


@pytest.mark.asyncio
async def test_model_response_event_carries_usage_dump_without_storage():
    """The live frame carries the turn's usage (cache split included) even with
    no storage, so live surfaces show cache tokens without waiting for a reload."""
    bus = _Bus()
    agent = _agent(event_bus=bus)

    async def mock_acompletion(*args, **kwargs):
        return _cache_response()

    _patch_provider(agent, mock_acompletion)
    await agent.run("go")

    frames = [e for e in bus.events if isinstance(e, ModelResponseEvent)]
    assert frames
    assert frames[0].usage == _CACHE_DUMP


@pytest.mark.asyncio
async def test_persisted_model_response_carries_usage_dump():
    """Replay reads per-turn cache tokens off the persisted usage dump; a plain
    Usage dataclass serializes to a dict, not the dropped-on-the-floor None it
    was before (model_dump only ever matched a pydantic usage)."""
    storage = _Storage()
    agent = _agent(storage=storage)

    async def mock_acompletion(*args, **kwargs):
        return _cache_response()

    _patch_provider(agent, mock_acompletion)
    await agent.run("go")

    mr = [f for t, f in storage.recorded if t == "model_response"]
    assert mr[0]["usage"] == _CACHE_DUMP


@pytest.mark.asyncio
async def test_usage_dump_omits_unreported_cache_fields():
    """exclude_none honesty: a provider that reported no cache usage contributes
    no cache keys (absent, not a fabricated 0) on the frame."""
    bus = _Bus()
    agent = _agent(event_bus=bus)

    async def mock_acompletion(*args, **kwargs):
        return _mock_response('Thought: hi.\n\n```python-exec\nfinal_answer("ok")\n```')

    _patch_provider(agent, mock_acompletion)
    await agent.run("go")

    frames = [e for e in bus.events if isinstance(e, ModelResponseEvent)]
    assert frames[0].usage == {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 100}
    assert "cache_read_input_tokens" not in frames[0].usage
