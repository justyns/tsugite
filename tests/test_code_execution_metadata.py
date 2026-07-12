"""code_execution events persist executed-block metadata for deterministic replay.

Instead of the web UI reconstructing execution semantics from raw_content with
regex (which breaks on nested fences inside a return_value string), the
runtime persists what it already knows when it executes the block: whether the
last statement was an expression, and what return_value() produced. The return
value is stored as a masked repr string, never json.dumps'd, so an arbitrary
(non-serializable) object never breaks recording.
"""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from tsugite.core.agent import TsugiteAgent
from tsugite.history import SessionStorage
from tsugite.providers.base import CompletionResponse, Usage


def _resp(content: str) -> CompletionResponse:
    return CompletionResponse(content=content, usage=Usage(total_tokens=10), cost=0.0)


def _patch(agent, *, side_effect=None, return_value=None):
    mock = AsyncMock(side_effect=side_effect, return_value=return_value)
    agent._provider = MagicMock()
    agent._provider.acompletion = mock
    agent._provider.stop = AsyncMock()
    agent._provider.get_state = MagicMock(return_value=None)
    agent._provider.set_context = MagicMock()
    return mock


def _agent(storage):
    return TsugiteAgent(model_string="openai:gpt-4o-mini", tools=[], instructions="", max_turns=5, storage=storage)


def _code_exec_event(storage):
    events = [e for e in storage.iter_events() if e.type == "code_execution"]
    assert len(events) == 1, f"expected one code_execution, got {len(events)}"
    return events[0].data


@pytest.mark.asyncio
async def test_return_value_dict_records_expr_and_type(tmp_path: Path):
    storage = SessionStorage.create(agent_name="t", model="openai:gpt-4o-mini", session_path=tmp_path / "s.jsonl")
    agent = _agent(storage)
    _patch(agent, return_value=_resp('```python-exec\nreturn_value({"a": 1, "b": [1, 2, 3]})\n```'))

    await agent.run("structured")

    data = _code_exec_event(storage)
    assert data["last_statement_type"] == "expr"
    assert data["return_value_type"] == "dict"
    assert "a" in data["return_value_repr"]


@pytest.mark.asyncio
async def test_trailing_statement_has_no_return_value(tmp_path: Path):
    storage = SessionStorage.create(agent_name="t", model="openai:gpt-4o-mini", session_path=tmp_path / "s.jsonl")
    agent = _agent(storage)
    calls = 0

    async def side(*a, **k):
        nonlocal calls
        calls += 1
        return _resp("```python-exec\nx = 1\n```") if calls == 1 else _resp("done")

    _patch(agent, side_effect=side)

    await agent.run("assign")

    data = _code_exec_event(storage)
    assert data["last_statement_type"] == "statement"
    assert "return_value_repr" not in data
    assert "return_value_type" not in data


@pytest.mark.asyncio
async def test_non_serializable_return_value_stored_as_repr(tmp_path: Path):
    storage = SessionStorage.create(agent_name="t", model="openai:gpt-4o-mini", session_path=tmp_path / "s.jsonl")
    agent = _agent(storage)
    # A set is not JSON-serializable; recording must not attempt json.dumps on it.
    _patch(agent, return_value=_resp("```python-exec\nreturn_value({1, 2, 3})\n```"))

    await agent.run("set")

    data = _code_exec_event(storage)
    assert data["return_value_type"] == "set"
    assert isinstance(data["return_value_repr"], str)
