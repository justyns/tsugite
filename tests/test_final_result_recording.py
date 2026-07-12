"""The final answer must reach the durable history log on every run path.

Scheduled/subprocess sessions record history via the agent's live event
recording (no daemon SSE persist path), so the agent itself must write the
`final_result` event the conversation view renders. Interactive daemon turns
already get one persisted by the SSE handler; the agent must not duplicate it.
"""

from pathlib import Path

import pytest

from tsugite.agent_runner.history_integration import record_final_result
from tsugite.core.agent import TsugiteAgent
from tsugite.history import SessionStorage
from tsugite.providers.base import CompletionResponse, Usage


def _resp(content: str) -> CompletionResponse:
    return CompletionResponse(content=content, usage=Usage(total_tokens=10), cost=0.0)


def _patch(agent, *, side_effect=None, return_value=None):
    from unittest.mock import AsyncMock, MagicMock

    mock = AsyncMock(side_effect=side_effect, return_value=return_value)
    agent._provider = MagicMock()
    agent._provider.acompletion = mock
    agent._provider.stop = AsyncMock()
    agent._provider.get_state = MagicMock(return_value=None)
    agent._provider.set_context = MagicMock()
    return mock


def _storage(tmp_path: Path) -> SessionStorage:
    return SessionStorage.create(agent_name="t", model="openai:gpt-4o-mini", session_path=tmp_path / "s.jsonl")


def _finals(storage):
    return [e for e in storage.iter_events() if e.type == "final_result"]


@pytest.mark.asyncio
async def test_return_value_records_final_result(tmp_path: Path):
    storage = _storage(tmp_path)
    agent = TsugiteAgent(model_string="openai:gpt-4o-mini", tools=[], instructions="", max_turns=5, storage=storage)
    _patch(agent, return_value=_resp('```python-exec\nreturn_value("nothing to commit")\n```'))

    await agent.run("check repo")

    finals = _finals(storage)
    assert len(finals) == 1
    assert finals[0].data["result"] == "nothing to commit"
    # String answers carry no structured payload — same shape as the
    # SSE-persisted event (the client renders from `result`).
    assert finals[0].data.get("result_data") is None
    types = [e.type for e in storage.iter_events()]
    assert types.index("final_result") < types.index("session_end")


@pytest.mark.asyncio
async def test_structured_return_value_records_result_data(tmp_path: Path):
    storage = _storage(tmp_path)
    agent = TsugiteAgent(model_string="openai:gpt-4o-mini", tools=[], instructions="", max_turns=5, storage=storage)
    _patch(agent, return_value=_resp('```python-exec\nreturn_value({"a": 1})\n```'))

    await agent.run("structured")

    finals = _finals(storage)
    assert len(finals) == 1
    assert finals[0].data["result_data"] == {"a": 1}


@pytest.mark.asyncio
async def test_max_turns_records_final_result_with_last_text(tmp_path: Path):
    storage = _storage(tmp_path)
    agent = TsugiteAgent(model_string="openai:gpt-4o-mini", tools=[], instructions="", max_turns=1, storage=storage)
    _patch(agent, return_value=_resp("no code here, just prose"))

    await agent.run("task")

    finals = _finals(storage)
    assert len(finals) == 1
    assert finals[0].data["result"] == "no code here, just prose"


def test_record_final_result_skips_when_turn_already_has_one(tmp_path: Path):
    """An SSE-persisted final_result for the current turn must not be duplicated."""
    storage = _storage(tmp_path)
    storage.record("user_input", text="hi")
    storage.record("final_result", result="answer", result_data="answer")

    record_final_result(storage, result="answer", result_data="answer")

    assert len(_finals(storage)) == 1


def test_record_final_result_records_for_new_turn(tmp_path: Path):
    """A final_result from a previous turn doesn't suppress the current turn's."""
    storage = _storage(tmp_path)
    storage.record("user_input", text="turn one")
    storage.record("final_result", result="one")
    storage.record("user_input", text="turn two")

    record_final_result(storage, result="two", result_data="two")

    assert [e.data["result"] for e in _finals(storage)] == ["one", "two"]
