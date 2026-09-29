"""A run that exists to do work must not end as success when it did none (#925).

Interactive turns answer in prose legitimately, so only runs built with
`expects_action` (scheduled runs) get the `no_action` end.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from tsugite.core.agent import TsugiteAgent
from tsugite.history import get_history_backend
from tsugite.providers.base import CompletionResponse, Usage


def _agent(replies: list[str], **kwargs):
    storage = get_history_backend().create(agent_name="t", model="openai:gpt-4o-mini")
    agent = TsugiteAgent(
        model_string="openai:gpt-4o-mini", tools=[], instructions="", max_turns=5, storage=storage, **kwargs
    )
    agent._provider = MagicMock()
    agent._provider.acompletion = AsyncMock(
        side_effect=[CompletionResponse(content=r, usage=Usage(total_tokens=10), cost=0.0) for r in replies]
    )
    agent._provider.stop = AsyncMock()
    agent._provider.get_state = MagicMock(return_value=None)
    agent._provider.set_context = MagicMock()
    return agent, storage


def _end(storage) -> dict:
    (end,) = [e for e in storage.iter_events() if e.type == "session_end"]
    return end.data


PROSE = "I'm sorry, but I couldn't complete the check in this run."


@pytest.mark.asyncio
async def test_scheduled_run_that_executes_nothing_ends_no_action():
    agent, storage = _agent([PROSE], expects_action=True)

    result = await agent.run("weekly review", return_full_result=True)

    end = _end(storage)
    assert end["status"] == "no_action"
    assert "without executing any code" in end["error_message"]
    assert result.error == end["error_message"]


@pytest.mark.asyncio
async def test_scheduled_run_with_return_value_is_success():
    agent, storage = _agent(['```python-exec\nreturn_value("done")\n```'], expects_action=True)

    await agent.run("weekly review")

    assert _end(storage)["status"] == "success"


@pytest.mark.asyncio
async def test_scheduled_run_that_executes_code_is_success():
    agent, storage = _agent(
        ['```python-exec\nprint("checked")\n```', "All checked, nothing to report."], expects_action=True
    )

    await agent.run("weekly review")

    assert _end(storage)["status"] == "success"


@pytest.mark.asyncio
async def test_interactive_prose_answer_is_success():
    agent, storage = _agent([PROSE])

    await agent.run("what is 2+2?")

    assert _end(storage)["status"] == "success"
