"""A run that exists to do work must not end as success when it did none (#925).

Interactive turns answer in prose legitimately, so only runs built with
`expects_action` (scheduled runs) get the `no_action` end.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from tsugite.core.agent import TsugiteAgent
from tsugite.history import get_history_backend
from tsugite.providers.base import CompletionResponse, Usage


def _agent(replies: list[str], max_turns: int = 5, **kwargs):
    storage = get_history_backend().create(agent_name="t", model="openai:gpt-4o-mini")
    agent = TsugiteAgent(
        model_string="openai:gpt-4o-mini", tools=[], instructions="", max_turns=max_turns, storage=storage, **kwargs
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
    agent, storage = _agent([PROSE, PROSE], expects_action=True)

    result = await agent.run("weekly review", return_full_result=True)

    end = _end(storage)
    assert end["status"] == "no_action"
    assert "without executing any code" in end["error_message"]
    assert result.error == end["error_message"]


@pytest.mark.asyncio
async def test_scheduled_run_returning_notify_false_ends_silent():
    agent, storage = _agent(['```python-exec\nreturn_value({"notify": False})\n```'], expects_action=True)

    await agent.run("check for changes")

    end = _end(storage)
    assert end["status"] == "silent"
    assert end.get("error_message") is None


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


UNFENCED = 'import json\nrun_out = run(command="git status")\nprint(json.dumps(run_out))'
FENCED = '```python-exec\nreturn_value("done")\n```'


@pytest.mark.asyncio
async def test_unfenced_python_is_re_asked_then_executes():
    agent, storage = _agent([UNFENCED, FENCED], expects_action=True)

    await agent.run("weekly review")

    assert agent._provider.acompletion.await_count == 2
    notice = agent._provider.acompletion.await_args_list[1].kwargs["messages"][-1]["content"]
    assert "python-exec" in notice
    assert _end(storage)["status"] == "success"


@pytest.mark.asyncio
async def test_two_unfenced_replies_end_no_action_after_one_re_ask():
    agent, storage = _agent([UNFENCED, UNFENCED], expects_action=True)

    await agent.run("weekly review")

    assert agent._provider.acompletion.await_count == 2
    assert _end(storage)["status"] == "no_action"


@pytest.mark.asyncio
async def test_last_turn_prose_is_not_re_asked():
    agent, storage = _agent([PROSE], max_turns=1, expects_action=True)

    await agent.run("weekly review")

    assert agent._provider.acompletion.await_count == 1
    assert _end(storage)["status"] == "no_action"


@pytest.mark.asyncio
async def test_prose_after_executed_code_is_not_re_asked():
    agent, storage = _agent(['```python-exec\nprint("checked")\n```', UNFENCED], expects_action=True)

    await agent.run("weekly review")

    assert agent._provider.acompletion.await_count == 2
    assert _end(storage)["status"] == "success"


@pytest.mark.asyncio
async def test_interactive_prose_is_not_re_asked():
    agent, _ = _agent([UNFENCED])

    await agent.run("what is 2+2?")

    assert agent._provider.acompletion.await_count == 1
