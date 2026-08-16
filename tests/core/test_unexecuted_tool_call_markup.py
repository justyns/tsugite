"""A reply carrying tool-call markup instead of a ```python-exec block executed
nothing, so it is not an answer.

Models trained on native tool calling sometimes emit the raw envelope as prose.
This runtime never executes that markup, so accepting the text ends the run with
a fabricated completion summary recorded as a success.
"""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from tsugite.core.agent import TsugiteAgent
from tsugite.history import Session, get_history_backend
from tsugite.providers.base import CompletionResponse, Usage

TOOL_CALL_REPLY = (
    "I'll commit the pending work.\n\n"
    "<function_calls>\n"
    '<invoke name="run_shell">\n'
    '<parameter name="command">git commit -am "log the daily notes"</parameter>\n'
    "</invoke>\n"
    "</function_calls>\n\n"
    "1 commit: docs: log the daily notes"
)


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


@pytest.fixture
def storage(tmp_path: Path) -> Session:
    return get_history_backend().create(agent_name="t", model="openai:gpt-4o-mini")


def _agent(storage, max_turns: int = 4) -> TsugiteAgent:
    return TsugiteAgent(
        model_string="openai:gpt-4o-mini",
        tools=[],
        instructions="",
        max_turns=max_turns,
        storage=storage,
    )


def _end_status(storage: Session) -> str | None:
    ends = [e for e in storage.iter_events() if e.type == "session_end"]
    return ends[-1].data.get("status") if ends else None


@pytest.mark.asyncio
async def test_tool_call_markup_is_corrected_instead_of_answered(storage):
    """The fabricated summary must not become the answer; the model gets the turn
    back with a notice telling it the markup ran nothing."""
    agent = _agent(storage)
    mock = _patch(
        agent,
        side_effect=[_resp(TOOL_CALL_REPLY), _resp("Nothing to commit, the tree is clean.")],
    )

    result = await agent.run("commit anything outstanding")

    assert mock.await_count == 2, "the markup reply must cost a correction turn, not end the run"
    assert result == "Nothing to commit, the tree is clean."
    retry_messages = mock.await_args_list[1].kwargs["messages"]
    notice = retry_messages[-1]["content"]
    assert "python-exec" in notice, f"the correction must name the fence that executes: {notice!r}"


@pytest.mark.asyncio
async def test_repeated_tool_call_markup_is_not_recorded_as_success(storage):
    """A model that keeps handing back markup executed nothing all run, so the
    recorded outcome must not read as success."""
    agent = _agent(storage)
    _patch(agent, return_value=_resp(TOOL_CALL_REPLY))

    await agent.run("commit anything outstanding")

    assert _end_status(storage) != "success"


@pytest.mark.asyncio
async def test_ordinary_prose_answer_still_ends_the_run_as_success(storage):
    """The guard keys on tool-call markup alone - a plain answer is untouched."""
    agent = _agent(storage)
    mock = _patch(agent, return_value=_resp("The tree is clean; nothing to commit."))

    result = await agent.run("commit anything outstanding")

    assert mock.await_count == 1
    assert result == "The tree is clean; nothing to commit."
    assert _end_status(storage) == "success"
