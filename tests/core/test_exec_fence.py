"""The exec fence is a distinct sentinel (```python-exec) so that ordinary
prose ```python blocks (documentation, quoted code, explanations) are shown
but NOT auto-executed.

Reproduces justyns/tsugite#479: an illustrative ```python snippet inside a
prose answer used to execute, throwing a spurious runtime error.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from tsugite.core.agent import TsugiteAgent
from tsugite.providers.base import CompletionResponse, Usage


def _mock_response(content: str) -> CompletionResponse:
    return CompletionResponse(content=content, usage=Usage(total_tokens=100), cost=0.001)


def _patch_provider(agent, return_value=None):
    mock = AsyncMock(return_value=return_value)
    agent._provider = MagicMock()
    agent._provider.acompletion = mock
    agent._provider.stop = AsyncMock()
    agent._provider.get_state = MagicMock(return_value=None)
    agent._provider.set_context = MagicMock()
    return mock


def _agent(max_turns: int = 3) -> TsugiteAgent:
    return TsugiteAgent(model_string="openai:gpt-4o-mini", tools=[], instructions="", max_turns=max_turns)


@pytest.mark.asyncio
async def test_python_exec_block_is_parsed_as_code():
    parsed = _agent()._parse_response_from_text("Thought: run it.\n\n```python-exec\nx = 1\nprint(x)\n```")
    assert parsed.num_code_blocks == 1
    assert parsed.code == "x = 1\nprint(x)"
    assert "-exec" not in parsed.code


@pytest.mark.asyncio
async def test_bare_python_block_is_not_executed():
    """A bare ```python block is illustrative — the parser must not treat it as code."""
    parsed = _agent()._parse_response_from_text(
        "Here's how the whitelist works:\n\n```python\nmetadata = {k: v for k, v in old_session.items()}\n```"
    )
    assert parsed.num_code_blocks == 0
    assert parsed.code == ""


@pytest.mark.asyncio
async def test_exec_block_runs_even_when_bare_python_also_present():
    """A response mixing an executable block with an illustrative one runs only the
    exec block; the bare block is left as illustration."""
    parsed = _agent()._parse_response_from_text(
        "Running this:\n\n```python-exec\nx = compute()\n```\n\nFor reference the helper is:\n\n```python\ndef helper():\n    ...\n```"
    )
    assert parsed.num_code_blocks == 1
    assert parsed.code == "x = compute()"
    assert parsed.has_bare_python is True


@pytest.mark.asyncio
async def test_indented_python_exec_is_not_executed():
    """The exec fence must be at the start of a line — an indented one (e.g. inside a
    list item) is not an executable block."""
    parsed = _agent()._parse_response_from_text("- like so:\n    ```python-exec\n    run()\n    ```")
    assert parsed.num_code_blocks == 0
    assert parsed.code == ""


@pytest.mark.asyncio
async def test_bare_python_only_response_does_not_execute_and_nudges():
    """When the model emits only a bare ```python block, nothing runs and it gets
    a corrective next-turn hint pointing at ```python-exec."""
    agent = _agent()
    agent.executor.execute = AsyncMock()
    _patch_provider(
        agent,
        return_value=_mock_response("Sure, here's the snippet:\n\n```python\nresult = do_thing()\n```"),
    )

    await agent.run("show me")

    agent.executor.execute.assert_not_called()
    step = agent.memory.steps[-1]
    assert step.code == ""
    assert "python-exec" in (step.xml_observation or "")
