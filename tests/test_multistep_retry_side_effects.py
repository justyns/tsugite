"""`_execute_step_with_retries` in tsugite/agent_runner/steps.py retries a
failed step by invoking the full agent run again. If the step's first attempt
already executed code (with possible side effects), the second attempt will
happily re-issue those calls. Gate retry on whether any step-level code ran.
"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from tsugite.agent_runner.runner import _RunSetup
from tsugite.agent_runner.steps import _execute_step_with_retries
from tsugite.exceptions import AgentExecutionError
from tsugite.md_agents import parse_agent
from tsugite.options import ExecutionOptions


def _setup() -> _RunSetup:
    return _RunSetup(
        exec_options=ExecutionOptions(),
        hooks_dir=Path("."),
        hook_message="",
        hook_vars={},
        agent_stem="demo",
    )


def _agent():
    return parse_agent("---\nname: demo\nextends: none\n---\nbody")


def _make_step(max_retries: int = 2):
    step = MagicMock()
    step.name = "demo"
    step.max_retries = max_retries
    step.retry_delay = 0
    step.timeout = None
    step.assign_var = None
    step.model_kwargs = {}
    step.content = "do something"
    step.spawn_agent_path = None
    return step


@pytest.mark.asyncio
async def test_retry_skipped_when_prior_attempt_executed_code(monkeypatch):
    step = _make_step(max_retries=2)
    call_count = {"n": 0}

    async def fake_execute(*args, **kwargs):
        call_count["n"] += 1
        exec_step = MagicMock()
        exec_step.code = "x = http_request('POST', ...)"
        raise AgentExecutionError("something went wrong", execution_steps=[exec_step])

    monkeypatch.setattr("tsugite.agent_runner.steps._run_unit", fake_execute)

    with pytest.raises(Exception):
        await _execute_step_with_retries(
            step=step,
            step_context={},
            agent=_agent(),
            i=1,
            total_steps=1,
            steps=[step],
            step_header="Step 1",
            prompt="task",
            setup=_setup(),
        )

    assert call_count["n"] == 1, f"step retried after side-effecting code ran; fired {call_count['n']} times"


@pytest.mark.asyncio
async def test_retry_still_fires_for_pre_execution_failures(monkeypatch):
    """If the step failed before any code ran (template / setup error), retry
    is safe and should still happen.
    """
    step = _make_step(max_retries=2)
    call_count = {"n": 0}

    async def fake_execute(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] < 2:
            raise AgentExecutionError("pre-exec error", execution_steps=[])
        return "ok"

    monkeypatch.setattr("tsugite.agent_runner.steps._run_unit", fake_execute)

    result, _duration = await _execute_step_with_retries(
        step=step,
        step_context={},
        agent=_agent(),
        i=1,
        total_steps=1,
        steps=[step],
        step_header="Step 1",
        prompt="task",
        setup=_setup(),
    )
    assert result == "ok"
    assert call_count["n"] == 2
