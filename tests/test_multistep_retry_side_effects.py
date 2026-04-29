"""`_execute_step_with_retries` in tsugite/agent_runner/runner.py retries a
failed step by invoking the full agent run again. If the step's first attempt
already executed code (with possible side effects), the second attempt will
happily re-issue those calls. Gate retry on whether any step-level code ran.
"""

from unittest.mock import MagicMock

import pytest

from tsugite.agent_runner.runner import _execute_step_with_retries
from tsugite.exceptions import AgentExecutionError
from tsugite.options import ExecutionOptions


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

    monkeypatch.setattr(
        "tsugite.agent_runner.runner._execute_agent_with_prompt",
        fake_execute,
    )
    monkeypatch.setattr(
        "tsugite.agent_runner.runner._render_step_template",
        lambda *a, **kw: "rendered",
    )
    monkeypatch.setattr(
        "tsugite.agent_runner.runner._build_prepared_agent_for_step",
        lambda *a, **kw: MagicMock(),
    )

    with pytest.raises(Exception):
        await _execute_step_with_retries(
            step=step,
            step_context={},
            agent=MagicMock(),
            i=1,
            total_steps=1,
            steps=[step],
            step_header="Step 1",
            exec_options=ExecutionOptions(),
            custom_logger=None,
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

    monkeypatch.setattr(
        "tsugite.agent_runner.runner._execute_agent_with_prompt",
        fake_execute,
    )
    monkeypatch.setattr(
        "tsugite.agent_runner.runner._render_step_template",
        lambda *a, **kw: "rendered",
    )
    monkeypatch.setattr(
        "tsugite.agent_runner.runner._build_prepared_agent_for_step",
        lambda *a, **kw: MagicMock(),
    )

    result, _duration = await _execute_step_with_retries(
        step=step,
        step_context={},
        agent=MagicMock(),
        i=1,
        total_steps=1,
        steps=[step],
        step_header="Step 1",
        exec_options=ExecutionOptions(),
        custom_logger=None,
    )
    assert result == "ok"
    assert call_count["n"] == 2
