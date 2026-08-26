"""Multi-step agents run on the same engine as single-shot agents.

A step is not a different kind of run, it is a different prompt inside one run.
These tests pin the consequences of that: steps share one history session, they
go through the normal preparation pipeline (so they get attachments, skills and
hooks), and the retry backoff yields to the event loop instead of blocking it.

The isolation guarantee that makes steps useful is pinned here too: a step sees
the preamble plus its own content, never a sibling's instructions, and never a
sibling's conversation.
"""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from tsugite.agent_runner.runner import _RunSetup
from tsugite.agent_runner.steps import _execute_step_with_retries
from tsugite.md_agents import parse_agent
from tsugite.options import ExecutionOptions

TWO_STEP_AGENT = """---
name: fold_in_demo
model: ollama:qwen2.5-coder:7b
extends: none
tools: []
---
Shared preamble line.

<!-- tsu:step name="gather" assign="findings" -->
GATHER_INSTRUCTIONS run the suite

<!-- tsu:step name="fix" -->
FIX_INSTRUCTIONS given {{ findings }}
"""


def _write(tmp_path: Path, body: str, name: str = "agent.md") -> Path:
    path = tmp_path / name
    path.write_text(body)
    return path


def _make_step(**overrides):
    step = MagicMock()
    step.name = "demo"
    step.max_retries = 0
    step.retry_delay = 0
    step.timeout = None
    step.assign_var = None
    step.model_kwargs = {}
    step.content = "do something"
    step.spawn_agent_path = None
    for key, value in overrides.items():
        setattr(step, key, value)
    return step


@pytest.mark.asyncio
async def test_retry_delay_yields_to_the_event_loop(monkeypatch):
    """retry_delay must not block the loop.

    The runner is awaited from the daemon's shared loop; a blocking sleep here
    stalls every other session for the duration of the backoff.
    """
    attempts = {"n": 0}

    async def fake_run_unit(*args, **kwargs):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise RuntimeError("first attempt fails")
        return "ok"

    monkeypatch.setattr("tsugite.agent_runner.steps._run_unit", fake_run_unit)

    ticks = {"n": 0}

    async def ticker():
        while True:
            ticks["n"] += 1
            await asyncio.sleep(0.005)

    tick_task = asyncio.create_task(ticker())
    try:
        await _execute_step_with_retries(
            step=_make_step(max_retries=1, retry_delay=0.2),
            step_context={},
            agent=parse_agent("---\nname: demo\nextends: none\n---\nbody"),
            i=1,
            total_steps=1,
            steps=[_make_step()],
            step_header="[Step 1/1: demo]",
            prompt="task",
            setup=_RunSetup(
                exec_options=ExecutionOptions(),
                hooks_dir=Path("."),
                hook_message="",
                hook_vars={},
                agent_stem="demo",
            ),
        )
    finally:
        tick_task.cancel()

    assert attempts["n"] == 2, "the step should have been retried"
    assert ticks["n"] > 1, "the event loop was blocked during retry_delay"


@pytest.mark.asyncio
async def test_steps_share_one_history_session(tmp_path, monkeypatch):
    """A multi-step run is one conversation, not one conversation per step.

    Each step used to call the executor with continue_conversation_id=None, so
    open_or_create_session created a fresh orphan session per step (and per
    retry, and per loop iteration), none of which held the user's prompt.
    """
    from tsugite.agent_runner import runner

    created = []
    opened = []

    def fake_open_or_create_session(*, agent_path, agent_name, model, continue_conversation_id=None, workspace=None):
        if continue_conversation_id is None:
            session = MagicMock()
            session.session_id = f"session-{len(created)}"
            created.append(session.session_id)
            return session
        opened.append(continue_conversation_id)
        session = MagicMock()
        session.session_id = continue_conversation_id
        return session

    monkeypatch.setattr(
        "tsugite.agent_runner.history_integration.open_or_create_session",
        fake_open_or_create_session,
    )

    async def fake_agent_run(self, task, return_full_result=False, stream=False):
        return "step done"

    monkeypatch.setattr("tsugite.core.agent.TsugiteAgent.run", fake_agent_run)

    agent_file = _write(tmp_path, TWO_STEP_AGENT)
    await runner.run_agent_async(agent_file, "test prompt")

    assert len(created) == 1, f"expected one history session for the whole run, got {len(created)}: {created}"


@pytest.mark.asyncio
async def test_step_prompts_go_through_the_preparation_pipeline(tmp_path, monkeypatch):
    """Steps must be prepared by AgentPreparer, not a bespoke shim.

    `available_tools` is supplied by AgentPreparer and was absent from the
    multi-step context shim, so a step template referencing it used to raise
    UndefinedError under StrictUndefined.
    """
    from tsugite.agent_runner import runner

    agent_file = _write(
        tmp_path,
        """---
name: prepared_steps
model: ollama:qwen2.5-coder:7b
extends: none
tools: []
---
<!-- tsu:step name="only" -->
Installed tool count: {{ available_tools | length }}
""",
    )

    prompts = []

    async def fake_agent_run(self, task, return_full_result=False, stream=False):
        prompts.append(task)
        return "done"

    monkeypatch.setattr("tsugite.core.agent.TsugiteAgent.run", fake_agent_run)

    await runner.run_agent_async(agent_file, "test prompt")

    assert len(prompts) == 1
    assert "Installed tool count:" in prompts[0]
    assert "{{" not in prompts[0], "template was not rendered by the preparation pipeline"
    assert "tsu:step" not in prompts[0], "the whole file was rendered instead of the step's content"


@pytest.mark.asyncio
async def test_steps_stay_isolated_from_each_other(tmp_path, monkeypatch):
    """The point of steps: a step sees the preamble and its own content only.

    Step 1 must not be able to read step 2's instructions in advance, and step 2
    must not inherit step 1's conversation - only the assigned variable crosses.
    """
    from tsugite.agent_runner import runner

    prompts = []
    histories = []

    async def fake_agent_run(self, task, return_full_result=False, stream=False):
        prompts.append(task)
        histories.append(list(self.previous_messages))
        return "FINDINGS_VALUE"

    monkeypatch.setattr("tsugite.core.agent.TsugiteAgent.run", fake_agent_run)

    agent_file = _write(tmp_path, TWO_STEP_AGENT)
    await runner.run_agent_async(agent_file, "test prompt")

    assert len(prompts) == 2, f"expected one prompt per step, got {len(prompts)}"
    gather, fix = prompts

    assert "GATHER_INSTRUCTIONS" in gather
    assert "FIX_INSTRUCTIONS" not in gather, "step 1 could see step 2's instructions"

    assert "FIX_INSTRUCTIONS" in fix
    assert "GATHER_INSTRUCTIONS" not in fix, "step 2 inherited step 1's instructions"

    assert "Shared preamble line." in gather
    assert "Shared preamble line." in fix

    assert "FINDINGS_VALUE" in fix, "the assigned variable did not reach the next step"

    assert histories[1] == [], "step 2 inherited step 1's conversation"


@pytest.mark.asyncio
async def test_step_variables_are_plain_strings_under_token_accounting(tmp_path, monkeypatch):
    """A step's assigned variable must survive injection into the Python namespace.

    With `return_token_usage=True` - which the daemon always sets, and the CLI
    sets whenever history is on - a step returns an AgentExecutionResult. That
    object is not JSON-serializable, so SubprocessExecutor.send_variables (the
    default backend) drops it with only a log line and the variable silently
    vanishes from the step's namespace.
    """
    from tsugite.agent_runner import runner
    from tsugite.options import ExecutionOptions

    injected = []

    async def fake_agent_run(self, task, return_full_result=False, stream=False):
        return "STEP_OUTPUT"

    async def capture_send_variables(self, variables):
        injected.append(variables)

    monkeypatch.setattr("tsugite.core.agent.TsugiteAgent.run", fake_agent_run)
    monkeypatch.setattr("tsugite.core.executor.LocalExecutor.send_variables", capture_send_variables)

    agent_file = _write(tmp_path, TWO_STEP_AGENT)
    await runner.run_agent_async(
        agent_file,
        "test prompt",
        exec_options=ExecutionOptions(return_token_usage=True),
    )

    assert injected, "no variables were injected into the second step"
    findings = injected[-1].get("findings")
    assert isinstance(findings, str), f"step variable is not a plain string: {type(findings).__name__}"

    json.dumps(injected[-1]["ctx"].__dict__.get("findings"))


@pytest.mark.asyncio
async def test_run_if_skip_is_not_retried_as_a_failure(tmp_path, monkeypatch):
    """`run_if` is a control-flow signal, not a step failure.

    The scheduler catches AgentSkippedError to mark a run skipped. Letting the
    step retry loop treat it as a generic exception burns every retry and then
    reports a hard error instead.
    """
    from tsugite.agent_runner import runner
    from tsugite.agent_runner.models import AgentSkippedError

    attempts = {"n": 0}

    async def fake_agent_run(self, task, return_full_result=False, stream=False):
        attempts["n"] += 1
        return "never reached"

    monkeypatch.setattr("tsugite.core.agent.TsugiteAgent.run", fake_agent_run)

    agent_file = _write(
        tmp_path,
        """---
name: guarded_steps
model: ollama:qwen2.5-coder:7b
extends: none
tools: []
run_if: "false"
---
<!-- tsu:step name="only" max_retries="3" -->
Body
""",
    )

    with pytest.raises(AgentSkippedError):
        await runner.run_agent_async(agent_file, "test prompt")

    assert attempts["n"] == 0, "a skipped step should not have run the agent loop"


@pytest.mark.asyncio
async def test_each_step_records_its_own_turn_not_a_repeat_of_the_prompt(tmp_path, monkeypatch):
    """The shared session should read as the staged workflow, not the same
    message N times. Every step used to record the top-level prompt verbatim."""
    from tsugite.agent_runner import runner

    recorded = []

    def fake_record_user_input(storage, text, **kwargs):
        recorded.append(text)

    monkeypatch.setattr("tsugite.agent_runner.history_integration.record_user_input", fake_record_user_input)

    async def fake_agent_run(self, task, return_full_result=False, stream=False):
        return "done"

    monkeypatch.setattr("tsugite.core.agent.TsugiteAgent.run", fake_agent_run)

    agent_file = _write(tmp_path, TWO_STEP_AGENT)
    await runner.run_agent_async(agent_file, "triage the build")

    assert recorded == ["[Step 1/2: gather]", "[Step 2/2: fix]"], f"got {recorded}"


@pytest.mark.asyncio
async def test_steps_never_resume_a_provider_session(tmp_path, monkeypatch):
    """Isolation has to hold on the resume channel too.

    For session-owning providers (claude_code, ACP) the previous_messages list is
    not where state lives - resuming the same provider session id is. Steps
    sharing one resume_session would inherit each other's conversation there
    while every message-list assertion still passed.
    """
    from tsugite.agent_runner import runner

    resume_args = []

    async def fake_agent_run(self, task, return_full_result=False, stream=False):
        resume_args.append(self._resume_session)
        return "done"

    monkeypatch.setattr("tsugite.core.agent.TsugiteAgent.run", fake_agent_run)
    monkeypatch.setattr(
        "tsugite.agent_runner.history_integration.get_resumable_session_state",
        lambda cid: SimpleNamespace(session_id="provider-session-1", compacted=False),
    )

    agent_file = _write(tmp_path, TWO_STEP_AGENT)
    await runner.run_agent_async(agent_file, "test prompt", continue_conversation_id="conv-1")

    assert resume_args == [None, None], f"steps resumed a provider session: {resume_args}"


@pytest.mark.asyncio
async def test_spawn_steps_get_the_same_pipeline_variables(tmp_path, monkeypatch):
    """A step that delegates via `agent=` is still a step.

    It used to be the one branch that bypassed AgentPreparer, so it silently
    lost tool directives and every pipeline-supplied variable. Because the
    preamble is prepended to all steps, an agent mixing normal and spawn steps
    broke on the spawn steps only.
    """
    from tsugite.agent_runner import runner

    fixture = Path(__file__).parent / "fixtures" / "agents" / "simple.md"
    assert fixture.exists()

    captured = {}

    def fake_spawn_agent(agent_path, prompt, **kwargs):
        captured["prompt"] = prompt
        return "SPAWNED_OK"

    monkeypatch.setattr("tsugite.tools.agents.spawn_agent", fake_spawn_agent)

    agent_file = _write(
        tmp_path,
        f"""---
name: spawn_pipeline
model: ollama:qwen2.5-coder:7b
extends: none
tools: []
---
Preamble mentions {{{{ agent_name }}}}.

<!-- tsu:step name="review" agent="{fixture}" assign="verdict" -->
Tools installed: {{{{ available_tools | length }}}}
""",
    )

    result = await runner.run_agent_async(agent_file, "test prompt")

    assert result == "SPAWNED_OK"
    assert "Preamble mentions spawn_pipeline." in captured["prompt"]
    assert "Tools installed:" in captured["prompt"]
    assert "{{" not in captured["prompt"], "spawn step prompt was not fully rendered"


@pytest.mark.asyncio
async def test_only_assigned_step_results_cross_between_steps(tmp_path, monkeypatch):
    """Pins the scoping contract deliberately rather than by accident.

    A step's own tool/exec directive variables belong to that step; only a
    step's `assign=` result crosses. Preamble directives keep working for every
    step because the preamble is prepended to each one and re-executes.
    """
    from tsugite.agent_runner import runner

    prompts = []

    async def fake_agent_run(self, task, return_full_result=False, stream=False):
        prompts.append(task)
        return "RESULT"

    monkeypatch.setattr("tsugite.core.agent.TsugiteAgent.run", fake_agent_run)

    agent_file = _write(
        tmp_path,
        """---
name: scoping
model: ollama:qwen2.5-coder:7b
extends: none
tools: []
---
<!-- tsu:exec name="preamble_calc" assign="shared" -->
"from-preamble"
<!-- /tsu:exec -->
preamble={{ shared }}

<!-- tsu:step name="first" assign="carried" -->
<!-- tsu:exec name="step_calc" assign="step_local" -->
"from-step-one"
<!-- /tsu:exec -->
local={{ step_local }}

<!-- tsu:step name="second" -->
carried={{ carried }} local_is_defined={{ step_local is defined }}
""",
    )

    await runner.run_agent_async(agent_file, "test prompt")

    first, second = prompts
    assert "local=from-step-one" in first
    assert "preamble=from-preamble" in first
    # The preamble re-executes for every step, so its variable is still there.
    assert "preamble=from-preamble" in second
    # The assigned step result crosses; the step's own directive variable does not.
    assert "carried=RESULT" in second
    assert "local_is_defined=False" in second


@pytest.mark.asyncio
async def test_multistep_run_aggregates_token_usage(tmp_path, monkeypatch):
    """A multi-step run must honour `return_token_usage`.

    Steps suppress their own accounting so their results stay plain strings, so
    the run as a whole is the only place the totals can be assembled. Callers
    that asked for the rich shape - the daemon always does - dereference
    `.token_count` on what comes back.
    """
    from tsugite.agent_runner import runner
    from tsugite.agent_runner.models import AgentExecutionResult

    async def fake_agent_run(self, task, return_full_result=False, stream=False):
        self.total_tokens = 100
        self.total_cost = 0.25
        self.cost_reported = True
        self.last_input_tokens = 80
        self.cache_creation_tokens = 5
        self.cache_read_tokens = 10
        self.memory.add_step(thought="t", code="c", output="o")
        return "STEP_OUTPUT"

    monkeypatch.setattr("tsugite.core.agent.TsugiteAgent.run", fake_agent_run)

    agent_file = _write(tmp_path, TWO_STEP_AGENT)
    result = await runner.run_agent_async(
        agent_file,
        "test prompt",
        exec_options=ExecutionOptions(return_token_usage=True),
    )

    assert isinstance(result, AgentExecutionResult), f"multi-step returned a bare {type(result).__name__}"
    assert result.response == "STEP_OUTPUT"
    assert result.token_count == 200, "per-step tokens were not summed across the run"
    assert result.cost == pytest.approx(0.5)
    assert result.cache_creation_tokens == 10
    assert result.cache_read_tokens == 20
    assert result.step_count == 2
    assert result.last_input_tokens == 80, "context size is the final step's, not the sum"


@pytest.mark.asyncio
async def test_multistep_run_returns_a_string_without_token_usage(tmp_path, monkeypatch):
    """The CLI path (history off) still gets a plain string back."""
    from tsugite.agent_runner import runner

    async def fake_agent_run(self, task, return_full_result=False, stream=False):
        return "STEP_OUTPUT"

    monkeypatch.setattr("tsugite.core.agent.TsugiteAgent.run", fake_agent_run)

    agent_file = _write(tmp_path, TWO_STEP_AGENT)
    result = await runner.run_agent_async(
        agent_file,
        "test prompt",
        exec_options=ExecutionOptions(return_token_usage=False),
    )

    assert result == "STEP_OUTPUT"


@pytest.mark.asyncio
async def test_looping_step_accumulates_every_iteration(tmp_path, monkeypatch):
    """A repeating step spends on every pass, so every pass counts."""
    from tsugite.agent_runner import runner
    from tsugite.agent_runner.models import AgentExecutionResult

    async def fake_agent_run(self, task, return_full_result=False, stream=False):
        self.total_tokens = 100
        return "STEP_OUTPUT"

    monkeypatch.setattr("tsugite.core.agent.TsugiteAgent.run", fake_agent_run)

    agent_file = _write(
        tmp_path,
        """---
name: looping
model: ollama:qwen2.5-coder:7b
extends: none
tools: []
---
<!-- tsu:step name="poll" repeat_while="iteration < 3" -->
Poll again.
""",
    )
    result = await runner.run_agent_async(
        agent_file,
        "test prompt",
        exec_options=ExecutionOptions(return_token_usage=True),
    )

    assert isinstance(result, AgentExecutionResult)
    assert result.token_count == 300, "only some iterations of the looping step were counted"
    # None, not 0.0: nothing reported a cost, which is not the same as free.
    assert result.cost is None
