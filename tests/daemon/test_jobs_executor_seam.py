"""Pluggable job-executor seam: registration, dispatch, and the complete/fail/cancel API.

A non-agent executor (e.g. a PTY-driven CLI) runs a Job and reports outcomes via
`complete_worker` / `fail_worker`, reusing the ENTIRE existing verifier/AC/retry/
stuck machinery. These tests extend the FakeRunner/FakeStore patterns from
test_jobs_orchestrator.py and add a FakeExecutor that records start/cancel calls.
"""

import asyncio
import json

import pytest
import tsugite_daemon.jobs_orchestrator as orch_mod
from tsugite_daemon.job_store import Job, JobState

# store/runner/orchestrator/event_bus fixtures are re-exported by tests/daemon/conftest.py.


class FakeExecutor:
    """Records start/cancel calls. The test drives complete_worker/fail_worker
    itself - the executor only kicks off / tears down work, exactly like the real
    duck-typed contract (async start(job, followup), async cancel(job))."""

    def __init__(self, seq: list | None = None):
        self.starts: list[tuple[str, str | None]] = []
        self.cancels: list[str] = []
        self._seq = seq  # optional shared ordering log for the cancel-before-prune test

    async def start(self, job, followup=None):
        self.starts.append((job.id, followup))

    async def cancel(self, job):
        self.cancels.append(job.id)
        if self._seq is not None:
            self._seq.append("cancel")


def _seed_running_executor_job(
    store, orchestrator, *, executor="fake", acceptance_criteria=None, workspace_path=None, model=None
):
    """Seed a RUNNING job whose executor is non-agent, without spawning anything
    (register_worker does the RUNNING transition + timer, not a worker start)."""
    job = store.add(
        Job(
            id="",
            parent_session_id="parent-1",
            prompt="do the thing",
            acceptance_criteria=acceptance_criteria or [],
            executor=executor,
            workspace_path=workspace_path,
            model=model,
        )
    )
    orchestrator.register_worker(job.id, None, timeout_minutes=30)
    return store.get(job.id)


async def _drain(orchestrator):
    """Await any background teardown/prune tasks scheduled during finalize."""
    await asyncio.gather(*list(orchestrator._bg_tasks))


# ── registration + dispatch ──


@pytest.mark.asyncio
async def test_create_dispatches_to_registered_executor_not_runner(store, runner, orchestrator):
    ex = FakeExecutor()
    orchestrator.register_executor("fake", ex)
    job, started = await orchestrator.create_and_start_job(
        parent_session_id="parent-1",
        prompt="do it",
        acceptance_criteria=[],
        executor="fake",
    )
    assert started is None, "an executor job creates no worker Session"
    assert ex.starts == [(job.id, None)], "executor.start must be called with followup=None on the initial spawn"
    assert runner.started == [], "no agent session may be spawned for an executor job"
    fresh = store.get(job.id)
    assert fresh.state == JobState.RUNNING.value
    assert fresh.executor == "fake"


@pytest.mark.asyncio
async def test_job_creation_is_logged_with_attribution(store, runner, orchestrator, caplog):
    import logging

    orchestrator.register_executor("cc", FakeExecutor())
    with caplog.at_level(logging.INFO, logger="tsugite_daemon.jobs_orchestrator"):
        job, _ = await orchestrator.create_and_start_job(
            parent_session_id="parent-1",
            prompt="fix the flaky test",
            acceptance_criteria=[],
            executor="cc",
            spawned_by="agent-tool",
        )
    line = next(
        (r.getMessage() for r in caplog.records if "created" in r.getMessage() and job.id in r.getMessage()), None
    )
    assert line is not None, "job creation must log a traceable line"
    assert "source=agent-tool" in line, "the trigger/source must be attributable"
    assert "parent_session=parent-1" in line, "the caller session must be logged"
    assert "executor=cc" in line


class StartupFailingExecutor:
    """An executor whose start() reports a failure synchronously (as the cc
    executor does on an untrusted workspace / missing binary) - i.e. before the
    job ever leaves QUEUED. start() does NOT raise: it calls fail_worker and
    returns, exactly like CCExecutor._fail."""

    def __init__(self, orchestrator, reason="workspace is not trusted"):
        self._orch = orchestrator
        self._reason = reason
        self.cancels: list[str] = []

    async def start(self, job, followup=None):
        await self._orch.fail_worker(job.id, self._reason)

    async def cancel(self, job):
        self.cancels.append(job.id)


@pytest.mark.asyncio
async def test_startup_failure_before_running_errors_the_job_not_zombies_it(store, runner, orchestrator):
    orchestrator.register_executor("failing", StartupFailingExecutor(orchestrator))
    job, started = await orchestrator.create_and_start_job(
        parent_session_id="parent-1",
        prompt="do it",
        acceptance_criteria=["x"],
        executor="failing",
    )
    fresh = store.get(job.id)
    assert fresh.state == JobState.ERRORED.value, (
        f"a worker that fails during startup (QUEUED) must ERROR, not zombie as running; got {fresh.state}"
    )
    assert "not trusted" in (fresh.error or ""), "the executor's failure reason must be persisted, not dropped"


@pytest.mark.asyncio
async def test_fail_worker_on_queued_job_errors_it(store, runner, orchestrator):
    orchestrator.register_executor("fake", FakeExecutor())
    job = store.add(Job(id="", parent_session_id="parent-1", prompt="x", executor="fake"))
    assert store.get(job.id).state == JobState.QUEUED.value
    await orchestrator.fail_worker(job.id, "spawn blew up before RUNNING")
    fresh = store.get(job.id)
    assert fresh.state == JobState.ERRORED.value
    assert "spawn blew up" in (fresh.error or "")


@pytest.mark.asyncio
async def test_unknown_executor_at_creation_raises_before_persisting(store, runner, orchestrator):
    with pytest.raises(ValueError, match="executor"):
        await orchestrator.create_and_start_job(
            parent_session_id="parent-1",
            prompt="do it",
            acceptance_criteria=[],
            executor="does-not-exist",
        )
    assert store.list_all() == [], "an unknown executor must be rejected before the Job record is created"


# ── complete_worker → same verify/done/retry flow ──


@pytest.mark.asyncio
async def test_complete_worker_no_acs_marks_done(store, runner, orchestrator):
    orchestrator.register_executor("fake", FakeExecutor())
    job = _seed_running_executor_job(store, orchestrator, acceptance_criteria=[])
    await orchestrator.complete_worker(job.id, "the executor did the thing")
    await _drain(orchestrator)
    fresh = store.get(job.id)
    assert fresh.state == JobState.DONE.value
    assert fresh.result == {"summary": "the executor did the thing"}
    assert all(s.agent != "job_verifier" for s in runner.started), "no verifier for a no-AC job"


@pytest.mark.asyncio
async def test_complete_worker_passing_predicate_marks_done(store, runner, orchestrator, tmp_path):
    (tmp_path / "hello.txt").write_text("hi")
    orchestrator.register_executor("fake", FakeExecutor())
    job = _seed_running_executor_job(
        store, orchestrator, acceptance_criteria=["file_exists:hello.txt"], workspace_path=str(tmp_path)
    )
    await orchestrator.complete_worker(job.id, "wrote hello.txt")
    await _drain(orchestrator)
    assert store.get(job.id).state == JobState.DONE.value
    assert runner.started == [], "an all-predicate pass must not spend a verifier session"


@pytest.mark.asyncio
async def test_complete_worker_failing_ac_retries_via_executor_with_named_followup(
    store, runner, orchestrator, tmp_path
):
    ex = FakeExecutor()
    orchestrator.register_executor("fake", ex)
    job = _seed_running_executor_job(
        store, orchestrator, acceptance_criteria=["file_exists:missing.txt"], workspace_path=str(tmp_path)
    )
    await orchestrator.complete_worker(job.id, "claims it wrote the file")
    fresh = store.get(job.id)
    assert fresh.state == JobState.RUNNING.value, "a failed predicate loops back to RUNNING for a retry"
    assert fresh.verify_attempts == 1
    assert len(ex.starts) == 1, "the retry must re-invoke the executor, not spawn an agent session"
    retry_job_id, followup = ex.starts[0]
    assert retry_job_id == job.id
    assert followup and "missing.txt" in followup, "the retry followup must name the failed criterion"
    assert runner.started == [], "no agent worker session for an executor retry"


@pytest.mark.asyncio
async def test_cc_job_model_does_not_reach_the_tsugite_verifier(store, runner, orchestrator):
    # A non-agent job's `model` is the driven tool's own model (a claude CLI alias
    # like "sonnet"), NOT a tsugite provider:model - the verifier is a tsugite agent
    # and would fail with "Invalid model string format" if it inherited the alias.
    orchestrator.register_executor("cc", FakeExecutor())
    job = _seed_running_executor_job(
        store, orchestrator, executor="cc", acceptance_criteria=["tests pass"], model="sonnet"
    )
    await orchestrator.complete_worker(job.id, "## Summary\ndone\n## Acceptance criteria\n- tests pass: yes")
    verifiers = [s for s in runner.started if getattr(s, "agent_file", None) == "job_verifier"]
    assert len(verifiers) == 1, "a prose AC must spawn one verifier"
    assert verifiers[0].model is None, "cc job.model (CLI alias) must not reach the tsugite verifier"


@pytest.mark.asyncio
async def test_cc_job_prose_ac_full_verifier_round_trip_reaches_done(store, runner, orchestrator):
    # End-to-end for a cc job: complete -> verifier spawns -> verdict routes back
    # -> DONE. cc jobs have no worker Session (worker_session_id is None), so the
    # verifier's verifier_for falls back to job.id; this proves that routing works.
    orchestrator.register_executor("cc", FakeExecutor())
    job = _seed_running_executor_job(store, orchestrator, executor="cc", acceptance_criteria=["tests pass"])
    await orchestrator.complete_worker(job.id, "## Summary\ndone\n## Acceptance criteria\n- tests pass: yes")
    assert store.get(job.id).state == JobState.VERIFYING.value
    verifier = next(s for s in runner.started if getattr(s, "agent_file", None) == "job_verifier")
    verdict = json.dumps(
        {"ac_results": [{"ac_text": "tests pass", "pass": True, "reason": "ok"}], "overall_pass": True}
    )
    await orchestrator.on_session_complete(verifier, verdict)
    assert store.get(job.id).state == JobState.DONE.value


@pytest.mark.asyncio
async def test_complete_worker_on_non_running_job_is_noop(store, runner, orchestrator):
    orchestrator.register_executor("fake", FakeExecutor())
    job = _seed_running_executor_job(store, orchestrator, acceptance_criteria=[])
    await orchestrator.cancel_job(job.id, reason="user gave up")
    await _drain(orchestrator)
    assert store.get(job.id).state == JobState.CANCELLED.value
    # A late completion after cancel must not resurrect / overwrite the record.
    await orchestrator.complete_worker(job.id, "late summary that must be ignored")
    fresh = store.get(job.id)
    assert fresh.state == JobState.CANCELLED.value
    assert fresh.result is None, "a late complete_worker must not stamp a summary onto a cancelled job"


# ── fail_worker → errored/retry ──


@pytest.mark.asyncio
async def test_fail_worker_marks_errored(store, runner, orchestrator):
    orchestrator.register_executor("fake", FakeExecutor())
    job = _seed_running_executor_job(store, orchestrator, acceptance_criteria=["x"])
    await orchestrator.fail_worker(job.id, "claude process exited with code 1")
    fresh = store.get(job.id)
    assert fresh.state == JobState.ERRORED.value
    assert "exited with code 1" in (fresh.error or "")


@pytest.mark.asyncio
async def test_fail_worker_detail_lands_on_the_job_and_clears_on_retry(store, runner, orchestrator):
    ex = FakeExecutor()
    orchestrator.register_executor("fake", ex)
    job = _seed_running_executor_job(store, orchestrator, acceptance_criteria=["x"])
    await orchestrator.fail_worker(
        job.id,
        "claude session exited (code 127 - command not found)",
        detail="sh: 1: claude: not found",
    )
    await _drain(orchestrator)
    fresh = store.get(job.id)
    assert fresh.state == JobState.ERRORED.value
    assert fresh.error_detail == "sh: 1: claude: not found", "the captured output tail must persist on the Job"
    assert fresh.to_payload()["error_detail"] == "sh: 1: claude: not found", "the tile payload must carry it"

    await orchestrator.retry_with_hint(job.id, "install claude first")
    assert store.get(job.id).error_detail is None, "a retry must clear the stale failure detail with the error"


@pytest.mark.asyncio
async def test_fail_worker_on_non_running_job_is_noop(store, runner, orchestrator):
    orchestrator.register_executor("fake", FakeExecutor())
    job = _seed_running_executor_job(store, orchestrator, acceptance_criteria=[])
    await orchestrator.complete_worker(job.id, "done")  # → DONE
    await _drain(orchestrator)
    assert store.get(job.id).state == JobState.DONE.value
    await orchestrator.fail_worker(job.id, "too late to fail")
    assert store.get(job.id).state == JobState.DONE.value, "fail_worker on a terminal job must be a no-op"


# ── cancel ordering: executor.cancel BEFORE worktree prune ──


@pytest.mark.asyncio
async def test_cancel_job_cancels_executor_before_worktree_prune(store, runner, orchestrator, tmp_path, monkeypatch):
    seq: list[str] = []
    ex = FakeExecutor(seq=seq)
    orchestrator.register_executor("fake", ex)
    job = _seed_running_executor_job(store, orchestrator, acceptance_criteria=[])
    # Give the job a worktree so cancel triggers a prune; the child holds its cwd
    # open, so the executor must be cancelled first.
    store.update(job.id, worktree_path=str(tmp_path / "wt"))
    monkeypatch.setattr(orch_mod, "_prune_worktree", lambda path: seq.append("prune"))

    await orchestrator.cancel_job(job.id, reason="user clicked cancel")
    await _drain(orchestrator)

    assert store.get(job.id).state == JobState.CANCELLED.value
    assert ex.cancels == [job.id], "the executor must be told to cancel"
    assert seq == ["cancel", "prune"], f"executor.cancel must run before the worktree prune, got {seq}"
    assert store.get(job.id).worktree_path is None


# ── respond_to_job: feed supervisor input into a live executor worker ──


@pytest.mark.asyncio
async def test_respond_to_job_feeds_followup_into_running_executor(store, runner, orchestrator):
    ex = FakeExecutor()
    orchestrator.register_executor("fake", ex)
    job = _seed_running_executor_job(store, orchestrator, acceptance_criteria=[])
    result = await orchestrator.respond_to_job(job.id, "the codeword is swordfish")
    assert ex.starts == [(job.id, "the codeword is swordfish")], "the message must reach the live worker as a followup"
    assert result.state == JobState.RUNNING.value, "steering must not consume the attempt or change state"


@pytest.mark.asyncio
async def test_respond_to_job_rejects_blank_agent_and_non_running(store, runner, orchestrator):
    ex = FakeExecutor()
    orchestrator.register_executor("fake", ex)
    running = _seed_running_executor_job(store, orchestrator, acceptance_criteria=[])
    with pytest.raises(ValueError, match="message"):
        await orchestrator.respond_to_job(running.id, "   ")

    agent_job = store.add(Job(id="", parent_session_id="parent-1", prompt="agent work"))
    orchestrator.register_worker(agent_job.id, "ws-1", timeout_minutes=30)
    with pytest.raises(ValueError, match="agent"):
        await orchestrator.respond_to_job(agent_job.id, "hello")

    await orchestrator.cancel_job(running.id, reason="gave up")
    await _drain(orchestrator)
    with pytest.raises(ValueError, match="cancelled"):
        await orchestrator.respond_to_job(running.id, "hello")
    assert ex.starts == [], "no rejected respond may reach the executor"


# ── awaiting_input: worker-initiated pause / notify / resume ──


@pytest.mark.asyncio
async def test_pause_worker_parks_awaiting_input_and_notifies_parent(store, runner, orchestrator):
    sent = []

    async def fake_reply(session_id, message, source="session", metadata=None):
        sent.append((session_id, message, source))
        return "ok"

    runner.reply_to_session = fake_reply
    orchestrator.register_executor("fake", FakeExecutor())
    job = _seed_running_executor_job(store, orchestrator, acceptance_criteria=["x"])

    await orchestrator.pause_worker(job.id, "what is the codeword?")
    await asyncio.sleep(0)

    fresh = store.get(job.id)
    assert fresh.state == JobState.AWAITING_INPUT.value
    assert fresh.pending_question == "what is the codeword?"
    assert fresh.verify_attempts == 0, "a pause is within-attempt - it must not consume a verify attempt"
    assert sent and sent[0][0] == "parent-1", "the spawning session must be woken with the question"
    assert "what is the codeword?" in sent[0][1]
    assert "respond_to_job" in sent[0][1], "the wake-up must name the tool that answers/resumes"


@pytest.mark.asyncio
async def test_pause_worker_noop_when_not_running(store, runner, orchestrator):
    orchestrator.register_executor("fake", FakeExecutor())
    job = _seed_running_executor_job(store, orchestrator, acceptance_criteria=[])
    await orchestrator.cancel_job(job.id, reason="gave up")
    await _drain(orchestrator)
    await orchestrator.pause_worker(job.id, "too late")
    fresh = store.get(job.id)
    assert fresh.state == JobState.CANCELLED.value, "a pause must never revive a resolved job"
    assert fresh.pending_question is None


@pytest.mark.asyncio
async def test_respond_to_job_answers_and_resumes_awaiting_input(store, runner, orchestrator):
    ex = FakeExecutor()
    orchestrator.register_executor("fake", ex)
    job = _seed_running_executor_job(store, orchestrator, acceptance_criteria=[])
    await orchestrator.pause_worker(job.id, "codeword?")

    result = await orchestrator.respond_to_job(job.id, "swordfish")

    assert ex.starts == [(job.id, "swordfish")], "the answer must reach the executor as a followup"
    assert result.state == JobState.RUNNING.value, "answering must resume the attempt"
    assert result.pending_question is None, "the answered question must clear"


@pytest.mark.asyncio
async def test_resume_worker_returns_awaiting_input_to_running(store, runner, orchestrator):
    orchestrator.register_executor("fake", FakeExecutor())
    job = _seed_running_executor_job(store, orchestrator, acceptance_criteria=[])
    await orchestrator.pause_worker(job.id, "codeword?")
    resumed = await orchestrator.resume_worker(job.id)
    assert resumed.state == JobState.RUNNING.value
    assert resumed.pending_question is None


@pytest.mark.asyncio
async def test_awaiting_input_times_out_to_stuck(store, runner, orchestrator):
    orchestrator.register_executor("fake", FakeExecutor())
    job = _seed_running_executor_job(store, orchestrator, acceptance_criteria=[])
    await orchestrator.pause_worker(job.id, "anyone there?")
    orchestrator._on_timeout(job.id)
    await _drain(orchestrator)
    assert store.get(job.id).state == JobState.STUCK.value, "an unanswered pause must eventually park, not dangle"


@pytest.mark.asyncio
async def test_recover_orphaned_jobs_errors_awaiting_input(store, runner, orchestrator):
    orchestrator.register_executor("fake", FakeExecutor())
    job = _seed_running_executor_job(store, orchestrator, acceptance_criteria=[])
    await orchestrator.pause_worker(job.id, "q?")
    recovered = orchestrator.recover_orphaned_jobs()
    assert recovered >= 1
    assert store.get(job.id).state == JobState.ERRORED.value, "no persistence in the MVP: restart mid-pause errors"


# ── parked (STUCK/ERRORED) reaps the executor's child but keeps the worktree ──


@pytest.mark.asyncio
async def test_errored_job_tears_down_executor_but_keeps_worktree(store, runner, orchestrator, tmp_path, monkeypatch):
    seq: list[str] = []
    ex = FakeExecutor(seq=seq)
    orchestrator.register_executor("fake", ex)
    job = _seed_running_executor_job(store, orchestrator, acceptance_criteria=["x"])
    store.update(job.id, worktree_path=str(tmp_path / "wt"))
    monkeypatch.setattr(orch_mod, "_prune_worktree", lambda path: seq.append("prune"))

    await orchestrator.fail_worker(job.id, "claude session exited (code 1)")
    await _drain(orchestrator)

    fresh = store.get(job.id)
    assert fresh.state == JobState.ERRORED.value
    assert ex.cancels == [job.id], "an errored job must reap the executor's child, or the PTY leaks"
    assert seq == ["cancel"], "ERRORED keeps the worktree for inspection - no prune"
    assert fresh.worktree_path == str(tmp_path / "wt")


@pytest.mark.asyncio
async def test_stuck_job_tears_down_executor_but_keeps_worktree(store, runner, orchestrator, tmp_path, monkeypatch):
    seq: list[str] = []
    ex = FakeExecutor(seq=seq)
    orchestrator.register_executor("fake", ex)
    job = _seed_running_executor_job(
        store, orchestrator, acceptance_criteria=["file_exists:missing.txt"], workspace_path=str(tmp_path)
    )
    store.update(job.id, max_attempts=1, worktree_path=str(tmp_path / "wt"))
    monkeypatch.setattr(orch_mod, "_prune_worktree", lambda path: seq.append("prune"))

    await orchestrator.complete_worker(job.id, "claims it wrote the file")
    await _drain(orchestrator)

    fresh = store.get(job.id)
    assert fresh.state == JobState.STUCK.value
    assert ex.cancels == [job.id], "a stuck job must reap the executor's child, or the PTY leaks"
    assert seq == ["cancel"], "STUCK keeps the worktree for inspection - no prune"


# ── payload carries the new fields ──


def test_to_payload_carries_executor_and_worker_terminal_id():
    job = Job(id="job-x", parent_session_id="p1", prompt="do", executor="fake", worker_terminal_id="term-123")
    payload = job.to_payload()
    assert payload["executor"] == "fake"
    assert payload["worker_terminal_id"] == "term-123"


@pytest.mark.asyncio
async def test_emit_job_event_prefers_worker_terminal_id_field_over_lookup(store, runner, event_bus):
    """When Job.worker_terminal_id is set (a PTY-driven executor stamped it), the
    payload uses it directly and does NOT fall back to terminal_store.list_for_parent."""

    class _TerminalStoreStub:
        def __init__(self):
            self.calls = 0

        def list_for_parent(self, parent_id):
            self.calls += 1
            return []

    term_store = _TerminalStoreStub()
    orchestrator = orch_mod.JobsOrchestrator(store, runner, event_bus=event_bus, terminal_store=term_store)
    job = store.add(Job(id="", parent_session_id="parent-1", prompt="x", executor="fake"))
    store.update(job.id, worker_terminal_id="term-abc", worker_session_id="ws-1")
    orchestrator._emit_job_event(store.get(job.id))
    emitted = [call.args for call in event_bus.emit.call_args_list if call.args[0] == "job_update"]
    assert emitted, "an event must have been emitted"
    assert emitted[-1][1]["worker_terminal_id"] == "term-abc"
    assert term_store.calls == 0, "the field must win; the list_for_parent lookup must be skipped"
