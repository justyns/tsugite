"""Pluggable job-executor seam: registration, dispatch, and the complete/fail/cancel API.

A non-agent executor (e.g. a PTY-driven CLI) runs a Job and reports outcomes via
`complete_worker` / `fail_worker`, reusing the ENTIRE existing verifier/AC/retry/
stuck machinery. These tests extend the FakeRunner/FakeStore patterns from
test_jobs_orchestrator.py and add a FakeExecutor that records start/cancel calls.
"""

import asyncio

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


def _seed_running_executor_job(store, orchestrator, *, executor="fake", acceptance_criteria=None, workspace_path=None):
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
