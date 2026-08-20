"""Job completion barrier: notify a parent once its last active job finishes."""

import asyncio
from types import SimpleNamespace

import pytest
from tsugite_daemon.job_prompts import _build_barrier_message
from tsugite_daemon.job_store import Job, JobState

from .test_jobs_orchestrator import _verifier_session, _worker_session


def _seed_batch(store, orchestrator, count, parent="parent-1", notify_when="all_done"):
    """Create `count` running jobs anchored on `parent`, all opted into the barrier."""
    jobs = []
    for i in range(count):
        job = store.add(Job(id="", parent_session_id=parent, prompt=f"task {i}", notify_when=notify_when))
        orchestrator.register_worker(job.id, f"worker-{job.id}", timeout_minutes=30)
        jobs.append(store.get(job.id))
    return jobs


def _capture_replies(runner):
    sent = []

    async def fake_reply(session_id, message, source="session", metadata=None):
        sent.append({"session_id": session_id, "message": message, "source": source, "metadata": metadata})
        return "ok"

    runner.reply_to_session = fake_reply
    return sent


async def _finish(orchestrator, job):
    await orchestrator.on_session_complete(_worker_session(job, f"worker-{job.id}"), "done")
    await asyncio.sleep(0)


# ── intake ──


@pytest.mark.asyncio
async def test_all_done_is_a_valid_notify_when(store, orchestrator):
    job, _ = await orchestrator.create_and_start_job(parent_session_id="parent-1", prompt="hi", notify_when="all_done")
    assert store.get(job.id).notify_when == "all_done"


# ── store query ──


def test_list_active_for_parent_excludes_terminal_and_other_parents(store):
    active = store.add(Job(id="", parent_session_id="parent-1", prompt="a"))
    store.update_state(active.id, JobState.RUNNING.value)
    resolved = store.add(Job(id="", parent_session_id="parent-1", prompt="b"))
    store.update_state(resolved.id, JobState.RUNNING.value)
    store.update_state(resolved.id, JobState.CANCELLED.value)
    other = store.add(Job(id="", parent_session_id="parent-2", prompt="c"))
    store.update_state(other.id, JobState.RUNNING.value)

    assert [j.id for j in store.list_active_for_parent("parent-1")] == [active.id]


def test_list_active_for_parent_counts_awaiting_input_as_active(store):
    job = store.add(Job(id="", parent_session_id="parent-1", prompt="a"))
    store.update_state(job.id, JobState.RUNNING.value)
    store.update_state(job.id, JobState.AWAITING_INPUT.value)

    assert [j.id for j in store.list_active_for_parent("parent-1")] == [job.id]


# ── the barrier ──


@pytest.mark.asyncio
async def test_barrier_fires_once_when_the_last_job_finishes(store, runner, orchestrator):
    sent = _capture_replies(runner)
    jobs = _seed_batch(store, orchestrator, 3)

    await _finish(orchestrator, jobs[0])
    await _finish(orchestrator, jobs[1])
    assert sent == [], "barrier must stay silent while a job is still active"

    await _finish(orchestrator, jobs[2])
    assert len(sent) == 1, "the last job finishing must fire exactly one aggregate notify"
    assert sent[0]["session_id"] == "parent-1"
    assert sent[0]["source"] == "jobs_all_complete"
    assert sent[0]["metadata"] == {"kind": "jobs_barrier"}
    for job in jobs:
        assert job.id in sent[0]["message"]


@pytest.mark.asyncio
async def test_barrier_ignores_jobs_of_another_parent(store, runner, orchestrator):
    sent = _capture_replies(runner)
    mine = _seed_batch(store, orchestrator, 1, parent="parent-1")
    _seed_batch(store, orchestrator, 1, parent="parent-2")

    await _finish(orchestrator, mine[0])
    assert len(sent) == 1
    assert sent[0]["session_id"] == "parent-1"


@pytest.mark.asyncio
async def test_barrier_waits_for_a_non_barrier_sibling(store, runner, orchestrator):
    """The batch is every `all_done` job, but the barrier waits on ALL active jobs."""
    sent = _capture_replies(runner)
    barrier_job = _seed_batch(store, orchestrator, 1)[0]
    plain = _seed_batch(store, orchestrator, 1, notify_when="never")[0]

    await _finish(orchestrator, barrier_job)
    assert sent == [], "a still-running sibling must hold the barrier closed"

    await _finish(orchestrator, plain)
    assert len(sent) == 1
    assert barrier_job.id in sent[0]["message"]
    assert plain.id not in sent[0]["message"], "only all_done jobs are summarised"


@pytest.mark.asyncio
async def test_barrier_does_not_fire_when_no_job_opted_in(store, runner, orchestrator):
    sent = _capture_replies(runner)
    jobs = _seed_batch(store, orchestrator, 2, notify_when="never")

    await _finish(orchestrator, jobs[0])
    await _finish(orchestrator, jobs[1])
    assert sent == []


@pytest.mark.asyncio
async def test_verifying_to_running_retry_does_not_fire_the_barrier(store, runner, orchestrator):
    sent = _capture_replies(runner)
    done_job, retried = _seed_batch(store, orchestrator, 2)
    store.update(retried.id, acceptance_criteria=["ships"])
    retried = store.get(retried.id)

    await _finish(orchestrator, done_job)
    # retried: worker -> verifier -> rejected -> fresh worker (verifying -> running)
    await orchestrator.on_session_complete(_worker_session(retried, f"worker-{retried.id}"), "attempt 1")
    await orchestrator.on_session_complete(
        _verifier_session(store.get(retried.id)),
        '{"ac_results": [{"ac_text": "ships", "pass": false, "reason": "no"}], "overall_pass": false}',
    )
    await asyncio.sleep(0)
    assert store.get(retried.id).state == JobState.RUNNING.value
    assert sent == [], "a mid-retry job is still active; the barrier must not fire"


@pytest.mark.asyncio
async def test_mark_done_manual_closes_the_batch(store, runner, orchestrator):
    sent = _capture_replies(runner)
    stuck_job = _seed_batch(store, orchestrator, 1)[0]
    store.update_state(stuck_job.id, JobState.STUCK.value)

    await orchestrator.mark_done_manual(stuck_job.id)
    await asyncio.sleep(0)
    assert len(sent) == 1, "a manual mark-done must close the batch too"
    assert stuck_job.id in sent[0]["message"]
    assert sent[0]["source"] == "jobs_all_complete"


@pytest.mark.asyncio
async def test_boot_recovery_fires_no_barrier(store, runner, orchestrator):
    sent = _capture_replies(runner)
    _seed_batch(store, orchestrator, 2)

    assert orchestrator.recover_orphaned_jobs() == 2
    await asyncio.sleep(0)
    assert sent == [], "boot recovery must not wake parents"


@pytest.mark.asyncio
async def test_retried_job_rejoins_the_next_batch(store, runner, orchestrator):
    sent = _capture_replies(runner)
    job = _seed_batch(store, orchestrator, 1)[0]

    orchestrator._finalize(job, JobState.ERRORED, error="boom")
    await asyncio.sleep(0)
    assert len(sent) == 1
    assert store.get(job.id).barrier_notified is True

    # A retry re-arms the job so the next batch reports it again.
    orchestrator._activate_worker(job.id, f"worker-{job.id}", kind="hint", timeout_minutes=30, clear_error=True)
    assert store.get(job.id).barrier_notified is False

    await _finish(orchestrator, store.get(job.id))
    assert len(sent) == 2, "the retried job must close a second batch"
    assert job.id in sent[1]["message"]


@pytest.mark.asyncio
async def test_barrier_reports_each_job_only_once(store, runner, orchestrator):
    """A job already summarised must not reappear when a later batch closes."""
    sent = _capture_replies(runner)
    first = _seed_batch(store, orchestrator, 1)[0]
    await _finish(orchestrator, first)
    assert len(sent) == 1

    second = _seed_batch(store, orchestrator, 1)[0]
    await _finish(orchestrator, second)
    assert len(sent) == 2
    assert second.id in sent[1]["message"]
    assert first.id not in sent[1]["message"]


# ── per-job notify_when is untouched ──


@pytest.mark.asyncio
async def test_all_done_does_not_fire_the_per_job_notify(store, runner, orchestrator):
    sent = _capture_replies(runner)
    jobs = _seed_batch(store, orchestrator, 2)

    await _finish(orchestrator, jobs[0])
    await _finish(orchestrator, jobs[1])
    assert [s["source"] for s in sent] == ["jobs_all_complete"]


@pytest.mark.asyncio
async def test_terminal_notify_still_fires_alongside_a_barrier_job(store, runner, orchestrator):
    sent = _capture_replies(runner)
    barrier_job = _seed_batch(store, orchestrator, 1)[0]
    per_job = _seed_batch(store, orchestrator, 1, notify_when="terminal")[0]

    await _finish(orchestrator, per_job)
    assert [s["source"] for s in sent] == ["job_complete"]

    await _finish(orchestrator, barrier_job)
    assert [s["source"] for s in sent] == ["job_complete", "jobs_all_complete"]


# ── message ──


def test_barrier_message_summarises_counts_and_outcomes():
    jobs = [
        Job(id="job-a", parent_session_id="p", prompt="write the docs", state=JobState.DONE.value),
        Job(id="job-b", parent_session_id="p", prompt="fix the bug", state=JobState.DONE.value),
        Job(
            id="job-c",
            parent_session_id="p",
            prompt="ship it" + "x" * 200,
            state=JobState.STUCK.value,
            error="AC1 not met\nmore detail",
        ),
    ]
    msg = _build_barrier_message(jobs)

    assert "3" in msg
    assert "2 done" in msg
    assert "1 stuck" in msg
    for job in jobs:
        assert job.id in msg
    assert "write the docs" in msg
    assert "AC1 not met" in msg
    assert "more detail" not in msg
    assert "x" * 200 not in msg
    assert "get_job" in msg


class _TurnState:
    """Minimal session store: just the turn flag the barrier consults."""

    def __init__(self, in_flight: bool):
        self.in_flight = in_flight

    def get_session(self, _session_id):
        return SimpleNamespace(turn_in_flight=self.in_flight)


@pytest.mark.asyncio
async def test_a_batch_spawned_across_one_turn_reports_once(store, runner, orchestrator):
    """A job can finish before its siblings are spawned, so a batch does not close
    while the turn that is filling it is still running."""
    sent = _capture_replies(runner)
    turn = _TurnState(in_flight=True)
    runner._store = turn

    first = _seed_batch(store, orchestrator, 1)[0]
    await _finish(orchestrator, first)
    second = _seed_batch(store, orchestrator, 1)[0]
    await _finish(orchestrator, second)
    assert [m for m in sent if m["source"] == "jobs_all_complete"] == []

    turn.in_flight = False
    orchestrator.close_batch_barrier("parent-1")
    await asyncio.sleep(0)

    barriers = [m for m in sent if m["source"] == "jobs_all_complete"]
    assert len(barriers) == 1
    assert "All 2 background job(s) finished" in barriers[0]["message"]
