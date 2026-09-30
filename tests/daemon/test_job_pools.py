"""Named job pools: a slot per active job, FIFO queue behind a full pool."""

import asyncio

import pytest
from tsugite_daemon.config import PoolConfig, load_daemon_config
from tsugite_daemon.job_store import JobState, JobStateTransitionError
from tsugite_daemon.jobs_orchestrator import JobsOrchestrator


async def _spawn(orchestrator, **kwargs):
    job, started = await orchestrator.create_and_start_job(parent_session_id="parent-1", prompt="do", **kwargs)
    return job, started


async def _drain(orchestrator):
    while orchestrator._bg_tasks:
        await asyncio.gather(*orchestrator._bg_tasks)
        # Awaiting already-done tasks does not yield, so their discard callbacks need a turn.
        await asyncio.sleep(0)


@pytest.fixture
def pooled(store, runner):
    return JobsOrchestrator(store, runner, pools={"dev": PoolConfig(max=1, queue_max=1), "ops": PoolConfig(max=1)})


@pytest.mark.asyncio
async def test_no_pools_section_caps_default_pool_at_three(orchestrator, runner, store):
    jobs = [(await _spawn(orchestrator))[0] for _ in range(4)]

    states = [store.get(j.id).state for j in jobs]
    assert states == ["running"] * 3 + ["queued"]
    assert len(runner.started) == 3


@pytest.mark.asyncio
async def test_unknown_pool_fails_and_lists_defined_pools(pooled, store):
    with pytest.raises(ValueError, match=r"Unknown job pool: 'nope'.*default.*dev.*ops"):
        await _spawn(pooled, pool="nope")

    assert store.list_all() == []


@pytest.mark.asyncio
async def test_terminal_job_starts_next_queued_job_fifo(pooled, runner, store):
    first, _ = await _spawn(pooled, pool="dev")
    second, started = await _spawn(pooled, pool="dev")
    assert started is None
    assert store.get(second.id).state == JobState.QUEUED.value
    assert len(runner.started) == 1

    await pooled.cancel_job(first.id)
    await _drain(pooled)

    assert store.get(second.id).state == JobState.RUNNING.value
    assert len(runner.started) == 2


@pytest.mark.asyncio
async def test_job_cancelled_before_its_dequeued_start_runs_never_starts(store, runner):
    orch = JobsOrchestrator(store, runner, pools={"dev": PoolConfig(max=1)})
    first, _ = await _spawn(orch, pool="dev")
    second, _ = await _spawn(orch, pool="dev")

    await orch.cancel_job(first.id)
    await orch.cancel_job(second.id)
    await _drain(orch)

    assert store.get(second.id).state == JobState.CANCELLED.value
    assert len(runner.started) == 1


@pytest.mark.asyncio
async def test_queued_jobs_start_in_submission_order(store, runner):
    orch = JobsOrchestrator(store, runner, pools={"dev": PoolConfig(max=1)})
    running, _ = await _spawn(orch, pool="dev")
    second, _ = await _spawn(orch, pool="dev")
    third, _ = await _spawn(orch, pool="dev")

    await orch.cancel_job(running.id)
    await _drain(orch)

    assert store.get(second.id).state == JobState.RUNNING.value
    assert store.get(third.id).state == JobState.QUEUED.value


@pytest.mark.asyncio
async def test_full_queue_makes_spawn_fail_without_creating_a_job(pooled, store):
    await _spawn(pooled, pool="dev")
    await _spawn(pooled, pool="dev")

    with pytest.raises(ValueError, match="pool 'dev' is full"):
        await _spawn(pooled, pool="dev")

    assert len(store.list_all()) == 2


@pytest.mark.asyncio
async def test_pools_do_not_share_slots(pooled, store):
    await _spawn(pooled, pool="dev")
    other, _ = await _spawn(pooled, pool="ops")

    assert store.get(other.id).state == JobState.RUNNING.value


@pytest.mark.asyncio
async def test_slot_is_held_while_verifying(pooled, runner, store):
    first, _ = await _spawn(pooled, pool="dev", acceptance_criteria=["the thing works"])
    second, _ = await _spawn(pooled, pool="dev")

    await pooled.on_session_complete(runner.started[0], "done")
    await _drain(pooled)

    assert store.get(first.id).state == JobState.VERIFYING.value
    assert store.get(second.id).state == JobState.QUEUED.value


@pytest.mark.asyncio
async def test_cancelling_a_queued_job_frees_its_place_in_the_queue(pooled, store):
    await _spawn(pooled, pool="dev")
    waiting, _ = await _spawn(pooled, pool="dev")

    await pooled.cancel_job(waiting.id)
    third, _ = await _spawn(pooled, pool="dev")

    assert store.get(third.id).state == JobState.QUEUED.value


@pytest.mark.asyncio
async def test_failed_start_of_queued_job_errors_it_and_starts_the_next(store, runner):
    orch = JobsOrchestrator(store, runner, pools={"dev": PoolConfig(max=1)})
    first, _ = await _spawn(orch, pool="dev")
    doomed, _ = await _spawn(orch, pool="dev")
    last, _ = await _spawn(orch, pool="dev")
    start = runner.start_session

    def failing_once(session, inherit_from=None):
        runner.start_session = start
        raise RuntimeError("no capacity")

    runner.start_session = failing_once
    await orch.cancel_job(first.id)
    await _drain(orch)

    assert store.get(doomed.id).state == JobState.ERRORED.value
    assert store.get(last.id).state == JobState.RUNNING.value


def test_release_without_a_running_loop_keeps_the_job_queued_and_unslotted(pooled, store):
    first = store.get(asyncio.run(_spawn(pooled, pool="dev"))[0].id)
    second = asyncio.run(_spawn(pooled, pool="dev"))[0]

    pooled._release_slot(first)

    assert pooled._pool_slots["dev"] == set()
    assert pooled._pool_waiting["dev"] == [second.id]


def test_pools_load_from_daemon_yaml(tmp_path):
    cfg = tmp_path / "daemon.yaml"
    cfg.write_text(
        f"default_workspace_dir: {tmp_path}\npools:\n  default: {{max: 4}}\n  dev: {{max: 2, queue_max: 10}}\n"
    )

    pools = load_daemon_config(cfg).pools

    assert pools["dev"] == PoolConfig(max=2, queue_max=10)
    assert pools["default"].max == 4 and pools["default"].queue_max is None


@pytest.mark.asyncio
async def test_finalizing_a_job_whose_pool_was_removed_from_config(store, runner):
    old = JobsOrchestrator(store, runner, pools={"gpu": PoolConfig(max=1)})
    job, _ = await _spawn(old, pool="gpu")
    old._finalize(job, JobState.STUCK)
    restarted = JobsOrchestrator(store, runner)

    await restarted.cancel_job(job.id)

    assert store.get(job.id).state == JobState.CANCELLED.value


async def _parked_job(orch):
    job, _ = await _spawn(orch, pool="dev")
    orch._finalize(job, JobState.STUCK)
    return job


@pytest.mark.asyncio
async def test_retry_with_hint_refuses_when_the_pool_is_full(store, runner):
    orch = JobsOrchestrator(store, runner, pools={"dev": PoolConfig(max=1)})
    parked = await _parked_job(orch)
    await _spawn(orch, pool="dev")

    with pytest.raises(JobStateTransitionError, match="pool 'dev' is full"):
        await orch.retry_with_hint(parked.id, "try again")

    assert store.get(parked.id).state == JobState.STUCK.value
    assert len(orch._pool_slots["dev"]) == 1


@pytest.mark.asyncio
async def test_retry_with_hint_holds_a_slot_until_the_job_finalizes(store, runner):
    orch = JobsOrchestrator(store, runner, pools={"dev": PoolConfig(max=1)})
    parked = await _parked_job(orch)

    await orch.retry_with_hint(parked.id, "try again")
    waiting, _ = await _spawn(orch, pool="dev")

    assert store.get(parked.id).state == JobState.RUNNING.value
    assert store.get(waiting.id).state == JobState.QUEUED.value


@pytest.mark.asyncio
async def test_failed_retry_gives_its_slot_back(store, runner):
    orch = JobsOrchestrator(store, runner, pools={"dev": PoolConfig(max=1)})
    parked = await _parked_job(orch)

    def failing(session, inherit_from=None):
        raise RuntimeError("no capacity")

    runner.start_session = failing

    with pytest.raises(RuntimeError):
        await orch.retry_with_hint(parked.id, "try again")

    assert orch._pool_slots["dev"] == set()
