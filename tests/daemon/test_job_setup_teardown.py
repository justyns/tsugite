"""Job setup/teardown commands around the worker."""

import asyncio
import os
import signal
import time
from pathlib import Path

import pytest
from tsugite_daemon.job_store import JobState

from .test_jobs_orchestrator import _make_git_repo


@pytest.fixture
def repo(tmp_path):
    path = tmp_path / "repo"
    _make_git_repo(path)
    return path


async def _drain(orchestrator):
    while orchestrator._bg_tasks:
        await asyncio.gather(*orchestrator._bg_tasks)


async def _create_async(orchestrator, repo, **kwargs):
    job, _ = await orchestrator.create_and_start_job(
        parent_session_id="parent-1", prompt="do", repo=str(repo), **kwargs
    )
    await _drain(orchestrator)
    return job


def _create(orchestrator, repo, **kwargs):
    return asyncio.run(_create_async(orchestrator, repo, **kwargs))


def test_setup_runs_in_the_worktree_before_the_worker_starts(orchestrator, runner, repo, store):
    seen_at_spawn = []
    start = runner.start_session

    def start_session(session, **kw):
        seen_at_spawn.append((Path(session.workspace_override) / "setup_ran").exists())
        return start(session, **kw)

    runner.start_session = start_session

    job = _create(orchestrator, repo, setup="pwd > setup_ran")

    worktree = Path(store.get(job.id).worktree_path)
    assert (worktree / "setup_ran").read_text().strip() == str(worktree)
    assert seen_at_spawn == [True]
    assert store.get(job.id).state == JobState.RUNNING.value


def test_failed_setup_errors_the_job_without_starting_a_worker(orchestrator, runner, repo, store):
    job = _create(orchestrator, repo, setup="echo boom >&2; exit 3")

    job = store.get(job.id)
    assert job.state == JobState.ERRORED.value
    assert job.error == "setup failed: exit code 3"
    assert "boom" in job.error_detail
    assert runner.started == []


def test_setup_that_backgrounds_a_process_returns(orchestrator, repo, store):
    started = time.monotonic()

    job = _create(orchestrator, repo, setup="sleep 30 > /dev/null & echo $! > server.pid", timeout_minutes=1)

    assert time.monotonic() - started < 10
    assert store.get(job.id).state == JobState.RUNNING.value
    os.kill(int((Path(store.get(job.id).worktree_path) / "server.pid").read_text()), signal.SIGTERM)


@pytest.mark.asyncio
async def test_cancel_during_setup_starts_no_worker_and_tears_down_after_setup(
    orchestrator, runner, repo, store, tmp_path
):
    log = tmp_path / "log"
    job, started = await orchestrator.create_and_start_job(
        parent_session_id="parent-1",
        prompt="do",
        repo=str(repo),
        setup=f"sleep 1; echo up >> {log}",
        teardown=f"echo down >> {log}",
    )
    assert started is None

    await orchestrator.cancel_job(job.id)
    await _drain(orchestrator)

    assert log.read_text().split() == ["up", "down"]
    assert runner.started == []
    assert store.get(job.id).state == JobState.CANCELLED.value


@pytest.mark.parametrize(
    ("teardown_when", "terminal", "runs"),
    [
        ("always", JobState.DONE, True),
        ("always", JobState.STUCK, True),
        ("on_success", JobState.DONE, True),
        ("on_success", JobState.ERRORED, False),
        ("on_failure", JobState.CANCELLED, True),
        ("on_failure", JobState.DONE, False),
    ],
)
def test_teardown_runs_in_the_worktree_on_matching_terminal_states(
    orchestrator, repo, store, tmp_path, teardown_when, terminal, runs
):
    marker = tmp_path / "torn_down"
    job = _create(orchestrator, repo, teardown=f"pwd > {marker}", teardown_when=teardown_when)
    worktree = store.get(job.id).worktree_path
    if terminal == JobState.DONE:
        store.update_state(job.id, JobState.VERIFYING.value)

    orchestrator._finalize(store.get(job.id), terminal)

    if runs:
        assert marker.read_text().strip() == worktree
    else:
        assert not marker.exists()
    assert Path(worktree).exists() == (terminal not in (JobState.DONE, JobState.CANCELLED))


@pytest.mark.asyncio
async def test_teardown_runs_before_the_prune_with_a_running_loop(orchestrator, repo, store, tmp_path):
    marker = tmp_path / "torn_down"
    job = await _create_async(orchestrator, repo, teardown=f"pwd > {marker}")
    worktree = store.get(job.id).worktree_path

    await orchestrator.cancel_job(job.id)
    await _drain(orchestrator)

    assert marker.read_text().strip() == worktree
    assert not Path(worktree).exists()


def test_teardown_runs_once_when_a_parked_job_is_cancelled(orchestrator, repo, store, tmp_path):
    log = tmp_path / "log"
    job = _create(orchestrator, repo, teardown=f"echo down >> {log}")

    orchestrator._finalize(store.get(job.id), JobState.STUCK)
    asyncio.run(orchestrator.cancel_job(job.id))

    assert log.read_text().split() == ["down"]


def test_mark_done_runs_an_on_success_teardown(orchestrator, repo, store, tmp_path):
    marker = tmp_path / "torn_down"
    job = _create(orchestrator, repo, teardown=f"touch {marker}", teardown_when="on_success")
    orchestrator._finalize(store.get(job.id), JobState.STUCK)
    assert not marker.exists()

    asyncio.run(orchestrator.mark_done_manual(job.id))

    assert marker.exists()


@pytest.mark.asyncio
async def test_boot_recovery_runs_the_teardown(orchestrator, repo, store, tmp_path):
    marker = tmp_path / "torn_down"
    job = await _create_async(orchestrator, repo, teardown=f"touch {marker}")

    orchestrator.recover_orphaned_jobs()
    await _drain(orchestrator)

    assert store.get(job.id).state == JobState.ERRORED.value
    assert marker.exists()


def test_unknown_teardown_when_is_rejected(orchestrator, repo, store):
    with pytest.raises(ValueError, match="teardown_when"):
        _create(orchestrator, repo, teardown="true", teardown_when="sometimes")
    assert store.list_all() == []


@pytest.mark.parametrize(("teardown_when", "setup_runs"), [("always", 2), ("on_success", 1)])
def test_retry_reruns_setup_only_after_teardown_ran(orchestrator, repo, store, tmp_path, teardown_when, setup_runs):
    counter = tmp_path / "setup_runs"
    job = _create(orchestrator, repo, setup=f"echo x >> {counter}", teardown="true", teardown_when=teardown_when)
    orchestrator._finalize(store.get(job.id), JobState.STUCK)

    asyncio.run(orchestrator.retry_with_hint(job.id, hint="try again"))

    assert len(counter.read_text().splitlines()) == setup_runs


@pytest.mark.asyncio
async def test_retry_waits_for_the_previous_teardown_before_setup(orchestrator, repo, store, tmp_path):
    log = tmp_path / "log"
    job = await _create_async(orchestrator, repo, setup=f"echo up >> {log}", teardown=f"sleep 1; echo down >> {log}")
    orchestrator._finalize(store.get(job.id), JobState.STUCK)

    await orchestrator.retry_with_hint(job.id, hint="try again")

    assert log.read_text().split() == ["up", "down", "up"]


def test_failed_setup_on_retry_keeps_the_job_parked_with_the_output(orchestrator, repo, store, tmp_path):
    flag = tmp_path / "fail_setup"
    job = _create(orchestrator, repo, setup=f"test ! -e {flag} || (echo not again >&2; exit 4)", teardown="true")
    orchestrator._finalize(store.get(job.id), JobState.STUCK)
    flag.touch()

    with pytest.raises(RuntimeError, match="setup failed"):
        asyncio.run(orchestrator.retry_with_hint(job.id, hint="try again"))

    job = store.get(job.id)
    assert job.state == JobState.STUCK.value
    assert "not again" in job.error_detail


def test_retry_reruns_a_setup_that_failed(orchestrator, repo, store, tmp_path):
    flag, counter = tmp_path / "fail", tmp_path / "setup_runs"
    flag.touch()
    job = _create(
        orchestrator,
        repo,
        setup=f"echo x >> {counter}; test ! -e {flag}",
        teardown="true",
        teardown_when="on_success",
    )
    assert store.get(job.id).state == JobState.ERRORED.value
    flag.unlink()

    asyncio.run(orchestrator.retry_with_hint(job.id, hint="try again"))

    assert len(counter.read_text().splitlines()) == 2
    assert store.get(job.id).state == JobState.RUNNING.value


def test_fresh_workspace_retry_reruns_setup(orchestrator, repo, store, tmp_path):
    counter = tmp_path / "setup_runs"
    job = _create(orchestrator, repo, setup=f"echo x >> {counter}", teardown="true", teardown_when="on_success")
    orchestrator._finalize(store.get(job.id), JobState.STUCK)

    asyncio.run(orchestrator.retry_with_hint(job.id, hint="try again", fresh_workspace=True))

    assert len(counter.read_text().splitlines()) == 2


@pytest.mark.asyncio
async def test_cancel_during_retry_setup_starts_no_worker_and_tears_down_after_setup(
    orchestrator, runner, repo, store, tmp_path
):
    log, started, go = tmp_path / "log", tmp_path / "setup_started", tmp_path / "go"
    go.touch()
    job = await _create_async(
        orchestrator,
        repo,
        setup=f"touch {started}; until [ -e {go} ]; do sleep 0.02; done; echo up >> {log}",
        teardown=f"echo down >> {log}",
    )
    started.unlink()
    go.unlink()
    orchestrator._finalize(store.get(job.id), JobState.STUCK)
    workers_before = len(runner.started)

    retry = asyncio.create_task(orchestrator.retry_with_hint(job.id, hint="try again"))
    while not started.exists() and not retry.done():
        await asyncio.sleep(0.02)
    await orchestrator.cancel_job(job.id)
    go.touch()
    await retry
    await _drain(orchestrator)

    assert log.read_text().split() == ["up", "down", "up", "down"]
    assert len(runner.started) == workers_before
    assert store.get(job.id).state == JobState.CANCELLED.value


@pytest.mark.asyncio
async def test_cancel_while_retry_waits_for_the_previous_teardown_skips_setup(
    orchestrator, runner, repo, store, tmp_path
):
    log, go = tmp_path / "log", tmp_path / "go"
    job = await _create_async(
        orchestrator,
        repo,
        setup=f"echo up >> {log}",
        teardown=f"until [ -e {go} ]; do sleep 0.02; done; echo down >> {log}",
    )
    orchestrator._finalize(store.get(job.id), JobState.STUCK)
    workers_before = len(runner.started)

    retry = asyncio.create_task(orchestrator.retry_with_hint(job.id, hint="try again"))
    await asyncio.sleep(0)
    await orchestrator.cancel_job(job.id)
    go.touch()
    await retry
    await _drain(orchestrator)

    assert log.read_text().split() == ["up", "down"]
    assert len(runner.started) == workers_before
    assert store.get(job.id).state == JobState.CANCELLED.value


def test_cancel_after_a_failed_retry_setup_runs_teardown(orchestrator, repo, store, tmp_path):
    flag, log = tmp_path / "fail", tmp_path / "log"
    job = _create(orchestrator, repo, setup=f"echo up >> {log}; test ! -e {flag}", teardown=f"echo down >> {log}")
    orchestrator._finalize(store.get(job.id), JobState.STUCK)
    flag.touch()
    with pytest.raises(RuntimeError, match="setup failed"):
        asyncio.run(orchestrator.retry_with_hint(job.id, hint="try again"))

    asyncio.run(orchestrator.cancel_job(job.id))

    assert log.read_text().split() == ["up", "down", "up", "down"]
