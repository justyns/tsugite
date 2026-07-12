"""Tests for the agent-facing jobs tool wrappers in tsugite.tools.jobs.

Covers list_jobs filters + the limit clamp, get_job, and - load-bearing for this
session's fix - that list_jobs/get_job go through JobStore's LOCKED accessors
(list_all()/get()) instead of reaching into the private `._jobs._jobs` dict.

No daemon, no real LLM/network. The module-level `_jobs_orchestrator` global is
wired via monkeypatch (mirroring tests/daemon/test_job_command_parser.py); the
@tool(require_daemon=True) decorator returns the bare function, so we call the
tools directly.
"""

import pytest
from tsugite_daemon.job_store import Job, JobState, JobStore

import tsugite.tools.jobs as jobs_tool


class _StoreSpy:
    """Wraps a real JobStore and records which accessors the tools call.

    Crucially it also exposes a private `_jobs` dict that is DELIBERATELY EMPTY,
    while the locked accessors (get/list_all/...) delegate to the wrapped real
    store. So any caller that bypasses the locked methods and reaches into
    `._jobs._jobs` (the pre-fix behaviour) would observe zero jobs and the
    assertions below would fail - that's what makes these tests load-bearing.
    """

    def __init__(self, real: JobStore):
        self._real = real
        self._jobs: dict = {}  # intentionally empty trap for `._jobs._jobs` access
        self.list_all_calls = 0
        self.get_calls: list[str] = []

    def list_all(self) -> list[Job]:
        self.list_all_calls += 1
        return self._real.list_all()

    def get(self, job_id: str):
        self.get_calls.append(job_id)
        return self._real.get(job_id)


class _OrchStub:
    def __init__(self, store):
        self._jobs = store


@pytest.fixture
def real_store(tmp_path):
    return JobStore(tmp_path / "jobs.json")


@pytest.fixture
def store_spy(real_store):
    return _StoreSpy(real_store)


@pytest.fixture
def wire(monkeypatch, store_spy):
    """Point the tool module's global at an orchestrator stub backed by the spy."""
    monkeypatch.setattr(jobs_tool, "_jobs_orchestrator", _OrchStub(store_spy))
    return store_spy


def _add(store: JobStore, *, parent="parent-1", prompt="do x", created_at=None, state=None) -> Job:
    job = store.add(Job(id="", parent_session_id=parent, prompt=prompt))
    fields = {}
    if created_at is not None:
        fields["created_at"] = created_at
    if fields:
        store.update(job.id, **fields)
    if state is not None:
        store.update(job.id, state=state)
    return store.get(job.id)


# ── get_job ──


def test_get_job_uses_locked_get_accessor(wire, real_store):
    """get_job must resolve the record via the store's locked .get(), not by
    indexing the private dict."""
    job = _add(real_store, prompt="alpha")
    result = jobs_tool.get_job(job.id)
    assert result["id"] == job.id
    assert result["prompt"] == "alpha"
    # Load-bearing: proves get_job went through the locked accessor.
    assert wire.get_calls == [job.id], "get_job must call store.get(), not reach into ._jobs._jobs"


def test_get_job_unknown_returns_error(wire, real_store):
    result = jobs_tool.get_job("job-does-not-exist")
    assert result == {"error": "Unknown job: job-does-not-exist"}


def test_get_job_uninitialised_orchestrator_returns_error(monkeypatch):
    monkeypatch.setattr(jobs_tool, "_jobs_orchestrator", None)
    assert jobs_tool.get_job("job-x") == {"error": "Jobs orchestrator not initialised"}


# ── list_jobs ──


def test_list_jobs_uses_locked_list_all_accessor(wire, real_store):
    """list_jobs must enumerate via store.list_all() (locked), not list(._jobs._jobs.values())."""
    _add(real_store, prompt="one")
    _add(real_store, prompt="two")
    out = jobs_tool.list_jobs()
    assert wire.list_all_calls == 1, "list_jobs must call list_all() exactly once"
    assert len(out) == 2, "list_jobs must see jobs via the locked accessor, not the empty private dict"
    assert {row["prompt"] for row in out} == {"one", "two"}


def test_list_jobs_filters_by_session_id(wire, real_store):
    _add(real_store, parent="p1", prompt="mine")
    _add(real_store, parent="p2", prompt="theirs")
    out = jobs_tool.list_jobs(session_id="p1")
    assert [row["prompt"] for row in out] == ["mine"]


def test_list_jobs_filters_by_state(wire, real_store):
    _add(real_store, prompt="queued-one")  # stays QUEUED
    running = _add(real_store, prompt="running-one")
    real_store.update_state(running.id, JobState.RUNNING.value)
    out = jobs_tool.list_jobs(state=JobState.RUNNING.value)
    assert [row["id"] for row in out] == [running.id]
    assert out[0]["state"] == JobState.RUNNING.value


def test_list_jobs_filters_by_since_and_until(wire, real_store):
    _add(real_store, prompt="old", created_at="2020-01-01T00:00:00+00:00")
    _add(real_store, prompt="mid", created_at="2022-06-01T00:00:00+00:00")
    _add(real_store, prompt="new", created_at="2024-12-31T00:00:00+00:00")

    since_only = {row["prompt"] for row in jobs_tool.list_jobs(since="2022-01-01T00:00:00+00:00")}
    assert since_only == {"mid", "new"}

    until_only = {row["prompt"] for row in jobs_tool.list_jobs(until="2022-12-31T00:00:00+00:00")}
    assert until_only == {"old", "mid"}

    windowed = {
        row["prompt"]
        for row in jobs_tool.list_jobs(since="2021-01-01T00:00:00+00:00", until="2023-01-01T00:00:00+00:00")
    }
    assert windowed == {"mid"}


def test_list_jobs_orders_newest_first(wire, real_store):
    _add(real_store, prompt="old", created_at="2020-01-01T00:00:00+00:00")
    _add(real_store, prompt="new", created_at="2024-01-01T00:00:00+00:00")
    _add(real_store, prompt="mid", created_at="2022-01-01T00:00:00+00:00")
    out = jobs_tool.list_jobs()
    assert [row["prompt"] for row in out] == ["new", "mid", "old"]


def test_list_jobs_respects_limit(wire, real_store):
    for i in range(5):
        _add(real_store, prompt=f"j{i}", created_at=f"20{20 + i}-01-01T00:00:00+00:00")
    out = jobs_tool.list_jobs(limit=2)
    assert len(out) == 2
    # Newest first → the two highest years.
    assert [row["prompt"] for row in out] == ["j4", "j3"]


@pytest.mark.parametrize("bad_limit", [0, -3])
def test_list_jobs_limit_clamped_to_at_least_one(wire, real_store, bad_limit):
    """`max(1, int(limit))` clamp: a limit of 0 or negative must still return one
    row, not an empty/reversed slice."""
    _add(real_store, prompt="a", created_at="2021-01-01T00:00:00+00:00")
    _add(real_store, prompt="b", created_at="2022-01-01T00:00:00+00:00")
    out = jobs_tool.list_jobs(limit=bad_limit)
    assert len(out) == 1, f"limit={bad_limit} must clamp to 1 row, got {len(out)}"
    assert out[0]["prompt"] == "b", "the single clamped row must be the newest"


def test_list_jobs_row_shape_is_lean(wire, real_store):
    """The returned rows expose only the lean projection, not the full Job dict."""
    job = _add(real_store, prompt="x" * 200)
    real_store.update(job.id, verify_attempts=2, error="boom line one\nline two")
    [row] = jobs_tool.list_jobs()
    assert set(row.keys()) == {
        "id",
        "state",
        "prompt",
        "created_at",
        "resolved_at",
        "worker_session_id",
        "verifier_session_id",
        "verify_attempts",
        "error",
    }
    assert len(row["prompt"]) == 120, "prompt must be truncated to 120 chars"
    assert row["error"] == "boom line one", "error must be first-line-only"
    assert row["verify_attempts"] == 2


def test_list_jobs_uninitialised_orchestrator_returns_error(monkeypatch):
    monkeypatch.setattr(jobs_tool, "_jobs_orchestrator", None)
    assert jobs_tool.list_jobs() == [{"error": "Jobs orchestrator not initialised"}]


# ── cancel_job / respond_to_job tool wrappers ──


def test_cancel_job_tool_wraps_orchestrator_cancel(monkeypatch):
    from types import SimpleNamespace

    seen = {}

    def fake_cancel(job_id, reason):
        seen["args"] = (job_id, reason)
        return SimpleNamespace(id=job_id, state="cancelled")

    monkeypatch.setattr(jobs_tool, "_jobs_orchestrator", SimpleNamespace(cancel_job=fake_cancel))
    monkeypatch.setattr(jobs_tool, "_call", lambda fn, *a, timeout=30, **k: fn(*a, **k))

    out = jobs_tool.cancel_job("job-1", reason="agent gave up")
    assert seen["args"] == ("job-1", "agent gave up")
    assert out == {"job_id": "job-1", "state": "cancelled"}


def test_cancel_job_tool_surfaces_errors_as_dict(monkeypatch):
    from types import SimpleNamespace

    def boom(job_id, reason):
        raise ValueError("Unknown job: job-x")

    monkeypatch.setattr(jobs_tool, "_jobs_orchestrator", SimpleNamespace(cancel_job=boom))
    monkeypatch.setattr(jobs_tool, "_call", lambda fn, *a, timeout=30, **k: fn(*a, **k))
    assert jobs_tool.cancel_job("job-x") == {"error": "Unknown job: job-x"}

    monkeypatch.setattr(jobs_tool, "_jobs_orchestrator", None)
    assert jobs_tool.cancel_job("job-x") == {"error": "Jobs orchestrator not initialised"}


def test_respond_to_job_tool_wraps_orchestrator_respond(monkeypatch):
    from types import SimpleNamespace

    seen = {}

    def fake_respond(job_id, message):
        seen["args"] = (job_id, message)
        return SimpleNamespace(id=job_id, state="running")

    monkeypatch.setattr(jobs_tool, "_jobs_orchestrator", SimpleNamespace(respond_to_job=fake_respond))
    monkeypatch.setattr(jobs_tool, "_call", lambda fn, *a, timeout=30, **k: fn(*a, **k))

    out = jobs_tool.respond_to_job("job-1", "the codeword is swordfish")
    assert seen["args"] == ("job-1", "the codeword is swordfish")
    assert out == {"job_id": "job-1", "state": "running"}


def test_respond_to_job_tool_surfaces_errors_as_dict(monkeypatch):
    from types import SimpleNamespace

    def boom(job_id, message):
        raise ValueError("job 'job-x' is cancelled")

    monkeypatch.setattr(jobs_tool, "_jobs_orchestrator", SimpleNamespace(respond_to_job=boom))
    monkeypatch.setattr(jobs_tool, "_call", lambda fn, *a, timeout=30, **k: fn(*a, **k))
    assert jobs_tool.respond_to_job("job-x", "hi") == {"error": "job 'job-x' is cancelled"}

    monkeypatch.setattr(jobs_tool, "_jobs_orchestrator", None)
    assert jobs_tool.respond_to_job("job-x", "hi") == {"error": "Jobs orchestrator not initialised"}


# ── spawn_job timeout behaviour ──


def test_spawn_job_forwards_executor(monkeypatch):
    """The executor param (default 'agent') must reach create_and_start_job so an
    agent can spawn a job on a non-agent executor."""
    from types import SimpleNamespace

    monkeypatch.setattr(jobs_tool, "_jobs_orchestrator", SimpleNamespace(create_and_start_job=lambda **k: None))
    monkeypatch.setattr("tsugite_daemon.session_runner.get_current_session_id", lambda: "parent-1")

    seen = {}

    def fake_call(fn, *args, timeout=30, **kwargs):
        seen.update(kwargs)
        return SimpleNamespace(id="job-1"), SimpleNamespace(id="session-1")

    monkeypatch.setattr(jobs_tool, "_call", fake_call)

    jobs_tool.spawn_job(prompt="x")
    assert seen["executor"] == "agent", "default executor must forward as 'agent'"
    jobs_tool.spawn_job(prompt="x", executor="cc")
    assert seen["executor"] == "cc", "explicit executor must forward through"


def test_spawn_job_forwards_effort(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(jobs_tool, "_jobs_orchestrator", SimpleNamespace(create_and_start_job=lambda **k: None))
    monkeypatch.setattr("tsugite_daemon.session_runner.get_current_session_id", lambda: "parent-1")

    seen = {}

    def fake_call(fn, *args, timeout=30, **kwargs):
        seen.update(kwargs)
        return SimpleNamespace(id="job-1"), SimpleNamespace(id="session-1")

    monkeypatch.setattr(jobs_tool, "_call", fake_call)

    jobs_tool.spawn_job(prompt="x")
    assert seen["effort"] is None, "effort defaults to None"
    jobs_tool.spawn_job(prompt="x", executor="cc", effort="high")
    assert seen["effort"] == "high", "explicit effort must forward through"


def test_spawn_job_timeout_is_generous_and_warns_about_duplicates(monkeypatch):
    """Worktree provisioning for --repo can exceed the default 30s bridge
    timeout. The coroutine keeps running on the daemon loop even when the
    bridge gives up, so the agent must be told the job may still exist instead
    of blindly retrying into a duplicate."""
    import concurrent.futures
    from types import SimpleNamespace

    monkeypatch.setattr(jobs_tool, "_jobs_orchestrator", SimpleNamespace(create_and_start_job=lambda **k: None))
    monkeypatch.setattr("tsugite_daemon.session_runner.get_current_session_id", lambda: "parent-1")

    seen = {}

    def fake_call(fn, *args, timeout=30, **kwargs):
        seen["timeout"] = timeout
        raise concurrent.futures.TimeoutError()

    monkeypatch.setattr(jobs_tool, "_call", fake_call)

    with pytest.raises(RuntimeError, match="list_jobs"):
        jobs_tool.spawn_job(prompt="x", repo="/some/repo")
    assert seen["timeout"] >= 120, "spawn must allow slow worktree provisioning"
