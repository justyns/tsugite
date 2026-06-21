"""Tests for predicate acceptance-criteria types in the Jobs system.

Predicates short-circuit the LLM verifier round-trip when an AC is mechanically
decidable. Three recognised v1 prefixes: `exit_code:`, `file_exists:`, `cmd:`.
"""

import json
from unittest.mock import MagicMock

import pytest
from tsugite_daemon.job_store import Job, JobState, JobStore
from tsugite_daemon.jobs_orchestrator import (
    JobsOrchestrator,
    _build_verifier_prompt,
    _evaluate_predicate,
    _parse_ac_predicate,
    partition_acs,
)
from tsugite_daemon.session_store import Session, SessionStatus


class FakeStore:
    """Minimal stand-in for SessionStore - records append_event + holds sessions for get_session lookups."""

    def __init__(self):
        self.events: list[tuple[str, dict]] = []
        self.sessions: dict[str, Session] = {}

    def append_event(self, session_id, event):
        self.events.append((session_id, event))

    def get_session(self, session_id: str):
        return self.sessions.get(session_id)


class FakeRunner:
    """Minimal stand-in for SessionRunner."""

    def __init__(self):
        self.store = FakeStore()
        self.started: list[Session] = []
        self.cancelled: list[str] = []
        self._next_session_id_counter = 0
        self.store.sessions["parent-1"] = Session(id="parent-1", agent="default")

    def start_session(self, session: Session) -> Session:
        self._next_session_id_counter += 1
        if not session.id:
            session.id = f"session-{self._next_session_id_counter}"
        session.status = SessionStatus.RUNNING.value
        self.started.append(session)
        return session

    def cancel_session(self, session_id: str) -> None:
        self.cancelled.append(session_id)


@pytest.fixture
def store(tmp_path):
    return JobStore(tmp_path / "jobs.json")


@pytest.fixture
def runner():
    return FakeRunner()


@pytest.fixture
def event_bus():
    return MagicMock()


@pytest.fixture
def orchestrator(store, runner, event_bus):
    return JobsOrchestrator(store, runner, event_bus=event_bus)


def _seed_running_job(store, orchestrator, runner, acceptance_criteria=None, worktree_path=None):
    job = store.add(
        Job(
            id="",
            parent_session_id="parent-1",
            prompt="do the thing",
            acceptance_criteria=acceptance_criteria or [],
            worktree_path=worktree_path,
        )
    )
    orchestrator.register_worker(job.id, "worker-1", timeout_minutes=30)
    return store.get(job.id)


def _worker_session(job, session_id="worker-1", status=SessionStatus.COMPLETED.value):
    return Session(
        id=session_id,
        agent="default",
        source="spawned",
        status=status,
        metadata={"job_id": job.id},
    )


def _verifier_session(job, session_id="verifier-1", status=SessionStatus.COMPLETED.value):
    return Session(
        id=session_id,
        agent="job_verifier",
        source="spawned",
        status=status,
        metadata={"job_id": job.id, "verifier_for": job.worker_session_id},
    )


# ── parser ──


def test_exit_code_predicate_parses():
    """Happy paths for all 4 prefix variants."""
    # exit_code:<cmd> (no expected code, defaults to 0)
    assert _parse_ac_predicate("exit_code:pytest -q") == {
        "kind": "exit_code",
        "cmd": "pytest -q",
        "expected": 0,
    }
    # exit_code:<cmd>:<n>
    assert _parse_ac_predicate("exit_code:grep foo file:1") == {
        "kind": "exit_code",
        "cmd": "grep foo file",
        "expected": 1,
    }
    # cmd:<command> (sugar for exit_code:<cmd>:0)
    assert _parse_ac_predicate("cmd:make test") == {
        "kind": "cmd",
        "cmd": "make test",
    }
    # file_exists:<path>
    assert _parse_ac_predicate("file_exists:foo.py") == {
        "kind": "file_exists",
        "path": "foo.py",
    }


def test_unrecognized_prefix_falls_through_to_prose():
    """A non-predicate text returns None so it lands in the prose bucket."""
    assert _parse_ac_predicate("nonsense:foo") is None
    assert _parse_ac_predicate("the tests pass") is None
    assert _parse_ac_predicate("") is None
    assert _parse_ac_predicate("file:foo.py") is None
    # Whitespace-only prefix is ignored
    assert _parse_ac_predicate("   ") is None


def test_parse_ac_predicate_strips_leading_whitespace():
    """Leading/trailing whitespace on the AC string must not defeat the parser."""
    assert _parse_ac_predicate("  exit_code:true  ") == {
        "kind": "exit_code",
        "cmd": "true",
        "expected": 0,
    }


def test_partition_acs_separates_predicates_and_prose():
    """partition_acs splits a list into (predicates, prose) preserving original indices."""
    acs = [
        "file_exists:foo.py",
        "the tests pass",
        "exit_code:make lint",
        "code is readable",
    ]
    predicates, prose = partition_acs(acs)
    # Predicates carry their original index from the AC list.
    assert len(predicates) == 2
    assert predicates[0]["ac_index"] == 0
    assert predicates[0]["ac_text"] == "file_exists:foo.py"
    assert predicates[0]["predicate"]["kind"] == "file_exists"
    assert predicates[1]["ac_index"] == 2
    assert predicates[1]["ac_text"] == "exit_code:make lint"
    assert predicates[1]["predicate"]["kind"] == "exit_code"
    # Prose entries carry their original index too so we can merge results.
    assert len(prose) == 2
    assert prose[0]["ac_index"] == 1
    assert prose[0]["ac_text"] == "the tests pass"
    assert prose[1]["ac_index"] == 3
    assert prose[1]["ac_text"] == "code is readable"


# ── evaluator ──


def test_exit_code_predicate_actually_runs_command(tmp_path):
    """A real subprocess.run path - exit_code:true passes, exit_code:false fails."""
    ok = _evaluate_predicate(
        {"kind": "exit_code", "cmd": "true", "expected": 0},
        cwd=str(tmp_path),
        ac_index=0,
        ac_text="exit_code:true",
        attempt=1,
    )
    assert ok["pass"] is True
    assert ok["ac_index"] == 0
    assert ok["ac_text"] == "exit_code:true"
    assert ok["attempt"] == 1

    bad = _evaluate_predicate(
        {"kind": "exit_code", "cmd": "false", "expected": 0},
        cwd=str(tmp_path),
        ac_index=1,
        ac_text="exit_code:false",
        attempt=1,
    )
    assert bad["pass"] is False
    assert "exit" in (bad["reason"] or "").lower() or "non-zero" in (bad["reason"] or "").lower()


def test_exit_code_predicate_matches_expected_nonzero(tmp_path):
    """exit_code:<cmd>:1 passes when the command exits with 1."""
    result = _evaluate_predicate(
        {"kind": "exit_code", "cmd": "false", "expected": 1},
        cwd=str(tmp_path),
        ac_index=0,
        ac_text="exit_code:false:1",
        attempt=1,
    )
    assert result["pass"] is True


def test_cmd_predicate_passes_on_zero_exit(tmp_path):
    """cmd: is sugar for exit_code:<cmd>:0."""
    result = _evaluate_predicate(
        {"kind": "cmd", "cmd": "true"},
        cwd=str(tmp_path),
        ac_index=0,
        ac_text="cmd:true",
        attempt=1,
    )
    assert result["pass"] is True


def test_file_exists_predicate_evaluates(tmp_path):
    """Predicate path resolved relative to cwd; absolute paths used as-is."""
    (tmp_path / "foo").write_text("x")
    ok = _evaluate_predicate(
        {"kind": "file_exists", "path": "foo"},
        cwd=str(tmp_path),
        ac_index=0,
        ac_text="file_exists:foo",
        attempt=1,
    )
    assert ok["pass"] is True
    assert "exist" in (ok["reason"] or "").lower()

    bad = _evaluate_predicate(
        {"kind": "file_exists", "path": "bar"},
        cwd=str(tmp_path),
        ac_index=1,
        ac_text="file_exists:bar",
        attempt=1,
    )
    assert bad["pass"] is False


def test_file_exists_absolute_path(tmp_path):
    """An absolute path bypasses cwd resolution."""
    target = tmp_path / "abs"
    target.write_text("x")
    result = _evaluate_predicate(
        {"kind": "file_exists", "path": str(target)},
        cwd=None,
        ac_index=0,
        ac_text=f"file_exists:{target}",
        attempt=1,
    )
    assert result["pass"] is True


def test_predicate_timeout_marks_fail(tmp_path, monkeypatch):
    """A subprocess timeout returns pass=False with reason='timeout'."""
    import subprocess

    def fake_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=args[0] if args else "sleep", timeout=kwargs.get("timeout", 30))

    monkeypatch.setattr("tsugite_daemon.jobs_orchestrator.subprocess.run", fake_run)

    result = _evaluate_predicate(
        {"kind": "exit_code", "cmd": "sleep 60", "expected": 0},
        cwd=str(tmp_path),
        ac_index=0,
        ac_text="exit_code:sleep 60",
        attempt=1,
    )
    assert result["pass"] is False
    assert result["reason"] == "timeout"


def test_predicate_stderr_in_reason_on_failure(tmp_path):
    """On failure, the first ~100 chars of stderr append to reason."""
    result = _evaluate_predicate(
        {"kind": "exit_code", "cmd": "sh -c 'echo oops >&2; exit 1'", "expected": 0},
        cwd=str(tmp_path),
        ac_index=0,
        ac_text="exit_code:sh -c 'echo oops >&2; exit 1'",
        attempt=1,
    )
    assert result["pass"] is False
    assert "oops" in (result["reason"] or "")


def test_predicate_stderr_truncated_to_100_chars(tmp_path):
    """Long stderr is truncated."""
    long = "y" * 500
    result = _evaluate_predicate(
        {"kind": "exit_code", "cmd": f"sh -c 'echo {long} >&2; exit 1'", "expected": 0},
        cwd=str(tmp_path),
        ac_index=0,
        ac_text="exit_code:sh -c ...",
        attempt=1,
    )
    assert result["pass"] is False
    # Reason includes only ~100 chars of stderr (plus the framing).
    assert "y" * 500 not in (result["reason"] or "")
    assert "y" in (result["reason"] or "")


def test_predicate_evaluation_error_is_handled(tmp_path, monkeypatch):
    """An unexpected exception during predicate eval marks fail (not crash)."""

    def boom(*args, **kwargs):
        raise OSError("permission denied")

    monkeypatch.setattr("tsugite_daemon.jobs_orchestrator.subprocess.run", boom)

    result = _evaluate_predicate(
        {"kind": "exit_code", "cmd": "anything", "expected": 0},
        cwd=str(tmp_path),
        ac_index=0,
        ac_text="exit_code:anything",
        attempt=1,
    )
    assert result["pass"] is False
    assert "evaluation error" in (result["reason"] or "").lower()


def test_cmd_predicate_with_no_cwd_is_not_run_and_marked_unmet(monkeypatch):
    """A cmd:/exit_code: predicate on a job with NO worktree/repo (cwd resolves to
    None) must NOT be executed in the daemon's own cwd. Running it there could
    accidentally pass/fail against the daemon's filesystem. The predicate must be
    short-circuited to pass=False BEFORE subprocess.run is ever called."""
    ran: list = []

    def spy_run(*args, **kwargs):
        ran.append((args, kwargs))
        raise AssertionError("subprocess.run must NOT be called when cwd is None")

    monkeypatch.setattr("tsugite_daemon.jobs_orchestrator.subprocess.run", spy_run)

    result = _evaluate_predicate(
        {"kind": "exit_code", "cmd": "true", "expected": 0},
        cwd=None,
        ac_index=0,
        ac_text="exit_code:true",
        attempt=1,
    )
    assert ran == [], "predicate command must not be executed without a working directory"
    assert result["pass"] is False, "cmd predicate with no cwd must be recorded as unmet"
    assert "working directory" in (result["reason"] or "").lower()


@pytest.mark.asyncio
async def test_cmd_predicate_job_without_worktree_marks_criterion_unmet(store, runner, orchestrator, monkeypatch):
    """End-to-end through the orchestrator: a job with a cmd: predicate but no
    worktree_path/repo (cwd=None) must record the criterion as failed without
    shelling out, and drive the retry/stuck path rather than DONE."""
    ran: list = []

    def spy_run(*args, **kwargs):
        ran.append((args, kwargs))
        raise AssertionError("subprocess.run must NOT be called when the job has no cwd")

    monkeypatch.setattr("tsugite_daemon.jobs_orchestrator.subprocess.run", spy_run)

    # worktree_path defaults to None and no repo is set → _resolve_predicate_cwd → None.
    job = _seed_running_job(store, orchestrator, runner, acceptance_criteria=["cmd:true"])
    await orchestrator.on_session_complete(_worker_session(job), "claimed done")

    fresh = store.get(job.id)
    assert ran == [], "no command may run when the job has neither worktree nor repo"
    assert fresh.state != JobState.DONE.value, "an unmet command predicate must not flip the job to DONE"
    assert fresh.ac_results, "the unmet predicate must still be recorded in ac_results"
    assert fresh.ac_results[0]["pass"] is False


# ── orchestrator flow: predicate-only ──


@pytest.mark.asyncio
async def test_predicate_only_job_skips_verifier_spawn(store, runner, orchestrator, tmp_path):
    """A Job with all-predicate ACs never spawns the LLM verifier; transitions
    straight to DONE on pass."""
    (tmp_path / "foo.py").write_text("# hi\n")
    job = _seed_running_job(
        store,
        orchestrator,
        runner,
        acceptance_criteria=["file_exists:foo.py", "exit_code:true"],
        worktree_path=str(tmp_path),
    )
    await orchestrator.on_session_complete(_worker_session(job), "all done")

    final = store.get(job.id)
    assert final.state == JobState.DONE.value, f"all-passing predicate-only job must reach DONE, got {final.state}"
    # No verifier session in runner.started.
    assert all(s.agent_file != "job_verifier" for s in runner.started), "predicate-only job must NOT spawn the verifier"
    # ac_results populated with predicate verdicts.
    assert final.ac_results is not None
    assert len(final.ac_results) == 2
    assert all(r["pass"] for r in final.ac_results)


@pytest.mark.asyncio
async def test_predicate_failure_triggers_retry(store, runner, orchestrator, tmp_path):
    """A failing predicate must drive the retry path (verify_attempts increments
    and a new worker spawns), not DONE."""
    # foo.py is missing → file_exists predicate fails.
    job = _seed_running_job(
        store,
        orchestrator,
        runner,
        acceptance_criteria=["file_exists:foo.py"],
        worktree_path=str(tmp_path),
    )
    await orchestrator.on_session_complete(_worker_session(job), "claimed done")

    refreshed = store.get(job.id)
    assert refreshed.verify_attempts == 1, "failed predicate must burn one verify attempt"
    assert refreshed.state == JobState.RUNNING.value, "failed predicate under cap must loop, not stick"
    # Loop-back worker spawned.
    loop_workers = [s for s in runner.started if (s.metadata or {}).get("loop_attempt")]
    assert len(loop_workers) == 1


@pytest.mark.asyncio
async def test_predicate_failure_at_max_attempts_marks_stuck(store, runner, orchestrator, tmp_path):
    """All-predicate job with persistently failing predicate hits STUCK at max_attempts."""
    job = store.add(
        Job(
            id="",
            parent_session_id="parent-1",
            prompt="x",
            acceptance_criteria=["file_exists:never-exists.txt"],
            worktree_path=str(tmp_path),
            max_attempts=2,
        )
    )
    orchestrator.register_worker(job.id, "w0", timeout_minutes=30)

    # Round 1: worker → predicate fail → retry
    await orchestrator.on_session_complete(_worker_session(store.get(job.id), "w0"), "w0")
    assert store.get(job.id).state == JobState.RUNNING.value
    # Round 2: worker → predicate fail → STUCK
    await orchestrator.on_session_complete(_worker_session(store.get(job.id), "w1"), "w1")
    assert store.get(job.id).state == JobState.STUCK.value


# ── orchestrator flow: mixed predicate + prose ──


@pytest.mark.asyncio
async def test_mixed_predicate_and_prose_acs(store, runner, orchestrator, tmp_path):
    """Job with 2 predicates + 2 prose. Verifier prompt contains ONLY the 2 prose
    ACs. Predicate results pre-recorded with correct ac_index. Final ac_results
    has 4 entries across the full AC index space."""
    (tmp_path / "foo.py").write_text("# hi\n")
    job = _seed_running_job(
        store,
        orchestrator,
        runner,
        acceptance_criteria=[
            "file_exists:foo.py",  # ac_index 0, predicate, passes
            "the tests pass",  # ac_index 1, prose
            "exit_code:true",  # ac_index 2, predicate, passes
            "code is readable",  # ac_index 3, prose
        ],
        worktree_path=str(tmp_path),
    )
    await orchestrator.on_session_complete(_worker_session(job), "## Summary\nDid the thing.")
    refreshed = store.get(job.id)

    # Job should be VERIFYING (prose ACs still need LLM check).
    assert refreshed.state == JobState.VERIFYING.value

    # Verifier prompt contains ONLY the prose ACs, not the predicates.
    verifier_spawns = [s for s in runner.started if s.agent_file == "job_verifier"]
    assert len(verifier_spawns) == 1
    vprompt = verifier_spawns[0].prompt
    assert "the tests pass" in vprompt
    assert "code is readable" in vprompt
    assert "file_exists:foo.py" not in vprompt, "predicate ACs must NOT leak into verifier prompt"
    assert "exit_code:true" not in vprompt

    # Predicate ac_results already recorded with their ORIGINAL ac_index (0 and 2).
    assert refreshed.ac_results is not None
    predicate_results = sorted(refreshed.ac_results, key=lambda r: r["ac_index"])
    assert len(predicate_results) == 2
    assert predicate_results[0]["ac_index"] == 0
    assert predicate_results[0]["ac_text"] == "file_exists:foo.py"
    assert predicate_results[0]["pass"] is True
    assert predicate_results[1]["ac_index"] == 2
    assert predicate_results[1]["ac_text"] == "exit_code:true"
    assert predicate_results[1]["pass"] is True

    # Now finish the verifier with both prose ACs passing.
    verifier_json = json.dumps(
        {
            "ac_results": [
                {"ac_text": "the tests pass", "pass": True, "reason": "yes"},
                {"ac_text": "code is readable", "pass": True, "reason": "yes"},
            ],
            "overall_pass": True,
        }
    )
    await orchestrator.on_session_complete(_verifier_session(store.get(job.id), "v1"), verifier_json)

    final = store.get(job.id)
    assert final.state == JobState.DONE.value, "all predicates + all prose pass → DONE"
    # Final ac_results has 4 entries covering ac_index 0..3.
    assert len(final.ac_results) == 4
    by_index = {r["ac_index"]: r for r in final.ac_results}
    assert set(by_index.keys()) == {0, 1, 2, 3}
    assert by_index[0]["ac_text"] == "file_exists:foo.py"
    assert by_index[1]["ac_text"] == "the tests pass"
    assert by_index[2]["ac_text"] == "exit_code:true"
    assert by_index[3]["ac_text"] == "code is readable"


@pytest.mark.asyncio
async def test_mixed_predicate_fail_short_circuits_no_verifier(store, runner, orchestrator, tmp_path):
    """If a predicate fails in a mixed job, we DON'T spawn the verifier for the
    prose - we go straight to retry/stuck like all-predicate. The whole job fails
    if any AC fails, so there's no point spending tokens on verifying prose."""
    job = _seed_running_job(
        store,
        orchestrator,
        runner,
        acceptance_criteria=[
            "file_exists:never-exists.txt",  # predicate, fails
            "the tests pass",  # prose, never verified
        ],
        worktree_path=str(tmp_path),
    )
    await orchestrator.on_session_complete(_worker_session(job), "claimed done")

    refreshed = store.get(job.id)
    # No verifier spawned.
    assert all(s.agent_file != "job_verifier" for s in runner.started), (
        "failed predicate must short-circuit BEFORE the verifier spawn"
    )
    # Retried (under max_attempts) or stuck.
    assert refreshed.state in (JobState.RUNNING.value, JobState.STUCK.value)
    assert refreshed.verify_attempts == 1


@pytest.mark.asyncio
async def test_predicate_ac_results_recorded_on_predicate_only_done(store, runner, orchestrator, tmp_path):
    """Predicate-only DONE path must record ac_results so the UI can render verdicts."""
    (tmp_path / "foo.py").write_text("# hi\n")
    job = _seed_running_job(
        store,
        orchestrator,
        runner,
        acceptance_criteria=["file_exists:foo.py"],
        worktree_path=str(tmp_path),
    )
    await orchestrator.on_session_complete(_worker_session(job), "done")
    final = store.get(job.id)
    assert final.state == JobState.DONE.value
    assert final.ac_results
    assert final.ac_results[0]["ac_index"] == 0
    assert final.ac_results[0]["pass"] is True
    # Result should also carry ac_results for parity with the verifier path.
    assert final.result is not None
    assert "ac_results" in final.result


def test_build_verifier_prompt_only_includes_prose_acs():
    """_build_verifier_prompt called with prose-only AC list emits only those ACs."""
    job = Job(
        id="j1",
        parent_session_id="p",
        prompt="x",
        acceptance_criteria=["file_exists:foo", "the tests pass", "exit_code:true"],
    )
    # Build with prose-only filter.
    prompt = _build_verifier_prompt(job, worker_output="done", prose_acs=["the tests pass"])
    assert "the tests pass" in prompt
    assert "file_exists:foo" not in prompt
    assert "exit_code:true" not in prompt


def test_build_verifier_prompt_backward_compat_without_prose_acs_arg():
    """Old call sites without the prose_acs kwarg include all acceptance_criteria."""
    job = Job(id="j1", parent_session_id="p", prompt="x", acceptance_criteria=["the tests pass", "no regressions"])
    prompt = _build_verifier_prompt(job, worker_output="done")
    assert "the tests pass" in prompt
    assert "no regressions" in prompt


# ── cwd resolution ──


def test_predicate_cwd_resolution_uses_worktree_then_repo(tmp_path):
    """The orchestrator's predicate eval picks cwd from worktree_path then repo,
    falling back to None. This is structural - tested via the helper directly."""
    from tsugite_daemon.jobs_orchestrator import _resolve_predicate_cwd

    job_with_worktree = Job(id="j", parent_session_id="p", prompt="x", worktree_path=str(tmp_path))
    assert _resolve_predicate_cwd(job_with_worktree) == str(tmp_path)

    job_with_repo = Job(id="j", parent_session_id="p", prompt="x", repo=str(tmp_path))
    assert _resolve_predicate_cwd(job_with_repo) == str(tmp_path)

    job_with_both = Job(
        id="j",
        parent_session_id="p",
        prompt="x",
        worktree_path=str(tmp_path),
        repo="/nonexistent/repo",
    )
    assert _resolve_predicate_cwd(job_with_both) == str(tmp_path), "worktree wins over repo"

    job_with_neither = Job(id="j", parent_session_id="p", prompt="x")
    assert _resolve_predicate_cwd(job_with_neither) is None
