"""Tests for JobsOrchestrator: worker-done → verifier → done | looping | stuck paths."""

import asyncio
import json
from unittest.mock import MagicMock

import pytest

from tsugite.daemon.job_store import Job, JobState, JobStore
from tsugite.daemon.jobs_orchestrator import (
    MAX_VERIFY_ATTEMPTS,
    JobsOrchestrator,
    _parse_verifier_output,
)
from tsugite.daemon.session_store import Session, SessionStatus


class FakeStore:
    """Minimal stand-in for SessionStore — records append_event + holds sessions for get_session lookups."""

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
        self._notify_callback = None
        self.started: list[Session] = []
        self.cancelled: list[str] = []
        self._next_session_id_counter = 0
        # Seed the default parent session so create_and_start_job's parent-agent
        # lookup has something to find.
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
    bus = MagicMock()
    return bus


@pytest.fixture
def orchestrator(store, runner, event_bus):
    return JobsOrchestrator(store, runner, event_bus=event_bus)


def _seed_running_job(store, orchestrator, runner, acceptance_criteria=None, agent=None):
    job = store.add(
        Job(
            id="",
            parent_session_id="parent-1",
            prompt="do the thing",
            acceptance_criteria=acceptance_criteria or [],
            agent=agent,
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


# ── verifier parser ──


@pytest.mark.parametrize(
    "raw,expected_pass",
    [
        ('{"ac_results": [], "overall_pass": true}', True),
        ('  {"ac_results": [], "overall_pass": false}  ', False),
    ],
)
def test_parse_verifier_output_handles_strict_json(raw, expected_pass):
    """The verifier agent is required to return structured JSON (response_format=json_object);
    the parser is a thin json.loads wrapper that tolerates leading/trailing whitespace only."""
    parsed = _parse_verifier_output(raw)
    assert parsed is not None
    assert parsed["overall_pass"] is expected_pass


def test_parse_verifier_output_returns_none_for_unparseable():
    assert _parse_verifier_output("no json here, just prose") is None
    assert _parse_verifier_output("") is None
    # Fences are no longer stripped — the verifier MUST return raw JSON.
    assert _parse_verifier_output('```json\n{"overall_pass": true}\n```') is None


# ── orchestrator flow ──


@pytest.mark.asyncio
async def test_worker_complete_with_no_ac_short_circuits_to_done(store, runner, orchestrator):
    job = _seed_running_job(store, orchestrator, runner, acceptance_criteria=[])
    await orchestrator.on_session_complete(_worker_session(job), "all done")
    assert store.get(job.id).state == JobState.DONE.value
    assert store.get(job.id).result == {"summary": "all done"}
    # No verifier spawned because no AC.
    assert all(s.agent != "job_verifier" for s in runner.started)


@pytest.mark.asyncio
async def test_worker_complete_with_ac_spawns_verifier(store, runner, orchestrator):
    job = _seed_running_job(store, orchestrator, runner, acceptance_criteria=["tests pass"])
    await orchestrator.on_session_complete(_worker_session(job), "## Summary\nworked\n## AC\n- tests pass: yes")
    refreshed = store.get(job.id)
    assert refreshed.state == JobState.VERIFYING.value
    verifier_spawns = [s for s in runner.started if (s.metadata or {}).get("verifier_for")]
    assert len(verifier_spawns) == 1
    assert verifier_spawns[0].metadata["verifier_for"] == "worker-1"
    assert verifier_spawns[0].agent_file == "job_verifier"
    assert "tests pass" in verifier_spawns[0].prompt


@pytest.mark.asyncio
async def test_verifier_overall_pass_marks_done(store, runner, orchestrator):
    job = _seed_running_job(store, orchestrator, runner, acceptance_criteria=["a", "b"])
    worker_output = "## Summary\nDid the thing.\n## Acceptance criteria\n- a: yes\n- b: yes"
    await orchestrator.on_session_complete(_worker_session(job), worker_output)
    job = store.get(job.id)
    verifier_json = json.dumps(
        {
            "ac_results": [
                {"ac_text": "a", "pass": True, "reason": "yes"},
                {"ac_text": "b", "pass": True, "reason": "yes"},
            ],
            "overall_pass": True,
        }
    )
    await orchestrator.on_session_complete(_verifier_session(job), verifier_json)
    final = store.get(job.id)
    assert final.state == JobState.DONE.value
    # The worker's actual output must land in job.result, not an empty string.
    assert final.result is not None
    assert worker_output in final.result["summary"]


@pytest.mark.asyncio
async def test_verifier_fail_under_max_attempts_loops(store, runner, orchestrator):
    job = _seed_running_job(store, orchestrator, runner, acceptance_criteria=["pr open"])
    await orchestrator.on_session_complete(_worker_session(job), "worker output")
    job = store.get(job.id)

    verifier_json = json.dumps(
        {"ac_results": [{"ac_text": "pr open", "pass": False, "reason": "no PR"}], "overall_pass": False}
    )
    await orchestrator.on_session_complete(_verifier_session(job, "verifier-1"), verifier_json)

    refreshed = store.get(job.id)
    assert refreshed.state == JobState.RUNNING.value
    assert refreshed.verify_attempts == 1
    # New worker session spawned (this is the loop-back). Filter by metadata
    # because verifier + loop-back may both fall back to the same adapter key.
    loop_workers = [s for s in runner.started if (s.metadata or {}).get("loop_attempt")]
    assert len(loop_workers) == 1
    assert "pr open" in loop_workers[0].prompt


@pytest.mark.asyncio
async def test_persistent_verifier_fail_marks_stuck_at_max_attempts(store, runner, orchestrator):
    job = _seed_running_job(store, orchestrator, runner, acceptance_criteria=["x"])

    def fail_round(round_idx: int):
        return json.dumps({"ac_results": [{"ac_text": "x", "pass": False, "reason": "nope"}], "overall_pass": False})

    # Round 1: worker done → verifier → loop
    await orchestrator.on_session_complete(_worker_session(job, "w1"), "w1")
    job = store.get(job.id)
    await orchestrator.on_session_complete(_verifier_session(job, "v1"), fail_round(1))
    # Round 2: worker done → verifier → loop
    job = store.get(job.id)
    await orchestrator.on_session_complete(_worker_session(job, job.worker_session_id), "w2")
    job = store.get(job.id)
    await orchestrator.on_session_complete(_verifier_session(job, "v2"), fail_round(2))
    # Round 3: worker done → verifier → STUCK
    job = store.get(job.id)
    await orchestrator.on_session_complete(_worker_session(job, job.worker_session_id), "w3")
    job = store.get(job.id)
    await orchestrator.on_session_complete(_verifier_session(job, "v3"), fail_round(3))

    final = store.get(job.id)
    assert final.state == JobState.STUCK.value
    assert final.verify_attempts == MAX_VERIFY_ATTEMPTS
    assert "x" in final.error


@pytest.mark.asyncio
async def test_malformed_verifier_output_treated_as_fail_and_loops(store, runner, orchestrator):
    job = _seed_running_job(store, orchestrator, runner, acceptance_criteria=["x"])
    await orchestrator.on_session_complete(_worker_session(job), "worker output")
    job = store.get(job.id)
    await orchestrator.on_session_complete(_verifier_session(job), "this is not json at all")
    refreshed = store.get(job.id)
    # First malformed verdict still permits one more attempt (under max).
    assert refreshed.state == JobState.RUNNING.value
    assert refreshed.verify_attempts == 1


@pytest.mark.asyncio
async def test_worker_failed_session_marks_job_errored(store, runner, orchestrator):
    job = _seed_running_job(store, orchestrator, runner, acceptance_criteria=["x"])
    failed = _worker_session(job, status=SessionStatus.FAILED.value)
    failed.error = "boom"
    await orchestrator.on_session_complete(failed, "")
    assert store.get(job.id).state == JobState.ERRORED.value
    assert "boom" in (store.get(job.id).error or "")


@pytest.mark.asyncio
async def test_verifier_session_uses_existing_adapter_when_job_verifier_not_registered(store, runner, orchestrator):
    """If no HTTP adapter named 'job_verifier' exists, the verifier spawn must
    fall back to a registered adapter (with agent_file=job_verifier) so it
    actually runs instead of failing silently with "no adapter for agent"."""
    # Runner has only a 'default' adapter — no 'job_verifier'.
    runner._adapters = {"default": object()}

    job = _seed_running_job(store, orchestrator, runner, acceptance_criteria=["x"])
    await orchestrator.on_session_complete(_worker_session(job), "worker output")

    verifier_spawns = [s for s in runner.started if (s.metadata or {}).get("verifier_for")]
    assert len(verifier_spawns) == 1
    v = verifier_spawns[0]
    assert v.agent == "default", f"verifier should fall back to a registered adapter, got agent={v.agent!r}"
    assert v.agent_file == "job_verifier", (
        f"verifier must still load job_verifier.md via agent_file, got agent_file={v.agent_file!r}"
    )


@pytest.mark.asyncio
async def test_loopback_worker_uses_existing_adapter_when_job_worker_not_registered(store, runner, orchestrator):
    """On verifier failure (< max), the loop-back worker must also fall back to
    a registered adapter rather than agent=job.agent directly (which is the
    agent_file name, not the adapter key)."""
    runner._adapters = {"default": object()}

    job = _seed_running_job(store, orchestrator, runner, acceptance_criteria=["x"], agent="job_worker")
    await orchestrator.on_session_complete(_worker_session(job), "w1")
    job = store.get(job.id)
    fail_json = json.dumps({"ac_results": [{"ac_text": "x", "pass": False, "reason": "no"}], "overall_pass": False})
    await orchestrator.on_session_complete(_verifier_session(job, "v1"), fail_json)

    loopback_workers = [s for s in runner.started if (s.metadata or {}).get("loop_attempt")]
    assert len(loopback_workers) == 1
    w = loopback_workers[0]
    assert w.agent == "default", f"loop-back worker should fall back to a registered adapter, got agent={w.agent!r}"
    assert w.agent_file == "job_worker", (
        f"loop-back worker must still load job_worker.md via agent_file, got agent_file={w.agent_file!r}"
    )


@pytest.mark.asyncio
async def test_session_without_job_id_is_ignored(store, runner, orchestrator):
    unrelated = Session(id="x", agent="default", source="spawned", status="completed", metadata={})
    # Should not raise, should not touch the store.
    await orchestrator.on_session_complete(unrelated, "ok")


def test_on_timeout_transitions_running_job_to_stuck(store, runner, orchestrator):
    """Direct call to _on_timeout while the Job is in RUNNING must mark it STUCK.

    Regression: the state machine previously rejected running→stuck, so the
    transition raised inside _finalize and the Job stayed running forever — the
    worker was cancelled but the tile never updated.
    """
    job = store.add(Job(id="", parent_session_id="parent-1", prompt="long task"))
    orchestrator.register_worker(job.id, "worker-1", timeout_minutes=30)
    assert store.get(job.id).state == JobState.RUNNING.value

    orchestrator._on_timeout(job.id)

    final = store.get(job.id)
    assert final.state == JobState.STUCK.value, "timeout from RUNNING must transition the Job to STUCK"
    assert "timeout" in (final.error or "")
    # Worker session should have been cancelled.
    assert "worker-1" in runner.cancelled


def test_register_worker_emits_tile_event(store, runner, orchestrator, event_bus):
    job = store.add(Job(id="", parent_session_id="parent-1", prompt="do thing"))
    orchestrator.register_worker(job.id, "worker-1", timeout_minutes=30)
    # event_bus.emit was called with "job_update"
    emitted = [call.args for call in event_bus.emit.call_args_list if call.args[0] == "job_update"]
    assert emitted, "register_worker should emit a job_update event"
    payload = emitted[-1][1]
    assert payload["job_id"] == job.id
    assert payload["state"] == JobState.RUNNING.value
    # And it should persist the job_status into the parent's JSONL
    persisted = [e for sid, e in runner.store.events if sid == "parent-1" and e.get("type") == "job_status"]
    assert persisted, "register_worker should persist a job_status event into parent JSONL"


def test_attach_chains_existing_callback(store, runner, orchestrator):
    calls = []

    async def existing(session, result):
        calls.append((session.id, result))

    runner._notify_callback = existing
    orchestrator.attach()
    # Construct a benign session that has no job_id (orchestrator ignores it),
    # then run the chained callback synchronously via the event loop.
    import asyncio

    session = Session(id="x", agent="default", source="spawned", status="completed", metadata={})
    asyncio.run(runner._notify_callback(session, "result"))
    assert calls == [("x", "result")]


# ── critical regressions from code-review pass ──


@pytest.mark.asyncio
async def test_verifier_returning_string_ac_results_does_not_crash(store, runner, orchestrator):
    """Verifier may return ac_results entries that aren't dicts (e.g. just strings).
    The orchestrator must not raise — treat as a verifier failure with a synthetic ac entry."""
    job = _seed_running_job(store, orchestrator, runner, acceptance_criteria=["x"])
    await orchestrator.on_session_complete(_worker_session(job), "w1")
    job = store.get(job.id)
    bad = json.dumps(
        {"ac_results": ["tests pass", {"ac_text": "x", "pass": False, "reason": "no"}], "overall_pass": False}
    )
    # Should not raise AttributeError.
    await orchestrator.on_session_complete(_verifier_session(job), bad)
    refreshed = store.get(job.id)
    # Treated as a failure → loop or stuck, not silently hung in VERIFYING.
    assert refreshed.state in (JobState.RUNNING.value, JobState.STUCK.value)


@pytest.mark.asyncio
async def test_verifier_session_failed_does_not_burn_attempts(store, runner, orchestrator):
    """When the verifier session itself FAILS (adapter crash), the Job must be marked
    ERRORED with the verifier-spawn-failure reason, NOT treated as a failed verdict
    that loops the worker and counts toward MAX_VERIFY_ATTEMPTS."""
    job = _seed_running_job(store, orchestrator, runner, acceptance_criteria=["x"])
    await orchestrator.on_session_complete(_worker_session(job), "w1")
    job = store.get(job.id)
    failed_verifier = _verifier_session(job, status=SessionStatus.FAILED.value)
    failed_verifier.error = "verifier model API down"
    await orchestrator.on_session_complete(failed_verifier, "FAILED: verifier model API down")
    refreshed = store.get(job.id)
    assert refreshed.state == JobState.ERRORED.value, (
        f"verifier infra failure must mark Job ERRORED, got {refreshed.state}"
    )
    assert "verifier" in (refreshed.error or "").lower()
    assert refreshed.verify_attempts == 0, "infra failure must not consume verify attempts"


@pytest.mark.asyncio
async def test_verifying_window_has_a_timer(store, runner, orchestrator):
    """A hung verifier must not strand the Job in VERIFYING forever. After the
    transition to VERIFYING, a timer must be armed so _on_timeout can rescue."""
    job = _seed_running_job(store, orchestrator, runner, acceptance_criteria=["x"])
    await orchestrator.on_session_complete(_worker_session(job), "w1")
    assert store.get(job.id).state == JobState.VERIFYING.value
    assert job.id in orchestrator._timeout_handles, (
        "VERIFYING transition must arm a timer so verifier hangs can be reaped"
    )


@pytest.mark.asyncio
async def test_create_and_start_job_routes_through_parent_agent(store, runner, orchestrator):
    """The worker must run through the parent session's adapter, not the first one
    in dict order — otherwise jobs leak across agents (credentials, tools, model)."""
    runner._adapters = {"coder": object(), "support": object()}
    # Parent session lives in `support`.
    runner.store.sessions["parent-A"] = Session(id="parent-A", agent="support")
    job, started = orchestrator.create_and_start_job(
        parent_session_id="parent-A",
        prompt="hello",
        acceptance_criteria=[],
        timeout_minutes=30,
        spawned_by="user-slash",
    )
    assert started.agent == "support", f"worker must route through parent's adapter 'support', got {started.agent!r}"


def test_on_timeout_skips_cancel_when_worker_already_terminal(store, runner, orchestrator):
    """If the worker has already completed/failed by the time the timeout handler
    fires, _on_timeout must NOT call cancel_session — otherwise it overwrites the
    real terminal status with CANCELLED and corrupts the audit trail."""
    job = store.add(Job(id="", parent_session_id="parent-1", prompt="x"))
    orchestrator.register_worker(job.id, "worker-1", timeout_minutes=30)
    # Pretend the worker already finished successfully.
    runner.store.sessions["worker-1"] = Session(id="worker-1", agent="default", status=SessionStatus.COMPLETED.value)
    orchestrator._on_timeout(job.id)
    assert "worker-1" not in runner.cancelled, "_on_timeout must not cancel a worker that already finished"


@pytest.mark.asyncio
async def test_cancelled_worker_marks_job_cancelled_not_errored(store, runner, orchestrator):
    """User-cancelled worker (status=CANCELLED) should set Job.state=CANCELLED,
    not ERRORED. ERRORED implies something went wrong; CANCELLED reflects intent."""
    job = _seed_running_job(store, orchestrator, runner, acceptance_criteria=["x"])
    cancelled = _worker_session(job, status=SessionStatus.CANCELLED.value)
    await orchestrator.on_session_complete(cancelled, "")
    assert store.get(job.id).state == JobState.CANCELLED.value


def test_create_and_start_job_marks_job_errored_when_spawn_fails(store, runner, orchestrator):
    """If start_session raises, the persisted Job must be moved out of QUEUED so it
    doesn't accumulate as a zombie in jobs.json."""

    def boom(session):
        raise RuntimeError("adapter not registered")

    runner.start_session = boom
    with pytest.raises(RuntimeError):
        orchestrator.create_and_start_job(
            parent_session_id="parent-1",
            prompt="hello",
            acceptance_criteria=[],
            timeout_minutes=30,
            spawned_by="user-slash",
        )
    # The Job exists; it must NOT be left in QUEUED.
    jobs = [j for j in store._jobs.values() if j.parent_session_id == "parent-1"]
    assert len(jobs) == 1
    assert jobs[0].state == JobState.ERRORED.value, (
        f"spawn failure must transition Job out of QUEUED, got {jobs[0].state}"
    )
    assert "adapter not registered" in (jobs[0].error or "")


@pytest.mark.asyncio
async def test_late_worker_complete_for_cancelled_job_is_ignored(store, runner, orchestrator, event_bus):
    """If the Job has already been advanced past RUNNING (e.g. user cancelled), a
    late worker-complete notify must NOT overwrite the terminal state's audit with
    a contradictory worker summary. The cancellation wins."""
    job = _seed_running_job(store, orchestrator, runner, acceptance_criteria=["x"])
    # Simulate the user cancelling between worker dispatch and worker completion.
    store.update_state(job.id, JobState.CANCELLED.value)
    event_bus.reset_mock()
    await orchestrator.on_session_complete(_worker_session(job), "worker output here")
    # The late result must NOT have been persisted; cancellation's audit is preserved.
    assert store.get(job.id).state == JobState.CANCELLED.value
    assert store.get(job.id).result is None, "worker output must not overwrite a cancelled job's record"


def test_verifier_agent_resolves_to_hermetic_config():
    """job_verifier.md must be reasoning-blind: no parent inheritance, no personal
    attachments, no auto-loaded skills, minimal tools, declared max_turns, and
    JSON-mode model_kwargs surviving the resolve step."""
    from pathlib import Path

    from tsugite.md_agents import parse_agent_file

    verifier_path = Path(__file__).parent.parent.parent / "tsugite" / "builtin_agents" / "job_verifier.md"
    agent = parse_agent_file(verifier_path)
    assert agent.config.extends == "none", "verifier must declare `extends: none` to stay reasoning-blind"
    assert set(agent.config.tools) == {"read_file", "run"}, f"verifier tools leaked: {agent.config.tools}"
    assert agent.config.attachments == [], f"verifier attachments leaked: {agent.config.attachments}"
    assert agent.config.auto_load_skills == [], f"verifier skills leaked: {agent.config.auto_load_skills}"
    assert agent.config.max_turns == 5, f"verifier max_turns clobbered: {agent.config.max_turns}"
    assert agent.config.model_kwargs.get("response_format") == {"type": "json_object"}, (
        f"verifier must request structured JSON output, got {agent.config.model_kwargs!r}"
    )


# ── code-review round 2: critical fixes ──


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("42", None),
        ("null", None),
        ("true", None),
        ("[1,2,3]", None),
        ('"oops"', None),
    ],
)
def test_parse_verifier_output_rejects_non_dict_json(raw, expected):
    """Bare-value JSON must be rejected — `parsed.get('overall_pass')` would crash on int/list/str/bool."""
    assert _parse_verifier_output(raw) is expected


@pytest.mark.asyncio
async def test_string_false_overall_pass_does_not_mark_done(store, runner, orchestrator):
    """Verifier returning {\"overall_pass\": \"false\"} (string) must NOT silently flip the Job to DONE.
    Strict-bool check required."""
    job = _seed_running_job(store, orchestrator, runner, acceptance_criteria=["x"])
    await orchestrator.on_session_complete(_worker_session(job), "w1")
    job = store.get(job.id)
    bad = json.dumps({"overall_pass": "false", "ac_results": [{"ac_text": "x", "pass": False, "reason": "n"}]})
    await orchestrator.on_session_complete(_verifier_session(job), bad)
    assert store.get(job.id).state != JobState.DONE.value, (
        "string 'false' for overall_pass must not pass; must be strict bool True"
    )


@pytest.mark.asyncio
async def test_verifier_session_id_persisted_and_cancelled_on_timeout(store, runner, orchestrator):
    """A hung verifier must be cancellable from _on_timeout."""
    job = _seed_running_job(store, orchestrator, runner, acceptance_criteria=["x"])
    await orchestrator.on_session_complete(_worker_session(job), "w1")
    refreshed = store.get(job.id)
    assert refreshed.verifier_session_id, "verifier session id must be stored on Job so timeout can cancel it"
    # Pretend the worker is already done (so its cancel is skipped) and verifier is hung.
    runner.store.sessions[refreshed.worker_session_id] = Session(
        id=refreshed.worker_session_id, agent="default", status=SessionStatus.COMPLETED.value
    )
    runner.store.sessions[refreshed.verifier_session_id] = Session(
        id=refreshed.verifier_session_id, agent="default", status=SessionStatus.RUNNING.value
    )
    orchestrator._on_timeout(job.id)
    assert refreshed.verifier_session_id in runner.cancelled, (
        "timeout must cancel the verifier session when it's still running"
    )


def test_attach_is_idempotent(store, runner, orchestrator):
    """Calling attach() twice on the same runner must not double-wrap callbacks."""
    orchestrator.attach()
    first = runner._notify_callback
    orchestrator.attach()
    second = runner._notify_callback
    assert first is second, "second attach() must short-circuit, not re-wrap"


@pytest.mark.asyncio
async def test_persistent_verifier_fail_stuck_state_preserves_ac_results(store, runner, orchestrator):
    """STUCK terminal state must persist the verifier's per-AC verdicts on job.result
    so the UI can render structured failure data, not just the error string."""
    job = _seed_running_job(store, orchestrator, runner, acceptance_criteria=["x"])
    fail = json.dumps({"ac_results": [{"ac_text": "x", "pass": False, "reason": "nope"}], "overall_pass": False})
    for i in range(MAX_VERIFY_ATTEMPTS):
        await orchestrator.on_session_complete(_worker_session(job, f"w{i}"), f"w{i}")
        job = store.get(job.id)
        await orchestrator.on_session_complete(_verifier_session(job, f"v{i}"), fail)
        job = store.get(job.id)
    assert job.state == JobState.STUCK.value
    assert job.result is not None and job.result.get("ac_results"), (
        "STUCK Job must preserve the last verifier's ac_results for the UI"
    )


@pytest.mark.asyncio
async def test_verifier_complete_for_non_verifying_job_is_skipped(store, runner, orchestrator):
    """If a duplicate verifier-complete fires after Job has already advanced past VERIFYING,
    the handler must NOT spawn a second concurrent worker."""
    job = _seed_running_job(store, orchestrator, runner, acceptance_criteria=["x"])
    await orchestrator.on_session_complete(_worker_session(job), "w1")
    job = store.get(job.id)
    fail = json.dumps({"ac_results": [{"ac_text": "x", "pass": False, "reason": "n"}], "overall_pass": False})
    # First verifier-complete advances Job back to RUNNING (retry path).
    await orchestrator.on_session_complete(_verifier_session(job, "v1"), fail)
    runner_started_count_after_first = len([s for s in runner.started if (s.metadata or {}).get("loop_attempt")])
    # Duplicate verifier-complete for the same verifier session — Job is now RUNNING.
    await orchestrator.on_session_complete(_verifier_session(job, "v1"), fail)
    runner_started_count_after_duplicate = len([s for s in runner.started if (s.metadata or {}).get("loop_attempt")])
    assert runner_started_count_after_first == runner_started_count_after_duplicate, (
        "duplicate verifier-complete on a non-VERIFYING Job must NOT spawn a second worker"
    )


@pytest.mark.asyncio
async def test_loop_attempt_metadata_matches_verify_attempts(store, runner, orchestrator):
    """loop_attempt should reflect the actual attempt number (1-indexed retry), not double-counted."""
    job = _seed_running_job(store, orchestrator, runner, acceptance_criteria=["x"])
    await orchestrator.on_session_complete(_worker_session(job, "w0"), "w0")
    job = store.get(job.id)
    fail = json.dumps({"ac_results": [{"ac_text": "x", "pass": False, "reason": "n"}], "overall_pass": False})
    await orchestrator.on_session_complete(_verifier_session(job, "v0"), fail)
    job = store.get(job.id)
    loop_workers = [s for s in runner.started if (s.metadata or {}).get("loop_attempt")]
    assert loop_workers, "loop-back worker must spawn"
    # After 1st failed verify: verify_attempts=1 in store, loop_attempt should be 1
    # (this is retry #1), not 2.
    assert loop_workers[0].metadata["loop_attempt"] == job.verify_attempts, (
        f"loop_attempt {loop_workers[0].metadata['loop_attempt']} should match verify_attempts {job.verify_attempts}"
    )


@pytest.mark.asyncio
async def test_empty_ac_results_with_overall_pass_false_synthesises_failure_reason(store, runner, orchestrator):
    """If verifier returns overall_pass=false with empty ac_results, the retry prompt and
    STUCK error must still carry actionable signal — not a headerless followup."""
    job = _seed_running_job(store, orchestrator, runner, acceptance_criteria=["x"])
    await orchestrator.on_session_complete(_worker_session(job), "w1")
    job = store.get(job.id)
    bad = json.dumps({"overall_pass": False, "ac_results": []})
    await orchestrator.on_session_complete(_verifier_session(job), bad)
    # Retry worker should have been spawned with a non-empty prompt that mentions
    # something specific, not just the empty header.
    loop_workers = [s for s in runner.started if (s.metadata or {}).get("loop_attempt")]
    assert loop_workers
    prompt = loop_workers[0].prompt
    assert "did not list" in prompt or "no failed criteria" in prompt or "review the AC list" in prompt, (
        f"followup prompt must include a synthetic failure context when ac_results is empty; got {prompt[:200]!r}"
    )


# ── Theme A: worktree provisioning ──


def _make_git_repo(path):
    import subprocess

    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=path, check=True)
    (path / "README.md").write_text("hi\n")
    subprocess.run(["git", "add", "."], cwd=path, check=True)
    subprocess.run(["git", "commit", "-qm", "init"], cwd=path, check=True)


def test_create_job_with_repo_provisions_worktree(store, runner, orchestrator, tmp_path):
    repo = tmp_path / "repo"
    _make_git_repo(repo)
    job, started = orchestrator.create_and_start_job(
        parent_session_id="parent-1",
        prompt="do",
        acceptance_criteria=[],
        repo=str(repo),
    )
    fresh = store.get(job.id)
    assert fresh.worktree_path is not None, "worktree_path must be set when --repo is given"
    from pathlib import Path

    wt = Path(fresh.worktree_path)
    assert wt.exists() and (wt / ".git").exists(), f"worktree must exist on disk at {wt}"
    assert (wt / "README.md").exists(), "worktree must contain the repo's files"
    # Worker's session must point at the worktree as its workspace.
    assert started.workspace_override == fresh.worktree_path


def test_create_job_without_repo_does_not_provision(store, runner, orchestrator):
    job, started = orchestrator.create_and_start_job(
        parent_session_id="parent-1",
        prompt="do",
        acceptance_criteria=[],
        repo=None,
    )
    assert store.get(job.id).worktree_path is None
    assert started.workspace_override is None


def test_worktree_pruned_on_done(store, runner, orchestrator, tmp_path):
    repo = tmp_path / "repo"
    _make_git_repo(repo)
    job, _ = orchestrator.create_and_start_job(
        parent_session_id="parent-1",
        prompt="do",
        acceptance_criteria=[],
        repo=str(repo),
    )
    wt = store.get(job.id).worktree_path
    from pathlib import Path

    assert Path(wt).exists()
    import asyncio

    asyncio.run(orchestrator.on_session_complete(_worker_session(store.get(job.id)), "done"))
    # No AC → short-circuit DONE → worktree pruned.
    assert not Path(wt).exists(), f"worktree at {wt} must be pruned after DONE"


def test_worktree_kept_on_stuck_for_inspection(store, runner, orchestrator, tmp_path):
    repo = tmp_path / "repo"
    _make_git_repo(repo)
    job, _ = orchestrator.create_and_start_job(
        parent_session_id="parent-1",
        prompt="do",
        acceptance_criteria=["x"],
        repo=str(repo),
    )
    wt = store.get(job.id).worktree_path
    from pathlib import Path

    fail = json.dumps({"ac_results": [{"ac_text": "x", "pass": False, "reason": "no"}], "overall_pass": False})
    import asyncio

    for i in range(MAX_VERIFY_ATTEMPTS):
        asyncio.run(orchestrator.on_session_complete(_worker_session(store.get(job.id), f"w{i}"), f"w{i}"))
        asyncio.run(orchestrator.on_session_complete(_verifier_session(store.get(job.id), f"v{i}"), fail))
    assert store.get(job.id).state == JobState.STUCK.value
    assert Path(wt).exists(), "worktree must be kept on STUCK for inspection"


# ── Theme B: tile actions ──


@pytest.mark.asyncio
async def test_cancel_job_finalizes_cancelled_and_cancels_worker(store, runner, orchestrator):
    job = _seed_running_job(store, orchestrator, runner, acceptance_criteria=[])
    runner.store.sessions["worker-1"] = Session(id="worker-1", agent="default", status=SessionStatus.RUNNING.value)
    await orchestrator.cancel_job(job.id, reason="user clicked cancel")
    assert store.get(job.id).state == JobState.CANCELLED.value
    assert "worker-1" in runner.cancelled
    assert "user clicked cancel" in (store.get(job.id).error or "")


@pytest.mark.asyncio
async def test_cancel_job_no_op_on_terminal(store, runner, orchestrator):
    job = _seed_running_job(store, orchestrator, runner, acceptance_criteria=[])
    await orchestrator.on_session_complete(_worker_session(job), "done")  # → DONE
    assert store.get(job.id).state == JobState.DONE.value
    await orchestrator.cancel_job(job.id, reason="too late")
    # Still DONE; cancel must not flip terminal state.
    assert store.get(job.id).state == JobState.DONE.value


@pytest.mark.asyncio
async def test_mark_done_manual_from_stuck(store, runner, orchestrator):
    job = _seed_running_job(store, orchestrator, runner, acceptance_criteria=["x"])
    fail = json.dumps({"ac_results": [{"ac_text": "x", "pass": False, "reason": "n"}], "overall_pass": False})
    for i in range(MAX_VERIFY_ATTEMPTS):
        await orchestrator.on_session_complete(_worker_session(store.get(job.id), f"w{i}"), f"w{i}")
        await orchestrator.on_session_complete(_verifier_session(store.get(job.id), f"v{i}"), fail)
    assert store.get(job.id).state == JobState.STUCK.value
    await orchestrator.mark_done_manual(job.id, reason="verifier was wrong, looks fine to me")
    final = store.get(job.id)
    assert final.state == JobState.DONE.value
    assert "verifier was wrong" in (final.result.get("manual_done_reason") or "")


@pytest.mark.asyncio
async def test_mark_done_manual_rejects_non_stuck(store, runner, orchestrator):
    job = _seed_running_job(store, orchestrator, runner, acceptance_criteria=[])
    with pytest.raises(ValueError, match="only.*stuck"):
        await orchestrator.mark_done_manual(job.id, reason="x")


@pytest.mark.asyncio
async def test_retry_with_hint_spawns_worker_does_not_reset_counter(store, runner, orchestrator):
    job = _seed_running_job(store, orchestrator, runner, acceptance_criteria=["x"])
    fail = json.dumps({"ac_results": [{"ac_text": "x", "pass": False, "reason": "n"}], "overall_pass": False})
    for i in range(MAX_VERIFY_ATTEMPTS):
        await orchestrator.on_session_complete(_worker_session(store.get(job.id), f"w{i}"), f"w{i}")
        await orchestrator.on_session_complete(_verifier_session(store.get(job.id), f"v{i}"), fail)
    assert store.get(job.id).state == JobState.STUCK.value
    pre = store.get(job.id).verify_attempts

    await orchestrator.retry_with_hint(job.id, hint="the actual problem is X, try Y")
    refreshed = store.get(job.id)
    assert refreshed.state == JobState.RUNNING.value
    # Counter MUST NOT reset (avoid infinite loops).
    assert refreshed.verify_attempts == pre
    # New worker must have the hint in its prompt.
    hint_workers = [s for s in runner.started if "the actual problem is X" in (s.prompt or "")]
    assert hint_workers, "retry_with_hint must spawn a new worker carrying the user's hint"


@pytest.mark.asyncio
async def test_retry_with_hint_rejects_non_stuck(store, runner, orchestrator):
    job = _seed_running_job(store, orchestrator, runner, acceptance_criteria=[])
    with pytest.raises(ValueError, match="only.*stuck"):
        await orchestrator.retry_with_hint(job.id, hint="x")


# ── notify + context injection + new tools ──


def test_render_jobs_context_xml_empty_for_no_jobs(store):
    from tsugite.daemon.jobs_orchestrator import render_jobs_context_xml

    assert render_jobs_context_xml(store, "parent-X") == ""


def test_render_jobs_context_xml_lists_active_and_recent(store, runner, orchestrator, tmp_path):
    from tsugite.daemon.jobs_orchestrator import render_jobs_context_xml

    # Active job
    active = _seed_running_job(store, orchestrator, runner, acceptance_criteria=[])

    # A recently-done job
    done_job = store.add(Job(id="", parent_session_id="parent-1", prompt="something done", acceptance_criteria=[]))
    store.update_state(done_job.id, JobState.RUNNING.value)
    store.update_state(done_job.id, JobState.VERIFYING.value)
    store.update_state(done_job.id, JobState.DONE.value)

    # Job for a DIFFERENT parent must NOT appear.
    runner.store.sessions["other-parent"] = Session(id="other-parent", agent="default")
    other = store.add(Job(id="", parent_session_id="other-parent", prompt="not mine"))

    xml = render_jobs_context_xml(store, "parent-1")
    assert "<active>" in xml
    assert active.id in xml
    assert "<recent>" in xml
    assert done_job.id in xml
    assert other.id not in xml, "context must not bleed jobs from other sessions"
    assert "something done" in xml


def test_render_jobs_context_xml_truncates_prompt_and_error(store):
    long = "x" * 200
    job = store.add(Job(id="", parent_session_id="parent-T", prompt=long))
    store.update_state(job.id, JobState.RUNNING.value)
    store.update_state(job.id, JobState.VERIFYING.value)
    store.update_state(job.id, JobState.STUCK.value)
    store.update(job.id, error=("y" * 500))

    from tsugite.daemon.jobs_orchestrator import render_jobs_context_xml

    xml = render_jobs_context_xml(store, "parent-T")
    # Prompt truncated to 80 + ellipsis
    assert "xxxxxxxx" in xml
    assert "x" * 200 not in xml
    # Error first-line truncated to 200
    assert "y" * 500 not in xml
    assert "y" * 200 in xml


@pytest.mark.asyncio
async def test_notify_true_fires_reply_to_session_on_terminal(store, runner, orchestrator):
    """If job.notify=True, the orchestrator should schedule a reply_to_session
    on terminal transition so the parent agent wakes up."""
    sent = []

    async def fake_reply(session_id, message, source="session", metadata=None):
        sent.append((session_id, message, source, metadata))
        return "ok"

    runner.reply_to_session = fake_reply

    job = store.add(Job(id="", parent_session_id="parent-1", prompt="do thing", acceptance_criteria=[], notify=True))
    orchestrator.register_worker(job.id, "worker-1", timeout_minutes=30)
    await orchestrator.on_session_complete(_worker_session(job), "all done")
    # Let the scheduled task run
    await asyncio.sleep(0)
    assert sent, "notify=True must trigger reply_to_session on terminal"
    assert "Job " + job.id in sent[0][1]
    assert "get_job" in sent[0][1]


@pytest.mark.asyncio
async def test_notify_false_does_not_fire_reply(store, runner, orchestrator):
    sent = []

    async def fake_reply(session_id, message, source="session", metadata=None):
        sent.append((session_id, message))
        return "ok"

    runner.reply_to_session = fake_reply

    job = store.add(Job(id="", parent_session_id="parent-1", prompt="do thing", acceptance_criteria=[], notify=False))
    orchestrator.register_worker(job.id, "worker-1", timeout_minutes=30)
    await orchestrator.on_session_complete(_worker_session(job), "all done")
    await asyncio.sleep(0)
    assert sent == [], "notify=False must NOT trigger reply_to_session"


def test_build_notify_message_includes_job_id_and_state():
    from tsugite.daemon.jobs_orchestrator import _build_notify_message

    job = Job(
        id="job-abc",
        parent_session_id="p",
        prompt="implement the foo endpoint" + "x" * 100,
        state=JobState.STUCK.value,
        error="Verifier failed: AC1 not met\nmore detail",
    )
    msg = _build_notify_message(job)
    assert "job-abc" in msg
    assert "stuck" in msg
    assert "get_job('job-abc')" in msg
    assert "AC1 not met" in msg
    # Prompt truncated
    assert "x" * 100 not in msg


# ── attempt history tracking ──


def test_create_and_start_job_records_initial_attempt(store, runner, orchestrator):
    job, started = orchestrator.create_and_start_job(parent_session_id="parent-1", prompt="hi", acceptance_criteria=[])
    fresh = store.get(job.id)
    assert len(fresh.attempts) == 1
    assert fresh.attempts[0]["kind"] == "initial"
    assert fresh.attempts[0]["worker_session_id"] == started.id
    assert fresh.attempts[0]["verifier_session_id"] is None
    assert fresh.attempts[0]["verifier_pass"] is None


@pytest.mark.asyncio
async def test_retry_loop_appends_attempts_with_verdicts(store, runner, orchestrator):
    job = _seed_running_job(store, orchestrator, runner, acceptance_criteria=["x"])
    # First worker → verifier → fail → retry; assert attempt history reflects each step.
    await orchestrator.on_session_complete(_worker_session(job, "w1"), "out1")
    job = store.get(job.id)
    assert job.attempts[-1]["verifier_session_id"] is not None
    fail = json.dumps({"ac_results": [{"ac_text": "x", "pass": False, "reason": "n"}], "overall_pass": False})
    await orchestrator.on_session_complete(_verifier_session(job, "v1"), fail)
    job = store.get(job.id)
    # Now there should be 2 attempts; first one has verdict=False, second is fresh.
    assert len(job.attempts) >= 2
    assert job.attempts[0]["verifier_pass"] is False
    assert job.attempts[-1]["kind"] == "retry"
    assert job.attempts[-1]["verifier_pass"] is None


@pytest.mark.asyncio
async def test_retry_with_hint_appends_hint_attempt(store, runner, orchestrator):
    job = _seed_running_job(store, orchestrator, runner, acceptance_criteria=["x"])
    fail = json.dumps({"ac_results": [{"ac_text": "x", "pass": False, "reason": "n"}], "overall_pass": False})
    for i in range(MAX_VERIFY_ATTEMPTS):
        await orchestrator.on_session_complete(_worker_session(store.get(job.id), f"w{i}"), f"w{i}")
        await orchestrator.on_session_complete(_verifier_session(store.get(job.id), f"v{i}"), fail)
    assert store.get(job.id).state == JobState.STUCK.value
    pre_count = len(store.get(job.id).attempts)
    await orchestrator.retry_with_hint(job.id, hint="the real issue is X")
    fresh = store.get(job.id)
    assert len(fresh.attempts) == pre_count + 1
    assert fresh.attempts[-1]["kind"] == "hint"


# ── Gap 1: AC kind metadata ──


def test_emit_job_event_normalizes_ac_to_dicts(store, runner, orchestrator):
    """Legacy plain-string AC must still round-trip into the tile event payload
    as dicts, so the frontend gets a single shape regardless of disk format."""
    job = store.add(
        Job(id="", parent_session_id="parent-1", prompt="x", acceptance_criteria=["plain text", "another"])
    )
    orchestrator._emit_job_event(store.get(job.id))
    emitted = [call.args[1] for call in orchestrator._event_bus.emit.call_args_list if call.args[0] == "job_update"]
    assert emitted, "expected a job_update emit"
    payload = emitted[-1]
    assert payload["acceptance_criteria"] == [
        {"text": "plain text", "kind": "llm"},
        {"text": "another", "kind": "llm"},
    ]


def test_emit_job_event_preserves_explicit_ac_kind(store, runner, orchestrator):
    job = store.add(
        Job(
            id="",
            parent_session_id="parent-1",
            prompt="x",
            acceptance_criteria=[{"text": "endpoint returns 200", "kind": "cmd"}],
        )
    )
    orchestrator._emit_job_event(store.get(job.id))
    emitted = [call.args[1] for call in orchestrator._event_bus.emit.call_args_list if call.args[0] == "job_update"]
    payload = emitted[-1]
    assert payload["acceptance_criteria"] == [{"text": "endpoint returns 200", "kind": "cmd"}]


# ── Gap 2: retry_with_hint reset_counter + fresh_workspace ──


@pytest.mark.asyncio
async def test_retry_with_hint_reset_counter_zeroes_verify_attempts(store, runner, orchestrator):
    """When reset_counter=True, verify_attempts must be zeroed so the retry gets
    a full budget of verifier rounds again."""
    job = _seed_running_job(store, orchestrator, runner, acceptance_criteria=["x"])
    fail = json.dumps({"ac_results": [{"ac_text": "x", "pass": False, "reason": "n"}], "overall_pass": False})
    for i in range(MAX_VERIFY_ATTEMPTS):
        await orchestrator.on_session_complete(_worker_session(store.get(job.id), f"w{i}"), f"w{i}")
        await orchestrator.on_session_complete(_verifier_session(store.get(job.id), f"v{i}"), fail)
    assert store.get(job.id).state == JobState.STUCK.value
    assert store.get(job.id).verify_attempts == MAX_VERIFY_ATTEMPTS

    await orchestrator.retry_with_hint(job.id, hint="try Y", reset_counter=True)
    refreshed = store.get(job.id)
    assert refreshed.verify_attempts == 0, "reset_counter=True must zero verify_attempts"
    assert refreshed.state == JobState.RUNNING.value


@pytest.mark.asyncio
async def test_retry_with_hint_fresh_workspace_no_op_when_no_repo(store, runner, orchestrator):
    """A Job with no `repo` (and no worktree) must treat fresh_workspace=True as
    a no-op rather than crashing."""
    job = _seed_running_job(store, orchestrator, runner, acceptance_criteria=["x"])
    fail = json.dumps({"ac_results": [{"ac_text": "x", "pass": False, "reason": "n"}], "overall_pass": False})
    for i in range(MAX_VERIFY_ATTEMPTS):
        await orchestrator.on_session_complete(_worker_session(store.get(job.id), f"w{i}"), f"w{i}")
        await orchestrator.on_session_complete(_verifier_session(store.get(job.id), f"v{i}"), fail)
    assert store.get(job.id).state == JobState.STUCK.value
    # Should not raise.
    await orchestrator.retry_with_hint(job.id, hint="x", fresh_workspace=True)
    assert store.get(job.id).state == JobState.RUNNING.value


# ── Gap 3: max_attempts + notify_when ──


@pytest.mark.asyncio
async def test_max_attempts_respected_by_verifier_loop(store, runner, orchestrator):
    """A Job with max_attempts=2 must transition to STUCK after 2 failed verifier
    rounds, even though the module default is 3."""
    job = store.add(
        Job(
            id="",
            parent_session_id="parent-1",
            prompt="x",
            acceptance_criteria=["x"],
            max_attempts=2,
        )
    )
    orchestrator.register_worker(job.id, "w0", timeout_minutes=30)
    fail = json.dumps({"ac_results": [{"ac_text": "x", "pass": False, "reason": "n"}], "overall_pass": False})
    # Round 0: worker → verifier → loop (still under cap of 2).
    await orchestrator.on_session_complete(_worker_session(store.get(job.id), "w0"), "w0")
    await orchestrator.on_session_complete(_verifier_session(store.get(job.id), "v0"), fail)
    assert store.get(job.id).state == JobState.RUNNING.value
    # Round 1: worker → verifier → STUCK (hit cap of 2).
    await orchestrator.on_session_complete(_worker_session(store.get(job.id), "w1"), "w1")
    await orchestrator.on_session_complete(_verifier_session(store.get(job.id), "v1"), fail)
    assert store.get(job.id).state == JobState.STUCK.value
    assert store.get(job.id).verify_attempts == 2


@pytest.mark.asyncio
async def test_max_attempts_higher_than_default_allows_more_rounds(store, runner, orchestrator):
    """max_attempts=5 must allow up to 5 verifier rounds before stuck."""
    job = store.add(
        Job(
            id="",
            parent_session_id="parent-1",
            prompt="x",
            acceptance_criteria=["x"],
            max_attempts=5,
        )
    )
    orchestrator.register_worker(job.id, "w0", timeout_minutes=30)
    fail = json.dumps({"ac_results": [{"ac_text": "x", "pass": False, "reason": "n"}], "overall_pass": False})
    # 5 rounds: rounds 1..4 should loop, round 5 should stick.
    for i in range(4):
        await orchestrator.on_session_complete(_worker_session(store.get(job.id), f"w{i}"), f"w{i}")
        await orchestrator.on_session_complete(_verifier_session(store.get(job.id), f"v{i}"), fail)
        assert store.get(job.id).state == JobState.RUNNING.value, f"round {i} should still loop"
    await orchestrator.on_session_complete(_worker_session(store.get(job.id), "w4"), "w4")
    await orchestrator.on_session_complete(_verifier_session(store.get(job.id), "v4"), fail)
    assert store.get(job.id).state == JobState.STUCK.value


@pytest.mark.asyncio
async def test_notify_when_done_fires_only_on_done(store, runner, orchestrator):
    """notify_when='done' must wake the parent on DONE but NOT on STUCK/CANCELLED."""
    sent: list = []

    async def fake_reply(session_id, message, source="session", metadata=None):
        sent.append((session_id, message))
        return "ok"

    runner.reply_to_session = fake_reply
    job = store.add(
        Job(
            id="",
            parent_session_id="parent-1",
            prompt="x",
            acceptance_criteria=[],
            notify_when="done",
        )
    )
    orchestrator.register_worker(job.id, "w0", timeout_minutes=30)
    await orchestrator.on_session_complete(_worker_session(store.get(job.id)), "all done")
    await asyncio.sleep(0)
    assert sent, "notify_when=done must fire on DONE"

    sent.clear()
    job2 = _seed_running_job(store, orchestrator, runner, acceptance_criteria=["x"])
    # Carry notify_when=done onto job2 via update
    store.update(job2.id, notify_when="done")
    cancelled = _worker_session(job2, status=SessionStatus.CANCELLED.value)
    await orchestrator.on_session_complete(cancelled, "")
    await asyncio.sleep(0)
    assert sent == [], "notify_when=done must NOT fire on CANCELLED"


@pytest.mark.asyncio
async def test_notify_when_terminal_fires_on_any_terminal_state(store, runner, orchestrator):
    """notify_when='terminal' fires on done, stuck, errored, and cancelled."""
    sent: list = []

    async def fake_reply(session_id, message, source="session", metadata=None):
        sent.append((session_id, message))
        return "ok"

    runner.reply_to_session = fake_reply

    # Make a CANCELLED Job with notify_when=terminal.
    job = store.add(
        Job(
            id="",
            parent_session_id="parent-1",
            prompt="x",
            acceptance_criteria=["x"],
            notify_when="terminal",
        )
    )
    orchestrator.register_worker(job.id, "w0", timeout_minutes=30)
    cancelled = _worker_session(job, status=SessionStatus.CANCELLED.value)
    await orchestrator.on_session_complete(cancelled, "")
    await asyncio.sleep(0)
    assert sent, "notify_when=terminal must fire on CANCELLED"


@pytest.mark.asyncio
async def test_notify_legacy_bool_maps_to_notify_when_terminal(store, runner, orchestrator):
    """A Job constructed with `notify=True` (legacy) must behave identically to
    notify_when='terminal' — wake on any terminal state."""
    sent: list = []

    async def fake_reply(session_id, message, source="session", metadata=None):
        sent.append((session_id, message))
        return "ok"

    runner.reply_to_session = fake_reply
    job = store.add(
        Job(
            id="",
            parent_session_id="parent-1",
            prompt="x",
            acceptance_criteria=[],
            notify=True,
        )
    )
    # __post_init__ should have promoted notify=True to notify_when="terminal".
    assert store.get(job.id).notify_when == "terminal"
    orchestrator.register_worker(job.id, "w0", timeout_minutes=30)
    await orchestrator.on_session_complete(_worker_session(store.get(job.id)), "ok")
    await asyncio.sleep(0)
    assert sent, "legacy notify=True must still wake the parent on terminal transition"


@pytest.mark.asyncio
async def test_notify_when_never_fires_no_notify(store, runner, orchestrator):
    sent: list = []

    async def fake_reply(session_id, message, source="session", metadata=None):
        sent.append((session_id, message))
        return "ok"

    runner.reply_to_session = fake_reply
    job = store.add(
        Job(
            id="",
            parent_session_id="parent-1",
            prompt="x",
            acceptance_criteria=[],
            notify_when="never",
        )
    )
    orchestrator.register_worker(job.id, "w0", timeout_minutes=30)
    await orchestrator.on_session_complete(_worker_session(store.get(job.id)), "ok")
    await asyncio.sleep(0)
    assert sent == []


def test_create_and_start_job_threads_max_attempts(store, runner, orchestrator):
    job, _ = orchestrator.create_and_start_job(
        parent_session_id="parent-1",
        prompt="x",
        acceptance_criteria=[],
        max_attempts=7,
        notify_when="stuck",
    )
    fresh = store.get(job.id)
    assert fresh.max_attempts == 7
    assert fresh.notify_when == "stuck"


def test_retry_with_hint_fresh_workspace_prunes_and_recreates_worktree(store, runner, orchestrator, tmp_path):
    """fresh_workspace=True on a Job with a repo must prune the existing worktree
    and recreate a new one from HEAD before spawning the retry worker."""
    repo = tmp_path / "repo"
    _make_git_repo(repo)
    job, _ = orchestrator.create_and_start_job(
        parent_session_id="parent-1",
        prompt="do",
        acceptance_criteria=["x"],
        repo=str(repo),
    )
    fail = json.dumps({"ac_results": [{"ac_text": "x", "pass": False, "reason": "n"}], "overall_pass": False})
    import asyncio

    for i in range(MAX_VERIFY_ATTEMPTS):
        asyncio.run(orchestrator.on_session_complete(_worker_session(store.get(job.id), f"w{i}"), f"w{i}"))
        asyncio.run(orchestrator.on_session_complete(_verifier_session(store.get(job.id), f"v{i}"), fail))
    assert store.get(job.id).state == JobState.STUCK.value

    from pathlib import Path

    original_wt = store.get(job.id).worktree_path
    assert original_wt and Path(original_wt).exists()

    # Drop a sentinel file inside the worktree to detect the recreate.
    sentinel = Path(original_wt) / "sentinel.txt"
    sentinel.write_text("worker noodled here")
    assert sentinel.exists()

    asyncio.run(orchestrator.retry_with_hint(job.id, hint="reset env", fresh_workspace=True))
    fresh_wt = store.get(job.id).worktree_path
    assert fresh_wt and Path(fresh_wt).exists()
    # Sentinel must be gone — the worktree was recreated from HEAD.
    assert not (Path(fresh_wt) / "sentinel.txt").exists(), (
        "fresh_workspace=True must wipe the worker's stale changes by recreating the worktree"
    )
