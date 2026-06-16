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
        self._completion_listeners: list = []
        self.started: list[Session] = []
        self.cancelled: list[str] = []
        self._next_session_id_counter = 0
        # Seed the default parent session so create_and_start_job's parent-agent
        # lookup has something to find.
        self.store.sessions["parent-1"] = Session(id="parent-1", agent="default")

    def add_completion_listener(self, callback) -> None:
        if callback not in self._completion_listeners:
            self._completion_listeners.append(callback)

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
    # Fences are no longer stripped - the verifier MUST return raw JSON.
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
async def test_verifier_all_pass_without_overall_pass_marks_done(store, runner, orchestrator):
    """`response_format: json_object` guarantees valid JSON, not a complete schema.
    A verifier that lists all-pass criteria but omits the `overall_pass` summary
    key must still be treated as a pass - not retried into STUCK on a missing key."""
    job = _seed_running_job(store, orchestrator, runner, acceptance_criteria=["a", "b"])
    await orchestrator.on_session_complete(_worker_session(job), "## Summary\ndone\n- a: yes\n- b: yes")
    job = store.get(job.id)
    verifier_json = json.dumps(
        {
            "ac_results": [
                {"ac_text": "a", "pass": True, "reason": "yes"},
                {"ac_text": "b", "pass": True, "reason": "yes"},
            ]
            # overall_pass deliberately omitted
        }
    )
    await orchestrator.on_session_complete(_verifier_session(job), verifier_json)
    assert store.get(job.id).state == JobState.DONE.value, (
        "all-pass ac_results must mark DONE even without overall_pass"
    )


@pytest.mark.asyncio
async def test_verifier_derived_overall_pass_does_not_over_pass(store, runner, orchestrator):
    """Guard: deriving overall_pass from ac_results must not pass a job that has a
    failing criterion just because the summary key is absent."""
    job = _seed_running_job(store, orchestrator, runner, acceptance_criteria=["a", "b"])
    await orchestrator.on_session_complete(_worker_session(job), "out")
    job = store.get(job.id)
    verifier_json = json.dumps(
        {
            "ac_results": [
                {"ac_text": "a", "pass": True, "reason": "y"},
                {"ac_text": "b", "pass": False, "reason": "n"},
            ]
            # overall_pass omitted; one criterion fails → must NOT be DONE
        }
    )
    await orchestrator.on_session_complete(_verifier_session(job), verifier_json)
    assert store.get(job.id).state != JobState.DONE.value


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


def _seed_running_job_orphan_parent(store, orchestrator, runner, *, acceptance_criteria=None, agent=None):
    """Seed a RUNNING job whose parent_session_id is NOT in the session store, so
    `_resolve_adapter_key` misses the parent lookup and must fall through to the
    runner's `_adapters` fallback. (The default fixture seeds 'parent-1', which
    would short-circuit the fallback via the parent-agent resolution path.)"""
    job = store.add(
        Job(
            id="",
            parent_session_id="orphan-parent",  # deliberately absent from runner.store.sessions
            prompt="do the thing",
            acceptance_criteria=acceptance_criteria or [],
            agent=agent,
        )
    )
    orchestrator.register_worker(job.id, "worker-1", timeout_minutes=30)
    return store.get(job.id)


@pytest.mark.asyncio
async def test_verifier_falls_back_to_registered_adapter_when_parent_lookup_misses(store, runner, orchestrator):
    """When the parent session can't be resolved (not in the store), the verifier
    spawn must fall back to a registered adapter key from runner._adapters rather
    than passing the agent_file name as the adapter key - otherwise the spawn fails
    with "no adapter for agent 'job_verifier'"."""
    # Parent lookup will MISS (orphan-parent not seeded), so the only routing signal
    # is _adapters. Use a sentinel key that is neither 'default' nor 'job_verifier'.
    assert "orphan-parent" not in runner.store.sessions
    runner._adapters = {"some_key": object()}

    job = _seed_running_job_orphan_parent(store, orchestrator, runner, acceptance_criteria=["x"])
    await orchestrator.on_session_complete(_worker_session(job), "worker output")

    verifier_spawns = [s for s in runner.started if (s.metadata or {}).get("verifier_for")]
    assert len(verifier_spawns) == 1
    v = verifier_spawns[0]
    assert v.agent == "some_key", (
        f"verifier must route through the registered adapter key when parent lookup misses, got agent={v.agent!r}"
    )
    assert v.agent_file == "job_verifier", (
        f"verifier must still load job_verifier.md via agent_file, got agent_file={v.agent_file!r}"
    )


@pytest.mark.asyncio
async def test_loopback_worker_falls_back_to_registered_adapter_when_parent_lookup_misses(store, runner, orchestrator):
    """On verifier failure (< max), the loop-back worker must also fall back to a
    registered adapter key when the parent can't be resolved, rather than using
    agent=job.agent directly (which is the agent_file name, not an adapter key)."""
    assert "orphan-parent" not in runner.store.sessions
    runner._adapters = {"some_key": object()}

    job = _seed_running_job_orphan_parent(store, orchestrator, runner, acceptance_criteria=["x"], agent="job_worker")
    await orchestrator.on_session_complete(_worker_session(job), "w1")
    job = store.get(job.id)
    fail_json = json.dumps({"ac_results": [{"ac_text": "x", "pass": False, "reason": "no"}], "overall_pass": False})
    await orchestrator.on_session_complete(_verifier_session(job, "v1"), fail_json)

    loopback_workers = [s for s in runner.started if (s.metadata or {}).get("loop_attempt")]
    assert len(loopback_workers) == 1
    w = loopback_workers[0]
    assert w.agent == "some_key", (
        f"loop-back worker must route through the registered adapter key when parent lookup misses, got agent={w.agent!r}"
    )
    assert w.agent_file == "job_worker", (
        f"loop-back worker must still load job_worker.md via agent_file, got agent_file={w.agent_file!r}"
    )


@pytest.mark.asyncio
async def test_verifier_routes_through_parent_session_adapter_when_present(store, runner, orchestrator):
    """Separate coverage of the parent-resolution path: when the parent session
    IS in the store, the verifier routes through the parent's agent, NOT the
    arbitrary first _adapters key."""
    # Parent 'parent-1' is seeded with agent='default'; add a decoy _adapters entry
    # that would win if the fallback (incorrectly) ran instead of parent resolution.
    runner._adapters = {"decoy_key": object()}
    job = _seed_running_job(store, orchestrator, runner, acceptance_criteria=["x"])
    await orchestrator.on_session_complete(_worker_session(job), "worker output")

    verifier_spawns = [s for s in runner.started if (s.metadata or {}).get("verifier_for")]
    assert len(verifier_spawns) == 1
    assert verifier_spawns[0].agent == "default", (
        "verifier must route through the parent session's agent when the parent resolves, "
        f"not the _adapters fallback; got agent={verifier_spawns[0].agent!r}"
    )


# ── real-SessionStore contract: get_session RAISES on miss ──


class RaisingStore:
    """Stand-in mirroring real SessionStore.get_session, which RAISES ValueError
    on an unknown session id (FakeStore returns None). The orchestrator guards
    every get_session call with `except (ValueError, KeyError)`, so these paths
    must finalize jobs / fall back to adapters without the exception escaping."""

    def __init__(self):
        self.events: list[tuple[str, dict]] = []
        self.sessions: dict[str, Session] = {}

    def append_event(self, session_id, event):
        self.events.append((session_id, event))

    def get_session(self, session_id: str):
        try:
            return self.sessions[session_id]
        except KeyError:
            raise ValueError(f"Unknown session: {session_id}") from None


class RaisingStoreRunner:
    """FakeRunner whose store raises ValueError on get_session misses."""

    def __init__(self):
        self.store = RaisingStore()
        self.started: list[Session] = []
        self.cancelled: list[str] = []
        self._next_session_id_counter = 0

    def start_session(self, session: Session) -> Session:
        self._next_session_id_counter += 1
        if not session.id:
            session.id = f"session-{self._next_session_id_counter}"
        session.status = SessionStatus.RUNNING.value
        self.started.append(session)
        return session

    def cancel_session(self, session_id: str) -> None:
        self.cancelled.append(session_id)


@pytest.mark.asyncio
async def test_cancel_job_finalizes_when_store_raises_on_session_miss(store, event_bus):
    """cancel_job calls _session_already_terminal(worker_session_id); with the real
    store contract that's a ValueError on a missing session. The job must still be
    finalized to CANCELLED and the exception must NOT propagate."""
    runner = RaisingStoreRunner()
    orchestrator = JobsOrchestrator(store, runner, event_bus=event_bus)
    job = store.add(Job(id="", parent_session_id="parent-1", prompt="x"))
    orchestrator.register_worker(job.id, "worker-1", timeout_minutes=30)
    # worker-1 is NOT in runner.store.sessions → get_session raises ValueError.
    await orchestrator.cancel_job(job.id, reason="user cancel")
    assert store.get(job.id).state == JobState.CANCELLED.value
    assert "user cancel" in (store.get(job.id).error or "")


def test_on_timeout_finalizes_when_store_raises_on_session_miss(store, event_bus):
    """_on_timeout probes _session_already_terminal for both worker and verifier ids;
    a ValueError-on-miss store must not break finalization to STUCK."""
    runner = RaisingStoreRunner()
    orchestrator = JobsOrchestrator(store, runner, event_bus=event_bus)
    job = store.add(Job(id="", parent_session_id="parent-1", prompt="x"))
    orchestrator.register_worker(job.id, "worker-1", timeout_minutes=30)
    orchestrator._on_timeout(job.id)
    final = store.get(job.id)
    assert final.state == JobState.STUCK.value
    assert "timeout" in (final.error or "")


def test_resolve_adapter_key_falls_through_to_fallback_when_store_raises(store, event_bus):
    """When the parent lookup raises ValueError (real-store miss), _resolve_adapter_key
    must swallow it and fall through to the _adapters fallback rather than crashing."""
    runner = RaisingStoreRunner()
    runner._adapters = {"some_key": object()}
    orchestrator = JobsOrchestrator(store, runner, event_bus=event_bus)
    # 'absent-parent' is not in runner.store.sessions → get_session raises ValueError.
    assert orchestrator._resolve_adapter_key("absent-parent") == "some_key"


@pytest.mark.asyncio
async def test_session_without_job_id_is_ignored(store, runner, orchestrator):
    unrelated = Session(id="x", agent="default", source="spawned", status="completed", metadata={})
    # Should not raise, should not touch the store.
    await orchestrator.on_session_complete(unrelated, "ok")


def test_on_timeout_transitions_running_job_to_stuck(store, runner, orchestrator):
    """Direct call to _on_timeout while the Job is in RUNNING must mark it STUCK.

    Regression: the state machine previously rejected running→stuck, so the
    transition raised inside _finalize and the Job stayed running forever - the
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


def test_attach_coexists_with_other_listeners(store, runner, orchestrator):
    """Attaching the orchestrator must not displace a previously registered
    completion listener - both fire."""
    calls = []

    async def existing(session, result):
        calls.append((session.id, result))

    runner.add_completion_listener(existing)
    orchestrator.attach()
    assert existing in runner._completion_listeners
    assert orchestrator.on_session_complete in runner._completion_listeners

    # Dispatch a benign session that has no job_id (orchestrator ignores it).
    session = Session(id="x", agent="default", source="spawned", status="completed", metadata={})

    async def dispatch():
        for cb in runner._completion_listeners:
            await cb(session, "result")

    asyncio.run(dispatch())
    assert calls == [("x", "result")]


# ── critical regressions from code-review pass ──


@pytest.mark.asyncio
async def test_verifier_returning_string_ac_results_does_not_crash(store, runner, orchestrator):
    """Verifier may return ac_results entries that aren't dicts (e.g. just strings).
    The orchestrator must not raise - treat as a verifier failure with a synthetic ac entry."""
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
    in dict order - otherwise jobs leak across agents (credentials, tools, model)."""
    runner._adapters = {"coder": object(), "support": object()}
    # Parent session lives in `support`.
    runner.store.sessions["parent-A"] = Session(id="parent-A", agent="support")
    job, started = await orchestrator.create_and_start_job(
        parent_session_id="parent-A",
        prompt="hello",
        acceptance_criteria=[],
        timeout_minutes=30,
        spawned_by="user-slash",
    )
    assert started.agent == "support", f"worker must route through parent's adapter 'support', got {started.agent!r}"


def test_on_timeout_skips_cancel_when_worker_already_terminal(store, runner, orchestrator):
    """If the worker has already completed/failed by the time the timeout handler
    fires, _on_timeout must NOT call cancel_session - otherwise it overwrites the
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
        asyncio.run(
            orchestrator.create_and_start_job(
                parent_session_id="parent-1",
                prompt="hello",
                acceptance_criteria=[],
                timeout_minutes=30,
                spawned_by="user-slash",
            )
        )
    # The Job exists; it must NOT be left in QUEUED.
    jobs = store.list_for_parent("parent-1")
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
    """Bare-value JSON must be rejected - `parsed.get('overall_pass')` would crash on int/list/str/bool."""
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
    """Calling attach() twice on the same runner must not double-register."""
    orchestrator.attach()
    orchestrator.attach()
    assert runner._completion_listeners.count(orchestrator.on_session_complete) == 1


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
    # Duplicate verifier-complete for the same verifier session - Job is now RUNNING.
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
    STUCK error must still carry actionable signal - not a headerless followup."""
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
    job, started = asyncio.run(
        orchestrator.create_and_start_job(
            parent_session_id="parent-1",
            prompt="do",
            acceptance_criteria=[],
            repo=str(repo),
        )
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
    job, started = asyncio.run(
        orchestrator.create_and_start_job(
            parent_session_id="parent-1",
            prompt="do",
            acceptance_criteria=[],
            repo=None,
        )
    )
    assert store.get(job.id).worktree_path is None
    assert started.workspace_override is None


def test_worktree_pruned_on_done(store, runner, orchestrator, tmp_path):
    repo = tmp_path / "repo"
    _make_git_repo(repo)
    job, _ = asyncio.run(
        orchestrator.create_and_start_job(
            parent_session_id="parent-1",
            prompt="do",
            acceptance_criteria=[],
            repo=str(repo),
        )
    )
    wt = store.get(job.id).worktree_path
    from pathlib import Path

    assert Path(wt).exists()

    async def _go():
        await orchestrator.on_session_complete(_worker_session(store.get(job.id)), "done")
        # Prune is offloaded to a background task (asyncio.to_thread) so it can't
        # block the loop; drain it before asserting the on-disk effect.
        await asyncio.gather(*list(orchestrator._bg_tasks))

    asyncio.run(_go())
    # No AC → short-circuit DONE → worktree pruned.
    assert not Path(wt).exists(), f"worktree at {wt} must be pruned after DONE"


@pytest.mark.asyncio
async def test_worktree_prune_runs_off_the_event_loop_thread(store, runner, orchestrator, monkeypatch, tmp_path):
    """`git worktree remove` shells out; running it inline on the daemon's single
    event loop freezes every other session/SSE/timer until it returns. _finalize
    must offload it to a worker thread."""
    import threading

    import tsugite.daemon.jobs_orchestrator as orch_mod

    repo = tmp_path / "repo"
    _make_git_repo(repo)
    job, _ = await orchestrator.create_and_start_job(
        parent_session_id="parent-1", prompt="do", acceptance_criteria=[], repo=str(repo)
    )
    loop_tid = threading.get_ident()
    seen: dict = {}

    def recording_prune(path):
        seen["tid"] = threading.get_ident()

    monkeypatch.setattr(orch_mod, "_prune_worktree", recording_prune)

    # No AC → worker completion short-circuits to DONE → prune.
    await orchestrator.on_session_complete(_worker_session(store.get(job.id)), "done")
    await asyncio.gather(*list(orchestrator._bg_tasks))

    assert "tid" in seen, "worktree was never pruned"
    assert seen["tid"] != loop_tid, "prune ran on the event-loop thread (must be offloaded via asyncio.to_thread)"
    assert store.get(job.id).worktree_path is None, "worktree_path must be cleared after finalize"


@pytest.mark.asyncio
async def test_initial_worktree_provision_runs_off_the_event_loop_thread(store, runner, orchestrator, monkeypatch):
    """`git worktree add` shells out; provisioning it inline on the daemon's single
    event loop would freeze every other session/SSE/timer until it returns. The
    initial spawn path must offload it via asyncio.to_thread, like the retry path."""
    import threading

    import tsugite.daemon.jobs_orchestrator as orch_mod

    loop_tid = threading.get_ident()
    seen: dict = {}

    def recording_provision(repo, job_id, workspace_root=None):
        seen["tid"] = threading.get_ident()
        return f"/tmp/fake-worktree/{job_id}"

    monkeypatch.setattr(orch_mod, "_provision_worktree", recording_provision)

    await orchestrator.create_and_start_job(
        parent_session_id="parent-1", prompt="do", acceptance_criteria=[], repo="some-repo"
    )

    assert "tid" in seen, "worktree was never provisioned"
    assert seen["tid"] != loop_tid, (
        "provisioning ran on the event-loop thread (must be offloaded via asyncio.to_thread)"
    )


@pytest.mark.asyncio
async def test_relative_repo_resolves_against_workspace_root(store, runner, orchestrator, tmp_path):
    """A relative --repo must resolve against the parent session's workspace root,
    NOT the daemon process CWD. Otherwise `/job --repo subdir` provisions the wrong
    directory (or errors) depending on where the daemon happens to be running."""
    from pathlib import Path

    repo = tmp_path / "myrepo"
    _make_git_repo(repo)
    # The parent session's effective workspace is tmp_path; --repo is relative to it.
    runner.store.sessions["parent-1"].workspace_override = str(tmp_path)

    job, _ = await orchestrator.create_and_start_job(
        parent_session_id="parent-1", prompt="do", acceptance_criteria=[], repo="myrepo"
    )

    fresh = store.get(job.id)
    assert fresh.worktree_path is not None, "a relative --repo under the workspace must still provision a worktree"
    wt = Path(fresh.worktree_path)
    assert wt.is_relative_to(repo.resolve()), (
        f"worktree {wt} must resolve under the workspace repo {repo}, not the daemon CWD"
    )
    assert (wt / "README.md").exists(), "worktree must contain the repo's files"


@pytest.mark.asyncio
async def test_relative_repo_falls_back_to_adapter_workspace_dir(store, runner, orchestrator, tmp_path):
    """When the parent session has no workspace_override (the normal chat case), a
    relative --repo resolves against the parent adapter's agent_config.workspace_dir."""
    from pathlib import Path
    from types import SimpleNamespace

    repo = tmp_path / "myrepo"
    _make_git_repo(repo)
    runner._adapters = {"default": SimpleNamespace(agent_config=SimpleNamespace(workspace_dir=tmp_path))}

    job, _ = await orchestrator.create_and_start_job(
        parent_session_id="parent-1", prompt="do", acceptance_criteria=[], repo="myrepo"
    )

    wt = Path(store.get(job.id).worktree_path)
    assert wt.is_relative_to(repo.resolve()), f"worktree {wt} must resolve under the adapter workspace_dir repo {repo}"


@pytest.mark.asyncio
async def test_no_verifier_spawn_when_timeout_finalizes_during_predicate_eval(store, runner, orchestrator, monkeypatch):
    """The per-phase timeout can fire while predicate ACs are being evaluated -
    that's an `await asyncio.to_thread(...)` yield point. By the time eval
    returns the Job may already be STUCK; the handler must re-check state and
    NOT go on to spawn a verifier / re-arm a timer on the terminal Job."""
    import tsugite.daemon.jobs_orchestrator as orch_mod

    # One predicate (drives the await) + one prose AC (would trigger a verifier spawn).
    job = _seed_running_job(store, orchestrator, runner, acceptance_criteria=["cmd:true", "prose criterion"])

    def eval_then_timeout(predicate, *, cwd, ac_index, ac_text, attempt, sandbox_override=None):
        # Simulate the timeout firing mid-await: finalize the Job to STUCK.
        orchestrator._on_timeout(job.id)
        return {"ac_index": ac_index, "ac_text": ac_text, "pass": True, "reason": "ok", "attempt": attempt}

    monkeypatch.setattr(orch_mod, "_evaluate_predicate", eval_then_timeout)

    spawned_before = len(runner.started)
    await orchestrator._handle_worker_complete(store.get(job.id), _worker_session(store.get(job.id)), "worker out")

    fresh = store.get(job.id)
    assert fresh.state == JobState.STUCK.value, "timeout should have finalized the Job to STUCK"
    assert len(runner.started) == spawned_before, "must NOT spawn a verifier on a terminal (STUCK) Job"
    assert job.id not in orchestrator._timeout_handles, "must NOT re-arm a timer on a terminal Job"


def test_worktree_kept_on_stuck_for_inspection(store, runner, orchestrator, tmp_path):
    repo = tmp_path / "repo"
    _make_git_repo(repo)
    job, _ = asyncio.run(
        orchestrator.create_and_start_job(
            parent_session_id="parent-1",
            prompt="do",
            acceptance_criteria=["x"],
            repo=str(repo),
        )
    )
    wt = store.get(job.id).worktree_path
    from pathlib import Path

    fail = json.dumps({"ac_results": [{"ac_text": "x", "pass": False, "reason": "no"}], "overall_pass": False})

    for i in range(MAX_VERIFY_ATTEMPTS):
        asyncio.run(orchestrator.on_session_complete(_worker_session(store.get(job.id), f"w{i}"), f"w{i}"))
        asyncio.run(orchestrator.on_session_complete(_verifier_session(store.get(job.id), f"v{i}"), fail))
    assert store.get(job.id).state == JobState.STUCK.value
    assert Path(wt).exists(), "worktree must be kept on STUCK for inspection"


def test_worktree_pruned_on_mark_done_manual(store, runner, orchestrator, tmp_path):
    """A STUCK --repo job keeps its worktree for inspection, but mark_done_manual
    is a clean exit - the worktree must be pruned and worktree_path cleared, the
    same as the DONE/CANCELLED paths in _finalize."""
    repo = tmp_path / "repo"
    _make_git_repo(repo)
    job, _ = asyncio.run(
        orchestrator.create_and_start_job(
            parent_session_id="parent-1",
            prompt="do",
            acceptance_criteria=["x"],
            repo=str(repo),
        )
    )
    wt = store.get(job.id).worktree_path
    from pathlib import Path

    fail = json.dumps({"ac_results": [{"ac_text": "x", "pass": False, "reason": "no"}], "overall_pass": False})

    for i in range(MAX_VERIFY_ATTEMPTS):
        asyncio.run(orchestrator.on_session_complete(_worker_session(store.get(job.id), f"w{i}"), f"w{i}"))
        asyncio.run(orchestrator.on_session_complete(_verifier_session(store.get(job.id), f"v{i}"), fail))
    assert store.get(job.id).state == JobState.STUCK.value
    assert Path(wt).exists(), "worktree should still be on disk while STUCK"

    asyncio.run(orchestrator.mark_done_manual(job.id, reason="looks fine to me"))
    final = store.get(job.id)
    assert final.state == JobState.DONE.value
    assert not Path(wt).exists(), f"worktree at {wt} must be pruned after mark_done_manual"
    assert final.worktree_path is None, "worktree_path must be cleared after pruning"


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

    job = store.add(
        Job(id="", parent_session_id="parent-1", prompt="do thing", acceptance_criteria=[], notify_when="terminal")
    )
    orchestrator.register_worker(job.id, "worker-1", timeout_minutes=30)
    await orchestrator.on_session_complete(_worker_session(job), "all done")
    # Let the scheduled task run
    await asyncio.sleep(0)
    assert sent, "notify_when=terminal must trigger reply_to_session on terminal"
    assert "Job " + job.id in sent[0][1]
    assert "get_job" in sent[0][1]


@pytest.mark.asyncio
async def test_notify_false_does_not_fire_reply(store, runner, orchestrator):
    sent = []

    async def fake_reply(session_id, message, source="session", metadata=None):
        sent.append((session_id, message))
        return "ok"

    runner.reply_to_session = fake_reply

    job = store.add(Job(id="", parent_session_id="parent-1", prompt="do thing", acceptance_criteria=[]))
    orchestrator.register_worker(job.id, "worker-1", timeout_minutes=30)
    await orchestrator.on_session_complete(_worker_session(job), "all done")
    await asyncio.sleep(0)
    assert sent == [], "notify_when=never (default) must NOT trigger reply_to_session"


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
    job, started = asyncio.run(
        orchestrator.create_and_start_job(parent_session_id="parent-1", prompt="hi", acceptance_criteria=[])
    )
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


def test_emit_job_event_includes_acceptance_criteria(store, runner, orchestrator):
    """AC strings flow through the tile event payload unchanged."""
    job = store.add(Job(id="", parent_session_id="parent-1", prompt="x", acceptance_criteria=["plain text", "another"]))
    orchestrator._emit_job_event(store.get(job.id))
    emitted = [call.args[1] for call in orchestrator._event_bus.emit.call_args_list if call.args[0] == "job_update"]
    assert emitted, "expected a job_update emit"
    assert emitted[-1]["acceptance_criteria"] == ["plain text", "another"]


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


def test_legacy_notify_bool_migrates_on_load(tmp_path):
    """A pre-feature jobs.json record with `notify: true` must load as
    notify_when='terminal' so older daemons' persisted state keeps working."""
    import json

    from tsugite.daemon.job_store import JobStore

    path = tmp_path / "jobs.json"
    path.write_text(
        json.dumps(
            {
                "jobs": [
                    {
                        "id": "job-legacy",
                        "parent_session_id": "p1",
                        "prompt": "x",
                        "state": "queued",
                        "notify": True,
                    }
                ]
            }
        )
    )
    reloaded = JobStore(path)
    assert reloaded.get("job-legacy").notify_when == "terminal"


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
    job, _ = asyncio.run(
        orchestrator.create_and_start_job(
            parent_session_id="parent-1",
            prompt="x",
            acceptance_criteria=[],
            max_attempts=7,
            notify_when="stuck",
        )
    )
    fresh = store.get(job.id)
    assert fresh.max_attempts == 7
    assert fresh.notify_when == "stuck"


@pytest.mark.asyncio
async def test_create_and_start_job_coerces_unknown_notify_when_to_never(store, runner, orchestrator):
    """An invalid notify_when value (typo, garbage from an old client) must be
    normalised to 'never' at intake so a bad string can't reach _should_notify and
    spam the parent - or worse, accidentally match a future state name."""
    job, _ = await orchestrator.create_and_start_job(
        parent_session_id="parent-1",
        prompt="x",
        acceptance_criteria=[],
        notify_when="whenever-i-feel-like-it",
    )
    fresh = store.get(job.id)
    assert fresh.notify_when == "never", f"unknown notify_when must be coerced to 'never', got {fresh.notify_when!r}"


# ── Gap 4: per-criterion ac_results ──


@pytest.mark.asyncio
async def test_verifier_result_stored_per_criterion(store, runner, orchestrator):
    """When the verifier completes, each AC verdict must land on Job.ac_results
    with ac_index, ac_text, pass, reason, and attempt fields."""
    job = _seed_running_job(store, orchestrator, runner, acceptance_criteria=["a", "b"])
    await orchestrator.on_session_complete(_worker_session(job), "w1")
    job = store.get(job.id)
    verifier_json = json.dumps(
        {
            "ac_results": [
                {"ac_text": "a", "pass": True, "reason": "yes"},
                {"ac_text": "b", "pass": False, "reason": "missing X"},
            ],
            "overall_pass": False,
        }
    )
    await orchestrator.on_session_complete(_verifier_session(job, "v1"), verifier_json)
    fresh = store.get(job.id)
    assert fresh.ac_results is not None
    assert len(fresh.ac_results) == 2
    a = next(r for r in fresh.ac_results if r["ac_text"] == "a")
    b = next(r for r in fresh.ac_results if r["ac_text"] == "b")
    assert a == {"ac_index": 0, "ac_text": "a", "pass": True, "reason": "yes", "attempt": 1}
    assert b == {"ac_index": 1, "ac_text": "b", "pass": False, "reason": "missing X", "attempt": 1}


@pytest.mark.asyncio
async def test_ac_results_accumulate_across_attempts(store, runner, orchestrator):
    """Each attempt appends one batch of N entries (N = #AC). Across two failed
    attempts the array should hold 2N entries tagged by attempt."""
    job = _seed_running_job(store, orchestrator, runner, acceptance_criteria=["a"])
    fail = json.dumps({"ac_results": [{"ac_text": "a", "pass": False, "reason": "n"}], "overall_pass": False})
    # Round 1
    await orchestrator.on_session_complete(_worker_session(store.get(job.id), "w1"), "w1")
    await orchestrator.on_session_complete(_verifier_session(store.get(job.id), "v1"), fail)
    # Round 2
    await orchestrator.on_session_complete(_worker_session(store.get(job.id), "w2"), "w2")
    await orchestrator.on_session_complete(_verifier_session(store.get(job.id), "v2"), fail)
    fresh = store.get(job.id)
    attempts_recorded = sorted({e["attempt"] for e in fresh.ac_results})
    assert attempts_recorded == [1, 2], f"expected attempts 1 and 2 in ac_results, got {attempts_recorded}"
    assert len(fresh.ac_results) == 2, "two attempts × one AC = two entries"


def test_emit_job_event_includes_ac_results(store, runner, orchestrator):
    """The tile event payload must carry ac_results so the frontend can render
    per-AC verdicts instead of synthesising state from Job.state."""
    job = store.add(
        Job(
            id="",
            parent_session_id="parent-1",
            prompt="x",
            acceptance_criteria=["a"],
            ac_results=[{"ac_index": 0, "ac_text": "a", "pass": True, "reason": None, "attempt": 1}],
        )
    )
    orchestrator._emit_job_event(store.get(job.id))
    emitted = [call.args[1] for call in orchestrator._event_bus.emit.call_args_list if call.args[0] == "job_update"]
    payload = emitted[-1]
    assert payload["ac_results"] == [{"ac_index": 0, "ac_text": "a", "pass": True, "reason": None, "attempt": 1}]


@pytest.mark.asyncio
async def test_ac_results_recorded_on_unparseable_verifier_output(store, runner, orchestrator):
    """Even when the verifier output is unparseable, a synthetic ac_result entry
    must land so the UI can show 'verifier output unparseable' rather than empty."""
    job = _seed_running_job(store, orchestrator, runner, acceptance_criteria=["x"])
    await orchestrator.on_session_complete(_worker_session(job), "w1")
    job = store.get(job.id)
    await orchestrator.on_session_complete(_verifier_session(job, "v1"), "definitely not json")
    fresh = store.get(job.id)
    assert fresh.ac_results, "unparseable verifier output must still produce an ac_results entry"
    assert fresh.ac_results[0]["pass"] is False
    assert "unparseable" in (fresh.ac_results[0].get("reason") or "").lower()


# ── Gap 5: notify metadata threading ──


@pytest.mark.asyncio
async def test_schedule_notify_passes_kind_job_notify(store, runner, orchestrator):
    """The notify dispatch must include kind='job_notify' in the metadata so the
    parent session's user_input event carries channel.kind for the frontend to
    detect notify messages without text-regexing the body."""
    sent: list[tuple] = []

    async def fake_reply(session_id, message, source="session", metadata=None):
        sent.append((session_id, message, source, metadata))
        return "ok"

    runner.reply_to_session = fake_reply

    job = store.add(
        Job(id="", parent_session_id="parent-1", prompt="do x", acceptance_criteria=[], notify_when="terminal")
    )
    orchestrator.register_worker(job.id, "w0", timeout_minutes=30)
    await orchestrator.on_session_complete(_worker_session(store.get(job.id)), "done")
    await asyncio.sleep(0)
    assert sent, "notify must fire"
    _, _, source, metadata = sent[-1]
    assert source == "job_complete"
    assert metadata is not None
    assert metadata.get("job_id") == job.id
    assert metadata.get("kind") == "job_notify", (
        f"notify metadata must carry kind='job_notify' so the frontend can flag it; got {metadata!r}"
    )


def test_retry_with_hint_fresh_workspace_prunes_and_recreates_worktree(store, runner, orchestrator, tmp_path):
    """fresh_workspace=True on a Job with a repo must prune the existing worktree
    and recreate a new one from HEAD before spawning the retry worker."""
    repo = tmp_path / "repo"
    _make_git_repo(repo)
    job, _ = asyncio.run(
        orchestrator.create_and_start_job(
            parent_session_id="parent-1",
            prompt="do",
            acceptance_criteria=["x"],
            repo=str(repo),
        )
    )
    fail = json.dumps({"ac_results": [{"ac_text": "x", "pass": False, "reason": "n"}], "overall_pass": False})

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
    # Sentinel must be gone - the worktree was recreated from HEAD.
    assert not (Path(fresh_wt) / "sentinel.txt").exists(), (
        "fresh_workspace=True must wipe the worker's stale changes by recreating the worktree"
    )


@pytest.mark.asyncio
async def test_concurrent_retry_with_hint_spawns_only_one_worker(store, runner, orchestrator, monkeypatch):
    """Two retries racing on the same STUCK job (e.g. two browser tabs) must not both
    pass the stuck guard and double-spawn. The check-then-act region straddles the
    fresh_workspace provisioning await, so without a per-job lock both callers spawn."""
    import threading

    import tsugite.daemon.jobs_orchestrator as orch_mod

    job = store.add(Job(id="", parent_session_id="parent-1", prompt="x", acceptance_criteria=["x"], repo="/fake/repo"))
    store.update(job.id, state=JobState.STUCK.value)

    # Park the FIRST provisioning mid-flight so the second caller's coroutine gets
    # scheduled while the first is still inside the check-then-act region. This is
    # what makes the race deterministic rather than timing-dependent.
    parked = threading.Event()
    release = threading.Event()

    def blocking_provision(repo, job_id, workspace_root=None):
        # Only the first caller parks (the test waits for `parked` before starting
        # the second, so the second always sees it set). Event ops are atomic, so
        # no extra mutex is needed.
        if not parked.is_set():
            parked.set()
            release.wait(timeout=5)
        return f"/tmp/fake-worktree/{job_id}"

    monkeypatch.setattr(orch_mod, "_provision_worktree", blocking_provision)
    monkeypatch.setattr(orch_mod, "_prune_worktree", lambda *a, **k: None)

    c1 = asyncio.create_task(orchestrator.retry_with_hint(job.id, hint="a", fresh_workspace=True))
    while not parked.is_set():
        await asyncio.sleep(0.01)
    c2 = asyncio.create_task(orchestrator.retry_with_hint(job.id, hint="b", fresh_workspace=True))
    await asyncio.sleep(0.05)  # let c2 reach its guard / block on the lock
    release.set()
    results = await asyncio.gather(c1, c2, return_exceptions=True)

    errors = [r for r in results if isinstance(r, Exception)]
    assert len(runner.started) == 1, (
        f"exactly one retry worker must spawn under concurrent retries, got {len(runner.started)}"
    )
    assert len(errors) == 1 and isinstance(errors[0], ValueError), (
        f"the second concurrent retry must be rejected (job no longer stuck), got {results}"
    )
    assert store.get(job.id).state == JobState.RUNNING.value


# ── review fixes: verifier workspace, late-notify diagnostics, restart recovery ──


@pytest.mark.asyncio
async def test_verifier_runs_in_worker_worktree(store, runner, orchestrator):
    """For repo jobs the verifier must inspect the worker's worktree, not the
    adapter's default workspace - otherwise `git diff` is empty there and every
    prose AC falsely fails."""
    job = store.add(
        Job(id="", parent_session_id="parent-1", prompt="p", acceptance_criteria=["change exists in foo.py"])
    )
    store.update(job.id, worktree_path="/tmp/wt/job-x")
    orchestrator.register_worker(job.id, "worker-1", timeout_minutes=30)
    await orchestrator.on_session_complete(_worker_session(store.get(job.id)), "did the thing")
    verifier = next(s for s in runner.started if s.agent_file == "job_verifier")
    assert verifier.workspace_override == "/tmp/wt/job-x"


@pytest.mark.asyncio
async def test_late_cancel_notify_preserves_terminal_diagnostics(store, runner, orchestrator):
    """_on_timeout finalizes STUCK then cancels the worker; the worker task's
    CANCELLED notify arrives afterwards. It must not clobber job.error - the
    timeout diagnostic is what the user needs to see."""
    job = _seed_running_job(store, orchestrator, runner)
    orchestrator._finalize(store.get(job.id), JobState.STUCK, error="timeout after 30 minutes")
    assert store.get(job.id).state == JobState.STUCK.value
    await orchestrator.on_session_complete(_worker_session(job, status=SessionStatus.CANCELLED.value), "CANCELLED")
    fresh = store.get(job.id)
    assert fresh.state == JobState.STUCK.value
    assert fresh.error == "timeout after 30 minutes", "late cancel notify must not overwrite the diagnostic"


def test_recover_orphaned_jobs_errors_active_jobs(tmp_path, runner, event_bus):
    """Jobs left QUEUED/RUNNING/VERIFYING by a previous daemon process have no
    timers and no live worker sessions - recovery must mark them errored (and
    retryable) instead of leaving them 'running' forever."""
    path = tmp_path / "jobs.json"
    s1 = JobStore(path)
    queued = s1.add(Job(id="", parent_session_id="parent-1", prompt="q"))
    running = s1.add(Job(id="", parent_session_id="parent-1", prompt="r"))
    s1.update_state(running.id, JobState.RUNNING.value)
    verifying = s1.add(Job(id="", parent_session_id="parent-1", prompt="v"))
    s1.update_state(verifying.id, JobState.RUNNING.value)
    s1.update_state(verifying.id, JobState.VERIFYING.value)
    done = s1.add(Job(id="", parent_session_id="parent-1", prompt="d"))
    s1.update_state(done.id, JobState.RUNNING.value)
    s1.update_state(done.id, JobState.VERIFYING.value)
    s1.update_state(done.id, JobState.DONE.value)

    # Simulated daemon restart: fresh store over the same file.
    s2 = JobStore(path)
    orch = JobsOrchestrator(s2, runner, event_bus=event_bus)
    recovered = orch.recover_orphaned_jobs()

    assert recovered == 3
    for jid in (queued.id, running.id, verifying.id):
        j = s2.get(jid)
        assert j.state == JobState.ERRORED.value
        assert "restart" in (j.error or "")
        assert j.resolved_at
    assert s2.get(done.id).state == JobState.DONE.value


@pytest.mark.asyncio
async def test_retry_with_hint_allowed_on_errored(store, runner, orchestrator):
    """Errored jobs (spawn failures, verifier infra failures, daemon-restart
    orphans) must be retryable - the UI offers Retry on them."""
    job = _seed_running_job(store, orchestrator, runner)
    await orchestrator.on_session_complete(_worker_session(job, status=SessionStatus.FAILED.value), "FAILED: boom")
    assert store.get(job.id).state == JobState.ERRORED.value

    runner.started.clear()
    updated = await orchestrator.retry_with_hint(job.id, hint="try again with X")
    assert updated.state == JobState.RUNNING.value
    assert updated.error is None
    assert len(runner.started) == 1
    assert "try again with X" in runner.started[0].prompt
