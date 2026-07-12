"""Tests for the JobStore: round-trip persistence, state-machine guards, terminal finality."""

import threading

import pytest
from tsugite_daemon.job_store import (
    Job,
    JobState,
    JobStateTransitionError,
    JobStore,
)


@pytest.fixture
def store_path(tmp_path):
    return tmp_path / "jobs.json"


@pytest.fixture
def store(store_path):
    return JobStore(store_path)


def _make_job(parent_session_id="parent-1", prompt="do a thing"):
    return Job(id="", parent_session_id=parent_session_id, prompt=prompt)


def test_add_assigns_id_and_persists(store, store_path):
    job = store.add(_make_job())
    assert job.id.startswith("job-")
    assert store.get(job.id) is job
    assert (store_path.parent / "daemon.db").exists()


def test_round_trip_via_reload(store_path):
    s1 = JobStore(store_path)
    job = s1.add(_make_job(prompt="alpha"))
    s1.update(job.id, worker_session_id="worker-1")
    s1.update_state(job.id, JobState.RUNNING.value)

    s2 = JobStore(store_path)
    loaded = s2.get(job.id)
    assert loaded.prompt == "alpha"
    assert loaded.worker_session_id == "worker-1"
    assert loaded.state == JobState.RUNNING.value


def test_initial_state_is_queued(store):
    job = store.add(_make_job())
    assert job.state == JobState.QUEUED.value


@pytest.mark.parametrize(
    "from_state,to_state",
    [
        (JobState.QUEUED, JobState.RUNNING),
        (JobState.RUNNING, JobState.VERIFYING),
        (JobState.RUNNING, JobState.STUCK),
        (JobState.VERIFYING, JobState.DONE),
        (JobState.VERIFYING, JobState.RUNNING),
        (JobState.VERIFYING, JobState.STUCK),
        (JobState.QUEUED, JobState.CANCELLED),
        (JobState.RUNNING, JobState.CANCELLED),
        (JobState.VERIFYING, JobState.CANCELLED),
        (JobState.QUEUED, JobState.ERRORED),
        (JobState.RUNNING, JobState.ERRORED),
        (JobState.VERIFYING, JobState.ERRORED),
    ],
)
def test_valid_transitions(store, from_state, to_state):
    job = store.add(_make_job())
    if from_state is not JobState.QUEUED:
        _force_state(store, job.id, from_state.value)
    updated = store.update_state(job.id, to_state.value)
    assert updated.state == to_state.value
    assert updated.updated_at >= job.created_at


@pytest.mark.parametrize(
    "from_state,to_state",
    [
        (JobState.QUEUED, JobState.VERIFYING),
        (JobState.QUEUED, JobState.DONE),
        (JobState.RUNNING, JobState.DONE),
        (JobState.DONE, JobState.RUNNING),
        (JobState.DONE, JobState.CANCELLED),
        (JobState.CANCELLED, JobState.RUNNING),
        (JobState.ERRORED, JobState.DONE),
    ],
)
def test_invalid_transition_raises(store, from_state, to_state):
    job = store.add(_make_job())
    if from_state is not JobState.QUEUED:
        _force_state(store, job.id, from_state.value)
    with pytest.raises(JobStateTransitionError):
        store.update_state(job.id, to_state.value)


@pytest.mark.parametrize(
    "from_state,to_state",
    [
        # The user-facing escape hatches (retry-with-hint, mark-done, retry an
        # errored job, cancel/dismiss) are real transitions, not out-of-band
        # mutations.
        (JobState.STUCK, JobState.RUNNING),
        (JobState.STUCK, JobState.DONE),
        (JobState.STUCK, JobState.CANCELLED),
        (JobState.ERRORED, JobState.RUNNING),
        (JobState.ERRORED, JobState.CANCELLED),
    ],
)
def test_parked_states_have_escape_transitions(store, from_state, to_state):
    job = store.add(_make_job())
    _force_state(store, job.id, from_state.value)
    updated = store.update_state(job.id, to_state.value)
    assert updated.state == to_state.value


def test_sink_states_are_terminal(store):
    """DONE and CANCELLED are sinks; STUCK/ERRORED are parked-but-retryable."""
    for terminal in (JobState.DONE, JobState.CANCELLED):
        job = store.add(_make_job())
        _force_state(store, job.id, terminal.value)
        for target in (JobState.RUNNING, JobState.VERIFYING):
            with pytest.raises(JobStateTransitionError):
                store.update_state(job.id, target.value)


def test_list_active_excludes_terminal(store):
    a = store.add(_make_job(prompt="a"))
    b = store.add(_make_job(prompt="b"))
    _force_state(store, b.id, JobState.DONE.value)
    c = store.add(_make_job(prompt="c"))
    store.update_state(c.id, JobState.RUNNING.value)
    active = store.list_active()
    ids = {j.id for j in active}
    assert a.id in ids
    assert c.id in ids
    assert b.id not in ids


def test_list_for_parent_filters_correctly(store):
    a = store.add(_make_job(parent_session_id="p1"))
    b = store.add(_make_job(parent_session_id="p1"))
    c = store.add(_make_job(parent_session_id="p2"))
    p1 = {j.id for j in store.list_for_parent("p1")}
    assert p1 == {a.id, b.id}
    assert {j.id for j in store.list_for_parent("p2")} == {c.id}


def test_update_persists_arbitrary_fields(store, store_path):
    job = store.add(_make_job())
    store.update(job.id, verify_attempts=2, result={"summary": "ok"}, error=None)
    reloaded = JobStore(store_path).get(job.id)
    assert reloaded.verify_attempts == 2
    assert reloaded.result == {"summary": "ok"}


def test_concurrent_state_updates_serialise(store):
    job = store.add(_make_job())
    barrier = threading.Barrier(8)
    successes = []
    failures = []

    def attempt():
        barrier.wait()
        try:
            store.update_state(job.id, JobState.RUNNING.value)
            successes.append(1)
        except JobStateTransitionError:
            failures.append(1)

    threads = [threading.Thread(target=attempt) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Exactly one thread should make the queued → running transition; the
    # rest must see the now-invalid transition and raise.
    assert len(successes) == 1
    assert len(failures) == 7
    assert store.get(job.id).state == JobState.RUNNING.value


def test_readers_do_not_raise_while_writer_mutates(monkeypatch, store):
    """A reader iterating list_all()/list_active()/list_for_parent() concurrently
    with a writer add()ing jobs must never see
    'RuntimeError: dictionary changed size during iteration'.

    Load-bearing for the locking added to the read accessors: without holding
    self._lock, those readers iterate self._jobs.values() while add() mutates the
    same dict from another thread, which CPython rejects mid-iteration.

    Determinism: persistence is stubbed to a no-op (we're exercising the in-memory
    dict, not disk), and the baseline dict is large so each reader iteration spans
    a wide window for a concurrent insert to land inside. Multiple writer/reader
    threads + a bounded iteration budget keep it from hanging while making the
    unguarded race fire reliably.
    """
    # Don't pay row serialization on every add - it both slows the test and
    # shrinks the contention window we're trying to hit.
    monkeypatch.setattr(store._storage, "upsert", lambda *a, **kw: None)

    # Sizeable baseline → each reader iteration spans many items, so a concurrent
    # insert is very likely to land inside one iteration. Without the lock this
    # raises within the first handful of iterations; the budgets below are just a
    # hang-guard, not the mechanism.
    for _ in range(1200):
        store.add(_make_job())

    errors: list[Exception] = []
    stop = threading.Event()
    start = threading.Barrier(4)  # 2 readers + 2 writers fire together

    def reader():
        start.wait()
        for _ in range(2000):
            if stop.is_set():
                return
            try:
                store.list_all()
                store.list_active()
                store.list_for_parent("parent-1")
            except Exception as e:  # noqa: BLE001 - surface ANY raise, esp. RuntimeError
                errors.append(e)
                stop.set()
                return

    def writer():
        start.wait()
        try:
            for _ in range(2000):
                if stop.is_set():
                    return
                store.add(_make_job())
        finally:
            stop.set()

    threads = [threading.Thread(target=reader) for _ in range(2)] + [threading.Thread(target=writer) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    assert not errors, f"readers raised while a writer mutated the store: {errors!r}"


def _force_state(store, job_id, target_state):
    """Walk the state machine via valid transitions to reach target_state.

    Tests use this to set up preconditions without bypassing the guard.
    """
    paths = {
        JobState.RUNNING.value: [JobState.RUNNING.value],
        JobState.VERIFYING.value: [JobState.RUNNING.value, JobState.VERIFYING.value],
        JobState.DONE.value: [
            JobState.RUNNING.value,
            JobState.VERIFYING.value,
            JobState.DONE.value,
        ],
        JobState.STUCK.value: [
            JobState.RUNNING.value,
            JobState.VERIFYING.value,
            JobState.STUCK.value,
        ],
        JobState.CANCELLED.value: [JobState.CANCELLED.value],
        JobState.ERRORED.value: [JobState.ERRORED.value],
    }
    for step in paths[target_state]:
        store.update_state(job_id, step)


def test_load_migrates_legacy_looping_to_running(store_path, tmp_path):
    """Pre-upgrade jobs.json with state='looping' (a removed state) must be migrated
    on load - otherwise the job is a permanent zombie (not terminal, no valid
    transitions out)."""
    import json

    legacy = {
        "jobs": [
            {
                "id": "job-legacy01",
                "parent_session_id": "p1",
                "prompt": "x",
                "state": "looping",
                "verify_attempts": 1,
            }
        ]
    }
    store_path.write_text(json.dumps(legacy))
    store = JobStore(store_path)
    job = store.get("job-legacy01")
    assert job is not None
    assert job.state == JobState.RUNNING.value, (
        f"legacy 'looping' state must be migrated to 'running', got {job.state!r}"
    )


def test_acceptance_criteria_loads_plain_strings(store_path):
    """jobs.json stores AC as a list of plain strings; round-trips on load."""
    import json

    saved = {
        "jobs": [
            {
                "id": "job-ac1",
                "parent_session_id": "p1",
                "prompt": "x",
                "state": "queued",
                "acceptance_criteria": ["tests pass", "PR open"],
            }
        ]
    }
    store_path.write_text(json.dumps(saved))
    store = JobStore(store_path)
    job = store.get("job-ac1")
    assert job is not None
    assert job.acceptance_criteria == ["tests pass", "PR open"]


def test_acceptance_criteria_legacy_dict_shape_coerced_to_strings(store_path):
    """Old daemon versions persisted AC as `{text, kind}` dicts. On load, drop
    `kind` and keep only the text string so downstream code sees one shape."""
    import json

    legacy = {
        "jobs": [
            {
                "id": "job-legacy-dict",
                "parent_session_id": "p1",
                "prompt": "x",
                "state": "queued",
                "acceptance_criteria": [
                    {"text": "tests pass", "kind": "test"},
                    {"text": "PR open", "kind": "llm"},
                ],
            }
        ]
    }
    store_path.write_text(json.dumps(legacy))
    store = JobStore(store_path)
    job = store.get("job-legacy-dict")
    assert job is not None
    assert job.acceptance_criteria == ["tests pass", "PR open"]


def test_acceptance_criteria_mixed_input_coerced_to_strings(store_path):
    """A mix of strings and legacy dicts at construction coerces to strings."""
    s = JobStore(store_path)
    job = s.add(
        Job(
            id="",
            parent_session_id="p1",
            prompt="x",
            acceptance_criteria=["plain text", {"text": "from old caller"}],
        )
    )
    assert job.acceptance_criteria == ["plain text", "from old caller"]


def test_legacy_job_without_max_attempts_loads_with_default(store_path):
    """Pre-feature jobs.json files have no max_attempts field. Load must default it
    to 3 rather than crashing."""
    import json

    legacy = {
        "jobs": [
            {
                "id": "job-old1",
                "parent_session_id": "p1",
                "prompt": "x",
                "state": "queued",
            }
        ]
    }
    store_path.write_text(json.dumps(legacy))
    store = JobStore(store_path)
    job = store.get("job-old1")
    assert job is not None
    assert job.max_attempts == 3
    assert job.notify_when == "never"


def test_legacy_job_with_notify_true_loads_as_notify_when_terminal(store_path):
    """Pre-feature jobs persisted notify=True but no notify_when. Load must
    promote the legacy bool to notify_when='terminal' so behaviour is preserved."""
    import json

    legacy = {
        "jobs": [
            {
                "id": "job-old2",
                "parent_session_id": "p1",
                "prompt": "x",
                "state": "queued",
                "notify": True,
            }
        ]
    }
    store_path.write_text(json.dumps(legacy))
    store = JobStore(store_path)
    job = store.get("job-old2")
    assert job is not None
    assert job.notify_when == "terminal"


def test_legacy_notify_with_explicit_notify_when_preserves_explicit(store_path):
    """Regression: a persisted record with both `notify: true` AND `notify_when: never`
    (e.g. old caller explicitly opted out) must load as "never". The load-time
    migration only promotes legacy notify when notify_when wasn't set."""
    import json

    legacy = {
        "jobs": [
            {
                "id": "job-mixed",
                "parent_session_id": "p1",
                "prompt": "x",
                "state": "queued",
                "notify": True,
                "notify_when": "never",
            }
        ]
    }
    store_path.write_text(json.dumps(legacy))
    store = JobStore(store_path)
    assert store.get("job-mixed").notify_when == "never"


def test_executor_defaults_to_agent_and_no_worker_terminal(store):
    """A Job created without an explicit executor runs on the built-in agent path
    and has no embedded worker terminal until an executor stamps one."""
    job = store.add(_make_job())
    assert job.executor == "agent"
    assert job.worker_terminal_id is None


def test_executor_and_worker_terminal_id_round_trip(store_path):
    """Both new fields persist across a reload so an orphaned/recovered executor
    job still renders its embedded terminal and routes cancel to the right executor."""
    s1 = JobStore(store_path)
    job = s1.add(Job(id="", parent_session_id="p1", prompt="x", executor="cc"))
    s1.update(job.id, worker_terminal_id="term-abc123")

    loaded = JobStore(store_path).get(job.id)
    assert loaded.executor == "cc"
    assert loaded.worker_terminal_id == "term-abc123"


def test_to_payload_includes_executor_and_worker_terminal_id(store):
    job = store.add(Job(id="", parent_session_id="p1", prompt="x", executor="cc"))
    store.update(job.id, worker_terminal_id="term-xyz")
    payload = store.get(job.id).to_payload()
    assert payload["executor"] == "cc"
    assert payload["worker_terminal_id"] == "term-xyz"


def test_legacy_job_without_executor_loads_as_agent(store_path):
    """Pre-feature jobs.json files have no executor field. Load must default it to
    'agent' rather than crashing."""
    import json

    legacy = {"jobs": [{"id": "job-old3", "parent_session_id": "p1", "prompt": "x", "state": "queued"}]}
    store_path.write_text(json.dumps(legacy))
    job = JobStore(store_path).get("job-old3")
    assert job is not None
    assert job.executor == "agent"
    assert job.worker_terminal_id is None


def test_terminal_jobs_pruned_beyond_cap(store_path):
    """jobs.json must not grow forever: oldest resolved jobs are dropped once
    the terminal-job count exceeds the cap. Active jobs are never pruned."""
    store = JobStore(store_path, max_terminal_jobs=3)
    keep_active = store.add(_make_job(prompt="active"))
    store.update_state(keep_active.id, JobState.RUNNING.value)

    done_ids = []
    for i in range(5):
        j = store.add(_make_job(prompt=f"old-{i}"))
        store.update_state(j.id, JobState.RUNNING.value)
        store.update_state(j.id, JobState.VERIFYING.value)
        store.update_state(j.id, JobState.DONE.value)
        store.update(j.id, resolved_at=f"2026-01-0{i + 1}T00:00:00+00:00")
        done_ids.append(j.id)

    # Trigger a prune pass via a fresh terminal transition.
    j = store.add(_make_job(prompt="newest"))
    store.update_state(j.id, JobState.CANCELLED.value)

    remaining = {job.id for job in store.list_all()}
    assert keep_active.id in remaining, "active jobs are never pruned"
    assert j.id in remaining, "the newest terminal job survives"
    # Oldest resolved jobs got dropped to respect the cap of 3 terminal records.
    terminal_remaining = [job for job in store.list_all() if job.id != keep_active.id]
    assert len(terminal_remaining) == 3
    assert done_ids[0] not in remaining
    assert done_ids[1] not in remaining

    # Pruning persists across reload.
    reloaded = JobStore(store_path, max_terminal_jobs=3)
    assert len(reloaded.list_all()) == 4
