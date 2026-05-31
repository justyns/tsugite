"""Tests for the JobStore: round-trip persistence, state-machine guards, terminal finality."""

import threading

import pytest

from tsugite.daemon.job_store import (
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
    assert store_path.exists()


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
        (JobState.STUCK, JobState.RUNNING),
        (JobState.CANCELLED, JobState.RUNNING),
        (JobState.ERRORED, JobState.RUNNING),
    ],
)
def test_invalid_transition_raises(store, from_state, to_state):
    job = store.add(_make_job())
    if from_state is not JobState.QUEUED:
        _force_state(store, job.id, from_state.value)
    with pytest.raises(JobStateTransitionError):
        store.update_state(job.id, to_state.value)


def test_terminal_states_are_terminal(store):
    for terminal in (JobState.DONE, JobState.STUCK, JobState.CANCELLED, JobState.ERRORED):
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
    on load — otherwise the job is a permanent zombie (not terminal, no valid
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


def test_acceptance_criteria_legacy_string_format_loads(store_path):
    """Legacy jobs.json stored AC as a list of plain strings. On load each entry
    must be normalised to {text, kind:"llm"} so downstream code can rely on the
    dict shape uniformly."""
    import json

    legacy = {
        "jobs": [
            {
                "id": "job-legacy-ac",
                "parent_session_id": "p1",
                "prompt": "x",
                "state": "queued",
                "acceptance_criteria": ["tests pass", "PR open"],
            }
        ]
    }
    store_path.write_text(json.dumps(legacy))
    store = JobStore(store_path)
    job = store.get("job-legacy-ac")
    assert job is not None
    assert job.acceptance_criteria == [
        {"text": "tests pass", "kind": "llm"},
        {"text": "PR open", "kind": "llm"},
    ]


def test_acceptance_criteria_new_dict_format_round_trips(store_path):
    """New dict-shaped AC must round-trip through save/reload unchanged."""
    s1 = JobStore(store_path)
    job = s1.add(
        Job(
            id="",
            parent_session_id="p1",
            prompt="x",
            acceptance_criteria=[
                {"text": "tests pass", "kind": "test"},
                {"text": "endpoint returns 200", "kind": "cmd"},
            ],
        )
    )
    s2 = JobStore(store_path)
    reloaded = s2.get(job.id)
    assert reloaded.acceptance_criteria == [
        {"text": "tests pass", "kind": "test"},
        {"text": "endpoint returns 200", "kind": "cmd"},
    ]


def test_acceptance_criteria_mixed_input_normalises_on_add(store_path):
    """Passing a mix of strings and dicts (e.g. from the slash command parser) must
    normalise to dicts at construction time."""
    s = JobStore(store_path)
    job = s.add(
        Job(
            id="",
            parent_session_id="p1",
            prompt="x",
            acceptance_criteria=["plain text", {"text": "typed", "kind": "ui"}],
        )
    )
    assert job.acceptance_criteria == [
        {"text": "plain text", "kind": "llm"},
        {"text": "typed", "kind": "ui"},
    ]


def test_acceptance_criteria_dict_without_kind_defaults_to_llm(store_path):
    s = JobStore(store_path)
    job = s.add(
        Job(
            id="",
            parent_session_id="p1",
            prompt="x",
            acceptance_criteria=[{"text": "no kind"}],
        )
    )
    assert job.acceptance_criteria == [{"text": "no kind", "kind": "llm"}]


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
