"""Record stores persist to state_dir/daemon.db (SQLite, write-through) instead
of rewriting a whole JSON file per mutation.

The legacy JSON path stays as the constructor argument: it's the one-time
migration source (imported only when the db collection is empty) and is left
byte-for-byte untouched afterwards.
"""

import json

from tsugite_daemon.job_store import Job, JobStore


def test_fresh_store_writes_daemon_db_not_json(tmp_path):
    store = JobStore(tmp_path / "jobs.json")
    store.add(Job(id="", parent_session_id="p", prompt="x"))
    assert (tmp_path / "daemon.db").exists(), "records must land in daemon.db"
    assert not (tmp_path / "jobs.json").exists(), "no JSON file writes after the migration to SQLite"


def test_mutation_durable_across_reopen_without_shutdown(tmp_path):
    """Write-through: a mutation is durable immediately - no flush, no close."""
    store = JobStore(tmp_path / "jobs.json")
    job = store.add(Job(id="", parent_session_id="p", prompt="x"))
    store.update(job.id, error="boom")
    reopened = JobStore(tmp_path / "jobs.json")
    assert reopened.get(job.id) is not None
    assert reopened.get(job.id).error == "boom"


def test_legacy_json_migrated_and_left_untouched(tmp_path):
    path = tmp_path / "jobs.json"
    path.write_text(
        json.dumps({"jobs": [{"id": "job-old", "parent_session_id": "p1", "prompt": "x", "state": "queued"}]})
    )
    original = path.read_text()

    store = JobStore(path)
    assert store.get("job-old") is not None, "legacy records must be imported"
    store.update("job-old", prompt="new prompt")

    assert path.read_text() == original, "the legacy JSON file must be left in place as a backup"
    reopened = JobStore(path)
    assert reopened.get("job-old").prompt == "new prompt", "the db, not the stale JSON, is the authority"


def test_migration_runs_only_when_db_collection_empty(tmp_path):
    path = tmp_path / "jobs.json"
    path.write_text(json.dumps({"jobs": [{"id": "job-a", "parent_session_id": "p", "prompt": "a", "state": "queued"}]}))
    s1 = JobStore(path)
    s1.update_state("job-a", "running")
    s1.update_state("job-a", "cancelled")

    s2 = JobStore(path)
    assert s2.get("job-a").state == "cancelled", "a second start must not re-import the stale legacy JSON"


def test_pruned_records_stay_pruned_across_reopen(tmp_path):
    """The retention-cap prune must delete rows durably - no ghosts that
    resurrect on the next daemon start."""
    store = JobStore(tmp_path / "jobs.json", max_terminal_jobs=1)
    a = store.add(Job(id="", parent_session_id="p", prompt="a"))
    b = store.add(Job(id="", parent_session_id="p", prompt="b"))
    for j in (a, b):
        store.update_state(j.id, "running")
        store.update_state(j.id, "cancelled")
    assert len(store.list_all()) == 1

    reopened = JobStore(tmp_path / "jobs.json")
    assert len(reopened.list_all()) == 1
