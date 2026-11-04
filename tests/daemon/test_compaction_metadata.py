"""Compaction must preserve user-authored durable metadata (task/refs/notes)."""

import pytest
from tsugite_daemon.session_store import Session, SessionSource, SessionStore


@pytest.fixture
def store(tmp_path):
    return SessionStore(tmp_path / "session_store.json")


def _compacted(store, metadata):
    s = Session(id="s-1", source=SessionSource.INTERACTIVE.value, user_id="u1", metadata=metadata)
    store.create_session(s)
    return store.compact_session(s.id)


def test_compaction_preserves_task_refs_notes(store):
    refs = [{"forge": "home", "repo": "octo-org/demo", "pr": 7, "source": "pinned"}]
    new = _compacted(
        store,
        {
            "task": "https://tracker/task/42",
            "refs": refs,
            "notes": "long-lived pinned session notes",
        },
    )
    assert new.metadata.get("task") == "https://tracker/task/42"
    assert new.metadata.get("refs") == refs
    assert new.metadata.get("notes") == "long-lived pinned session notes"


def test_compaction_clears_status_text(store):
    new = _compacted(store, {"status_text": "investigating", "notes": "keep me"})
    assert "status_text" not in new.metadata
    assert new.metadata.get("notes") == "keep me"
