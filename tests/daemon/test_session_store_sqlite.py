"""SessionStore persistence: write-through to daemon.db instead of a dirty-flag
plus whole-file session_store.json rewrites.

The dirty-flag pattern meant any mutation between flushes (a pin, a metadata
edit) was lost if the daemon crashed - and every flush re-serialized every
session on the event loop."""

import json

from tsugite_daemon.session_store import Session, SessionSource, SessionStore


def _store(tmp_path, **kw):
    return SessionStore(tmp_path / "session_store.json", **kw)


def test_pin_survives_reopen_without_flush(tmp_path):
    """Write-through: a mutation that previously only marked the store dirty
    must be durable immediately - no flush(), no clean shutdown."""
    store = _store(tmp_path)
    store.create_session(Session(id="s1", agent="a", source=SessionSource.INTERACTIVE.value, user_id="u"))
    store.set_pin("s1", True)

    reopened = _store(tmp_path)
    loaded = reopened.get_session("s1")
    assert loaded.pinned is True
    assert loaded.pin_position == 0


def test_no_json_writes_after_conversion(tmp_path):
    store = _store(tmp_path)
    store.create_session(Session(id="s1", agent="a"))
    assert (tmp_path / "daemon.db").exists(), "sessions must land in daemon.db"
    assert not (tmp_path / "session_store.json").exists(), "no legacy JSON writes"


def test_legacy_json_migrated_and_left_untouched(tmp_path):
    path = tmp_path / "session_store.json"
    path.write_text(
        json.dumps(
            {
                "sessions": {
                    "s-legacy": {
                        "id": "s-legacy",
                        "agent": "a",
                        "source": SessionSource.INTERACTIVE.value,
                        "user_id": "u",
                        "title": "old chat",
                    }
                }
            }
        )
    )
    original = path.read_text()

    store = SessionStore(path)
    assert store.get_session("s-legacy").title == "old chat"
    store.set_pin("s-legacy", True)

    assert path.read_text() == original, "legacy file stays as an untouched backup"
    reopened = SessionStore(path)
    assert reopened.get_session("s-legacy").pinned is True, "the db, not the stale JSON, is the authority"


def test_metadata_update_durable_without_flush(tmp_path):
    store = _store(tmp_path)
    store.create_session(Session(id="s1", agent="a", user_id="u"))
    store.set_metadata_bulk("s1", {"topic": "databases"})

    reopened = _store(tmp_path)
    assert reopened.get_session("s1").metadata.get("topic") == "databases"


def test_purged_sessions_deleted_durably(tmp_path):
    """Prune paths (schedule/background caps) must delete rows, not leave ghosts
    that resurrect on the next start."""
    store = _store(tmp_path)
    store.create_session(Session(id="s1", agent="a", user_id="u"))
    with store._lock:
        store._purge_session_state("s1")

    reopened = _store(tmp_path)
    try:
        gone = reopened.get_session("s1")
    except ValueError:
        gone = None
    assert gone is None, "purged session must not survive a reopen"
