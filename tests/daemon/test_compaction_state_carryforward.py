"""Compaction carries the per-session `state` dict forward to the successor."""

import pytest
from tsugite_daemon.session_store import Session, SessionSource, SessionStore

from tsugite.core.state import load_state, save_state, session_state_path


@pytest.fixture
def store(tmp_path):
    return SessionStore(tmp_path / "session_store.json")


def _create(store):
    session = Session(id="s-1", source=SessionSource.INTERACTIVE.value, user_id="u1")
    store.create_session(session)
    return session


def test_state_json_survives_compaction(store):
    session = _create(store)
    old_path = session_state_path(session.id)
    state = {"issue": 215, "batch": [1, 2, 3]}
    save_state(state, old_path)

    new_session = store.compact_session(session.id)

    assert load_state(session_state_path(new_session.id)) == state
    assert load_state(old_path) == state


def test_compaction_with_no_state_is_a_noop(store):
    session = _create(store)

    new_session = store.compact_session(session.id)

    assert not session_state_path(new_session.id).parent.exists()
    assert load_state(session_state_path(new_session.id)) == {}
