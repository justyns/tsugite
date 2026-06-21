"""Tests for named-route session lookup (Discord session_name config)."""

import time

import pytest
from tsugite_daemon.session_store import Session, SessionSource, SessionStatus, SessionStore


@pytest.fixture
def store(tmp_path):
    return SessionStore(tmp_path / "session_store.json")


def test_find_named_session_returns_none_when_absent(store):
    assert store.find_named_session("u1", "agent-x", "discord") is None


def test_get_or_create_named_session_creates_with_metadata(store):
    s = store.get_or_create_named_session("u1", "agent-x", "discord")
    assert s.metadata.get("session_name") == "discord"
    assert s.user_id == "u1"
    assert s.agent == "agent-x"


def test_get_or_create_named_session_idempotent(store):
    s1 = store.get_or_create_named_session("u1", "agent-x", "discord")
    s2 = store.get_or_create_named_session("u1", "agent-x", "discord")
    assert s1.id == s2.id


def test_named_session_isolated_per_user(store):
    s1 = store.get_or_create_named_session("u1", "agent-x", "discord")
    s2 = store.get_or_create_named_session("u2", "agent-x", "discord")
    assert s1.id != s2.id


def test_named_session_isolated_per_name(store):
    s1 = store.get_or_create_named_session("u1", "agent-x", "discord")
    s2 = store.get_or_create_named_session("u1", "agent-x", "slack")
    assert s1.id != s2.id


def test_find_named_session_skips_finished(store):
    s = store.get_or_create_named_session("u1", "agent-x", "discord")
    store.update_session(s.id, status=SessionStatus.COMPLETED.value)
    assert store.find_named_session("u1", "agent-x", "discord") is None


def test_find_named_session_skips_superseded(store):
    s = store.get_or_create_named_session("u1", "agent-x", "discord")
    successor = Session(
        id="successor",
        agent="agent-x",
        source=SessionSource.INTERACTIVE.value,
        user_id="u1",
        metadata={"session_name": "discord"},
    )
    store.create_session(successor)
    store.update_session(s.id, superseded_by=successor.id)

    found = store.find_named_session("u1", "agent-x", "discord")
    assert found is not None
    assert found.id == "successor"


def test_find_named_session_picks_latest_by_last_active(store):
    s1 = store.get_or_create_named_session("u1", "agent-x", "discord")
    s2 = Session(
        id="newer",
        agent="agent-x",
        source=SessionSource.INTERACTIVE.value,
        user_id="u1",
        metadata={"session_name": "discord"},
    )
    store.create_session(s2)
    time.sleep(0.01)
    store.update_session(s2.id, scratchpad="bumped")

    found = store.find_named_session("u1", "agent-x", "discord")
    assert found.id == s2.id
    assert found.id != s1.id


def test_named_session_metadata_preserved_across_compaction(store):
    s = store.get_or_create_named_session("u1", "agent-x", "discord")
    new_session = store.compact_session(s.id)
    assert new_session.metadata.get("session_name") == "discord"
    found = store.find_named_session("u1", "agent-x", "discord")
    assert found is not None
    assert found.id == new_session.id
