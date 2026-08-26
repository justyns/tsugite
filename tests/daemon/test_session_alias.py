"""The alias contract: one durable routing identity per (user_id, alias)."""

import pytest
from tsugite_daemon.session_store import AliasConflictError, Session, SessionSource, SessionStatus, SessionStore


@pytest.fixture
def store(tmp_path):
    return SessionStore(tmp_path / "session_store.json")


def _plain(store, session_id, user_id="u1"):
    return store.create_session(Session(id=session_id, source=SessionSource.INTERACTIVE.value, user_id=user_id))


def test_a_second_session_cannot_claim_a_live_alias(store):
    holder = store.claim_aliased_session("daily")
    other = _plain(store, "other")

    with pytest.raises(AliasConflictError, match="daily"):
        store.set_alias(other.id, "daily")

    assert store.find_named_session("daily").id == holder.id


def test_the_metadata_path_cannot_forge_an_alias(store):
    holder = store.claim_aliased_session("daily")
    other = _plain(store, "other")

    with pytest.raises(ValueError, match="session_name"):
        store.set_metadata_bulk(other.id, {"session_name": "daily"})

    assert store.find_named_session("daily").id == holder.id


def test_the_metadata_path_cannot_delete_an_alias(store):
    holder = store.claim_aliased_session("daily")

    with pytest.raises(ValueError, match="session_name"):
        store.delete_metadata(holder.id, "session_name")

    assert store.find_named_session("daily").id == holder.id


def test_claiming_an_alias_the_session_already_holds_is_idempotent(store):
    holder = store.claim_aliased_session("daily")

    store.set_alias(holder.id, "daily")

    assert store.find_named_session("daily").id == holder.id


def test_renaming_frees_the_old_alias(store):
    holder = store.claim_aliased_session("daily")

    store.set_alias(holder.id, "weekly")

    assert store.find_named_session("weekly").id == holder.id
    assert store.find_named_session("daily") is None

    other = _plain(store, "other")
    store.set_alias(other.id, "daily")
    assert store.find_named_session("daily").id == other.id


def test_clearing_frees_the_alias(store):
    holder = store.claim_aliased_session("daily")

    store.clear_alias(holder.id)

    assert store.find_named_session("daily") is None
    other = _plain(store, "other")
    store.set_alias(other.id, "daily")
    assert store.find_named_session("daily").id == other.id


def test_an_alias_is_one_identity_for_the_whole_daemon(store):
    """One person reaches the daemon as web-anonymous, justyn, a Discord id and a
    bare scheduler user; an alias claimed under any of them resolves from them all."""
    holder = store.claim_aliased_session("daily", user_id="web-anonymous")
    other = _plain(store, "other", user_id="justyn")

    with pytest.raises(AliasConflictError, match="daily"):
        store.set_alias(other.id, "daily")

    assert store.find_named_session("daily").id == holder.id


def test_a_finished_session_releases_its_alias(store):
    holder = store.claim_aliased_session("daily")
    store.update_session(holder.id, status=SessionStatus.COMPLETED.value)

    other = _plain(store, "other")
    store.set_alias(other.id, "daily")

    assert store.find_named_session("daily").id == other.id


def test_a_superseded_session_keeps_its_alias_on_the_successor(store):
    holder = store.claim_aliased_session("daily")
    successor = store.compact_session(holder.id)

    assert store.find_named_session("daily").id == successor.id

    other = _plain(store, "other")
    with pytest.raises(AliasConflictError, match="daily"):
        store.set_alias(other.id, "daily")


def test_a_branch_does_not_inherit_the_alias(store, monkeypatch):
    class _Backend:
        def create_branch(self, session_id, at_event_id):
            return "branch-1"

    monkeypatch.setattr("tsugite_daemon.session_store.get_history_backend", lambda: _Backend())
    holder = store.claim_aliased_session("daily")

    branch = store.branch_session(holder.id, at_event_id=1)

    assert "session_name" not in branch.metadata
    assert store.find_named_session("daily").id == holder.id


@pytest.mark.parametrize(
    "bad", ["", " ", "has space", "-leading", "_leading", "with:colon", "a" * 65, "sl/ash", "daily\n"]
)
def test_a_malformed_alias_is_rejected(store, bad):
    session = _plain(store, "s1")

    with pytest.raises(ValueError):
        store.set_alias(session.id, bad)


def test_get_or_create_named_session_rejects_a_malformed_alias(store):
    with pytest.raises(ValueError):
        store.claim_aliased_session("has space")


def test_set_alias_on_an_unknown_session_raises(store):
    with pytest.raises(ValueError, match="not found"):
        store.set_alias("nope", "daily")


def test_claiming_an_alias_twice_returns_the_same_session(store):
    first = store.claim_aliased_session("daily")
    second = store.claim_aliased_session("daily")

    assert first.id == second.id


def test_find_named_session_returns_none_when_absent(store):
    assert store.find_named_session("discord") is None


class TestDmRoute:
    """Where a platform's DMs land. Per user, because two people DMing the same bot
    must not share a conversation - which is exactly why it cannot be the alias."""

    def test_the_same_route_is_a_session_per_user(self, store):
        mine = store.get_or_create_dm_session("u1", "discord")
        theirs = store.get_or_create_dm_session("u2", "discord")

        assert mine.id != theirs.id

    def test_a_route_is_idempotent_for_one_user(self, store):
        first = store.get_or_create_dm_session("u1", "discord")
        second = store.get_or_create_dm_session("u1", "discord")

        assert first.id == second.id

    def test_two_routes_are_two_sessions(self, store):
        assert store.get_or_create_dm_session("u1", "discord").id != store.get_or_create_dm_session("u1", "slack").id

    def test_a_dm_route_claims_no_alias(self, store):
        """Otherwise the second person to DM the bot collides with the first."""
        store.get_or_create_dm_session("u1", "discord")

        assert store.find_named_session("discord") is None
