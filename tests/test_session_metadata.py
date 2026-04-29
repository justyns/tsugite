"""Tests for session metadata CRUD, read-only key enforcement, channel session index,
session runner metadata methods, and session_metadata tool."""

import asyncio
import threading
from unittest.mock import MagicMock, patch

import pytest

from tsugite.daemon.session_store import (
    READ_ONLY_METADATA_KEYS,
    Session,
    SessionSource,
    SessionStore,
)


@pytest.fixture
def store(tmp_path):
    return SessionStore(tmp_path / "session_store.json")


@pytest.fixture
def session_in_store(store):
    session = Session(id="test-1", agent="odyn", source=SessionSource.BACKGROUND.value)
    store.create_session(session)
    return session


# ── Metadata CRUD ──


class TestSessionMetadata:
    def test_set_metadata_key(self, store, session_in_store):
        result = store.set_metadata(session_in_store.id, "task", "https://tasks.example/42")
        assert result.metadata["task"] == "https://tasks.example/42"

    def test_set_metadata_overwrites(self, store, session_in_store):
        store.set_metadata(session_in_store.id, "status_text", "investigating")
        result = store.set_metadata(session_in_store.id, "status_text", "PR opened")
        assert result.metadata["status_text"] == "PR opened"

    def test_delete_metadata_key(self, store, session_in_store):
        store.set_metadata(session_in_store.id, "notes", "some notes")
        result = store.delete_metadata(session_in_store.id, "notes")
        assert "notes" not in result.metadata

    def test_delete_nonexistent_key_raises(self, store, session_in_store):
        with pytest.raises(ValueError, match="not found in metadata"):
            store.delete_metadata(session_in_store.id, "nonexistent")

    def test_set_metadata_bulk(self, store, session_in_store):
        result = store.set_metadata_bulk(
            session_in_store.id,
            {
                "type": "code",
                "task": "https://tasks.example/1",
                "pr": "https://github.com/org/repo/pull/1",
            },
        )
        assert result.metadata["type"] == "code"
        assert result.metadata["task"] == "https://tasks.example/1"
        assert result.metadata["pr"] == "https://github.com/org/repo/pull/1"

    def test_set_metadata_updates_last_active(self, store, session_in_store):
        old_active = session_in_store.last_active
        store.set_metadata(session_in_store.id, "type", "ops")
        updated = store.get_session(session_in_store.id)
        assert updated.last_active >= old_active

    def test_metadata_persists_across_reload(self, tmp_path):
        store1 = SessionStore(tmp_path / "store.json")
        s = Session(id="persist-test", agent="test")
        store1.create_session(s)
        store1.set_metadata("persist-test", "type", "research")
        store1.flush()

        store2 = SessionStore(tmp_path / "store.json")
        loaded = store2.get_session("persist-test")
        assert loaded.metadata["type"] == "research"

    def test_set_metadata_nonexistent_session_raises(self, store):
        with pytest.raises(ValueError, match="not found"):
            store.set_metadata("nonexistent", "key", "value")

    def test_arbitrary_keys_allowed(self, store, session_in_store):
        result = store.set_metadata(session_in_store.id, "custom_field", "custom_value")
        assert result.metadata["custom_field"] == "custom_value"


# ── Read-Only Key Enforcement ──


class TestReadOnlyKeys:
    @pytest.mark.parametrize("key", sorted(READ_ONLY_METADATA_KEYS))
    def test_set_read_only_key_raises(self, store, session_in_store, key):
        with pytest.raises(ValueError, match="read-only"):
            store.set_metadata(session_in_store.id, key, "forbidden")

    @pytest.mark.parametrize("key", sorted(READ_ONLY_METADATA_KEYS))
    def test_delete_read_only_key_raises(self, store, session_in_store, key):
        session_in_store.metadata[key] = "existing"
        with pytest.raises(ValueError, match="read-only"):
            store.delete_metadata(session_in_store.id, key)

    def test_bulk_with_read_only_key_rejects_all(self, store, session_in_store):
        with pytest.raises(ValueError, match="read-only"):
            store.set_metadata_bulk(
                session_in_store.id,
                {
                    "type": "code",
                    "source": "forbidden",
                },
            )
        updated = store.get_session(session_in_store.id)
        assert "type" not in updated.metadata


# ── Channel Session Index ──


class TestChannelSessionIndex:
    def test_create_channel_session(self, store):
        session = store.get_or_create_channel_session("channel-123", "odyn", "user-1")
        assert session.agent == "odyn"
        assert session.metadata.get("channel_id") == "channel-123"
        assert session.source == SessionSource.INTERACTIVE.value

    def test_reuse_existing_channel_session(self, store):
        s1 = store.get_or_create_channel_session("channel-123", "odyn", "user-1")
        s2 = store.get_or_create_channel_session("channel-123", "odyn", "user-2")
        assert s1.id == s2.id

    def test_different_channel_different_session(self, store):
        s1 = store.get_or_create_channel_session("channel-1", "odyn", "user-1")
        s2 = store.get_or_create_channel_session("channel-2", "odyn", "user-1")
        assert s1.id != s2.id

    def test_different_agent_different_session(self, store):
        s1 = store.get_or_create_channel_session("channel-1", "odyn", "user-1")
        s2 = store.get_or_create_channel_session("channel-1", "helper", "user-1")
        assert s1.id != s2.id

    def test_find_by_channel(self, store):
        store.get_or_create_channel_session("channel-99", "odyn", "user-1")
        found = store.find_by_channel("channel-99", "odyn")
        assert found is not None
        assert found.metadata["channel_id"] == "channel-99"

    def test_find_by_channel_missing(self, store):
        assert store.find_by_channel("nonexistent", "odyn") is None

    def test_channel_index_rebuilt_on_load(self, tmp_path):
        store1 = SessionStore(tmp_path / "store.json")
        store1.get_or_create_channel_session("ch-1", "odyn", "user-1")
        store1.flush()

        store2 = SessionStore(tmp_path / "store.json")
        found = store2.find_by_channel("ch-1", "odyn")
        assert found is not None


# ── Session Runner Metadata ──


class TestSessionRunnerMetadata:
    @pytest.fixture
    def runner_deps(self, tmp_path):
        store = SessionStore(tmp_path / "store.json")
        session = Session(id="runner-test", agent="odyn", source=SessionSource.BACKGROUND.value)
        store.create_session(session)
        event_bus = MagicMock()
        return store, event_bus, session

    def test_update_metadata_emits_event(self, runner_deps):
        from tsugite.daemon.session_runner import SessionRunner

        store, event_bus, session = runner_deps
        runner = SessionRunner(store, {}, event_bus=event_bus)
        runner.update_session_metadata(session.id, {"task": "https://example.com/1"})

        event_bus.emit.assert_called_once()
        call_args = event_bus.emit.call_args
        assert call_args[0][0] == "session_update"
        assert call_args[0][1]["action"] == "metadata_updated"
        assert call_args[0][1]["id"] == session.id
        assert "task" in call_args[0][1]["metadata"]

    def test_delete_metadata_emits_event(self, runner_deps):
        from tsugite.daemon.session_runner import SessionRunner

        store, event_bus, session = runner_deps
        store.set_metadata(session.id, "notes", "temp")
        runner = SessionRunner(store, {}, event_bus=event_bus)
        runner.delete_session_metadata(session.id, "notes")

        event_bus.emit.assert_called_once()
        call_args = event_bus.emit.call_args
        assert call_args[0][0] == "session_update"
        assert call_args[0][1]["action"] == "metadata_updated"


# ── Session Metadata Tool ──


class TestSessionMetadataTool:
    @pytest.fixture(autouse=True)
    def _setup_runner(self, tmp_path):
        """Set up a SessionRunner with a running event loop for tool calls."""
        from tsugite.daemon.session_runner import SessionRunner
        from tsugite.tools.sessions import set_session_runner

        self.store = SessionStore(tmp_path / "store.json")
        self.loop = asyncio.new_event_loop()
        self.runner = SessionRunner(self.store, {})

        # Run the loop in a background thread so run_coroutine_threadsafe works
        self._thread = threading.Thread(target=self.loop.run_forever, daemon=True)
        self._thread.start()
        set_session_runner(self.runner, self.loop)
        yield
        set_session_runner(None)
        self.loop.call_soon_threadsafe(self.loop.stop)
        self._thread.join(timeout=2)
        self.loop.close()

    def _create_session(self, sid="tool-test"):
        session = Session(id=sid, agent="odyn", source=SessionSource.BACKGROUND.value)
        self.store.create_session(session)
        return session

    def test_set_key_on_current_session(self):
        from tsugite.tools.sessions import session_metadata

        self._create_session("tool-test")
        with patch("tsugite.tools.sessions.get_current_session_id", return_value="tool-test"):
            result = session_metadata(key="type", value="code")
        assert result["session_id"] == "tool-test"
        assert result["metadata"]["type"] == "code"

    def test_delete_key(self):
        from tsugite.tools.sessions import session_metadata

        self._create_session("tool-del")
        self.store.set_metadata("tool-del", "notes", "to be deleted")
        result = session_metadata(key="notes", value=None, session_id="tool-del")
        assert "notes" not in result["metadata"]

    def test_read_only_key_returns_error(self):
        from tsugite.tools.sessions import session_metadata

        self._create_session("tool-ro")
        result = session_metadata(key="source", value="forbidden", session_id="tool-ro")
        assert "error" in result
        assert "read-only" in result["error"]

    def test_explicit_session_id(self):
        from tsugite.tools.sessions import session_metadata

        self._create_session("tool-explicit")
        result = session_metadata(key="pr", value="https://github.com/pr/1", session_id="tool-explicit")
        assert result["session_id"] == "tool-explicit"
        assert result["metadata"]["pr"] == "https://github.com/pr/1"

    def test_no_active_session_returns_error(self):
        from tsugite.tools.sessions import session_metadata

        with patch("tsugite.tools.sessions.get_current_session_id", return_value=None):
            result = session_metadata(key="type", value="code")
        assert "error" in result


# ── Topic Length Cap ──


class TestTopicLengthCap:
    """Topic is meant for IRC-style short context. Cap at 160 chars matches the
    de facto Freenode/EFnet/IRCnet/Undernet TOPICLEN of the era when "channel
    topic" took its modern meaning. Longer context belongs in a forked agent file.
    """

    def test_topic_at_cap_succeeds(self, store, session_in_store):
        topic = "x" * 160
        result = store.set_metadata(session_in_store.id, "topic", topic)
        assert result.metadata["topic"] == topic

    def test_topic_over_cap_rejected(self, store, session_in_store):
        with pytest.raises(ValueError, match="160"):
            store.set_metadata(session_in_store.id, "topic", "x" * 161)

    def test_topic_over_cap_via_bulk_rejected(self, store, session_in_store):
        with pytest.raises(ValueError, match="160"):
            store.set_metadata_bulk(session_in_store.id, {"topic": "x" * 161})
        updated = store.get_session(session_in_store.id)
        assert "topic" not in updated.metadata

    def test_other_keys_uncapped(self, store, session_in_store):
        long_value = "x" * 500
        result = store.set_metadata(session_in_store.id, "task", long_value)
        assert result.metadata["task"] == long_value

    def test_non_string_topic_rejected(self, store, session_in_store):
        with pytest.raises(ValueError, match="string"):
            store.set_metadata(session_in_store.id, "topic", 123)
