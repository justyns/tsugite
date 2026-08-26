"""How a session tool names a session.

An agent can act on the conversation it is running in without knowing its id:
omitting `session_id`, or passing `"current"`, both mean "this session". Every
reference then resolves through the compaction chain, so a tool call lands on
the live session rather than one that was superseded mid-conversation.
"""

import asyncio
import inspect
import threading
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from tsugite_daemon.session_runner import SessionRunner
from tsugite_daemon.session_store import Session, SessionSource, SessionStatus, SessionStore

from tsugite.tools import sessions as session_tools

SESSION_TOOLS = (
    "rename_session",
    "session_status",
    "cancel_session",
    "session_reply",
    "session_notify",
    "session_metadata",
)


@pytest.fixture
def store(tmp_path):
    return SessionStore(tmp_path / "store.json")


@pytest.fixture
def adapter():
    a = MagicMock()
    a.agent_name = "bot"
    a.handle_message = AsyncMock(return_value="ack")
    a.session_store = MagicMock()
    a.resolve_model.return_value = "test-model"
    return a


@pytest.fixture
def runner(store, adapter):
    """The session tools bound to a runner on a background event loop."""
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    r = SessionRunner(store, adapter)
    session_tools.set_session_runner(r, loop)
    yield r
    session_tools.set_session_runner(None)
    loop.call_soon_threadsafe(loop.stop)
    thread.join(timeout=2)
    loop.close()


def _chat(store: SessionStore, sid: str = "chat1") -> Session:
    return store.create_session(Session(id=sid, source=SessionSource.INTERACTIVE.value, user_id="alice"))


def _calling_session(sid: str | None):
    return patch.object(session_tools, "get_current_session_id", return_value=sid)


class TestResolveLive:
    def test_an_unknown_session_resolves_to_nothing(self, store):
        assert store.resolve_live("nope") is None

    def test_a_session_with_no_successor_is_its_own_live_end(self, store):
        chat = _chat(store)
        resolved = store.resolve_live(chat.id)
        assert resolved is not None
        assert resolved.id == chat.id

    def test_two_compactions_resolve_to_the_last_session(self, store):
        chat = _chat(store)
        first = store.compact_session(chat.id)
        second = store.compact_session(first.id)

        resolved = store.resolve_live(chat.id)
        assert resolved is not None
        assert resolved.id == second.id

    def test_a_chain_that_dead_ends_resolves_to_nothing(self, store):
        chat = _chat(store)
        store.update_session(chat.id, superseded_by="pruned-away")
        assert store.resolve_live(chat.id) is None


class TestToolsDefaultToTheCallingSession:
    @pytest.mark.parametrize("tool_name", SESSION_TOOLS)
    def test_the_session_argument_is_optional(self, tool_name):
        """An agent that cannot name its own session has to invent one."""
        param = inspect.signature(getattr(session_tools, tool_name)).parameters["session_id"]
        assert param.default is None

    def test_rename_without_a_session_id(self, store, runner):
        chat = _chat(store)
        with _calling_session(chat.id):
            result = session_tools.rename_session(title="Groceries")

        assert result["session_id"] == chat.id
        assert store.get_session(chat.id).title == "Groceries"

    def test_rename_with_the_current_alias(self, store, runner):
        """The reported failure: the model reached for "current" and got an error."""
        chat = _chat(store)
        with _calling_session(chat.id):
            result = session_tools.rename_session(title="Groceries", session_id="current")

        assert result["title"] == "Groceries"
        assert store.get_session(chat.id).title == "Groceries"

    def test_status_without_a_session_id(self, store, runner):
        chat = _chat(store)
        with _calling_session(chat.id):
            result = session_tools.session_status()

        assert result["id"] == chat.id

    def test_cancel_with_the_current_alias(self, store, runner):
        chat = _chat(store)
        with _calling_session(chat.id):
            session_tools.cancel_session(session_id="current")

        assert store.get_session(chat.id).status == SessionStatus.CANCELLED.value

    def test_reply_without_a_session_id(self, store, runner, adapter):
        chat = _chat(store)
        with _calling_session(chat.id):
            result = session_tools.session_reply(message="carry on")

        assert result["session_id"] == chat.id
        assert result["response"] == "ack"
        assert adapter.handle_message.await_count == 1

    def test_notify_defaults_only_the_session_it_modifies(self, store, runner):
        chat = _chat(store)
        watcher = _chat(store, "watcher")
        with _calling_session(chat.id):
            result = session_tools.session_notify(notify_session=watcher.id)

        assert result["session_id"] == chat.id
        assert result["notify_sessions"] == [watcher.id]

    def test_metadata_with_the_current_alias(self, store, runner):
        chat = _chat(store)
        with _calling_session(chat.id):
            result = session_tools.session_metadata(key="type", value="ops", session_id="current")

        assert result["session_id"] == chat.id
        assert store.get_session(chat.id).metadata["type"] == "ops"

    def test_no_session_to_act_on_is_a_clear_error(self, store, runner):
        with _calling_session(None), pytest.raises(ValueError, match="session"):
            session_tools.rename_session(title="Groceries")


class TestToolsFollowCompaction:
    def test_a_bare_id_reaches_the_session_it_became(self, store, runner):
        chat = _chat(store)
        first = store.compact_session(chat.id)
        second = store.compact_session(first.id)

        result = session_tools.rename_session(title="Groceries", session_id=chat.id)

        assert result["session_id"] == second.id
        assert store.get_session(second.id).title == "Groceries"

    def test_current_reaches_the_session_the_chat_became(self, store, runner):
        """The calling context still holds the id the turn started on."""
        chat = _chat(store)
        first = store.compact_session(chat.id)
        second = store.compact_session(first.id)

        with _calling_session(chat.id):
            session_tools.session_metadata(key="topic", value="dinner", session_id="current")

        assert store.get_session(second.id).metadata["topic"] == "dinner"
        assert "topic" not in store.get_session(chat.id).metadata


class TestToolsAddressAnAlias:
    """`name:<alias>` reaches a session by its routing identity, so a worker can
    message a lead it never learned the id of."""

    def test_a_tool_reaches_the_session_holding_the_alias(self, store, runner):
        lead = _chat(store, "lead")
        store.set_alias(lead.id, "lead")
        other = _chat(store, "other")

        with _calling_session(other.id):
            result = session_tools.rename_session(title="Feature X", session_id="name:lead")

        assert result["session_id"] == lead.id
        assert store.get_session(lead.id).title == "Feature X"

    def test_an_alias_reaches_the_session_its_holder_became(self, store, runner):
        lead = _chat(store, "lead")
        store.set_alias(lead.id, "lead")
        successor = store.compact_session(lead.id)

        result = session_tools.rename_session(title="Feature X", session_id="name:lead")

        assert result["session_id"] == successor.id

    def test_replying_to_an_alias_runs_a_turn_on_its_holder(self, store, runner, adapter):
        lead = _chat(store, "lead")
        store.set_alias(lead.id, "lead")
        worker = _chat(store, "worker")

        with _calling_session(worker.id):
            result = session_tools.session_reply(message="phase one done", session_id="name:lead")

        assert result["session_id"] == lead.id
        assert adapter.handle_message.await_count == 1

    def test_notifying_an_alias_stores_the_holders_id(self, store, runner):
        lead = _chat(store, "lead")
        store.set_alias(lead.id, "lead")
        worker = _chat(store, "worker")

        with _calling_session(worker.id):
            result = session_tools.session_notify(notify_session="name:lead")

        assert result["notify_sessions"] == [lead.id]

    def test_an_unheld_alias_is_a_clear_error(self, store, runner):
        chat = _chat(store)

        with _calling_session(chat.id), pytest.raises(ValueError, match="No session holds alias"):
            session_tools.rename_session(title="Feature X", session_id="name:lead")
