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
    store = SessionStore(tmp_path / "store.json")
    store.agents.ensure_default(tmp_path, "default")
    return store


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

    def test_reply_records_the_session_it_came_from(self, store, runner, adapter):
        """The recipient's transcript has to say who sent this, so the UI can badge
        it instead of rendering another agent's message as the person's own.

        The origin is read in the tool, not in `reply_to_session`: the runner hops
        to the daemon loop via `run_coroutine_threadsafe`, which starts a fresh
        context, and `reply_to_session` rebinds the current session to the target.
        """
        caller = _chat(store, "caller")
        target = _chat(store, "target")
        with _calling_session(caller.id):
            session_tools.session_reply(message="how goes it", session_id=target.id)

        context = adapter.handle_message.await_args.kwargs["channel_context"]
        assert context.metadata["from_session"] == caller.id
        assert context.metadata["session_id"] == target.id

    def test_reply_from_outside_a_session_records_no_origin(self, store, runner, adapter):
        target = _chat(store, "target")
        with _calling_session(None):
            session_tools.session_reply(message="how goes it", session_id=target.id)

        assert "from_session" not in adapter.handle_message.await_args.kwargs["channel_context"].metadata

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


class TestReplyingToAFinishedSession:
    def test_a_finished_session_still_takes_the_reply(self, store, runner, adapter):
        """`session_reply` with an explicit id revives a finished session; `reply_to_session` on its own declines the turn."""
        chat = _chat(store)
        store.update_session(chat.id, status=SessionStatus.COMPLETED.value)

        result = session_tools.session_reply(message="one more thing", session_id=chat.id)

        assert result["response"] == "ack"
        assert adapter.handle_message.await_count == 1


def _fyis(store, sid):
    return [e for e in store.read_events(sid) if e["type"] == "delivery" and e["kind"] == "fyi"]


class TestSessionFyi:
    def test_lands_as_a_card_naming_the_sender_without_a_turn(self, store, runner, adapter):
        caller = _chat(store, "caller")
        target = _chat(store, "target")
        with _calling_session(caller.id):
            result = session_tools.session_fyi(message="build is green", session_id=target.id, title="CI")

        assert result == {"session_id": target.id, "status": "delivered"}
        [card] = _fyis(store, target.id)
        assert (card["message"], card["title"], card["source"]) == ("build is green", "CI", "session")
        assert card["from_session"] == caller.id
        assert adapter.handle_message.await_count == 0

    def test_is_held_while_the_target_is_mid_turn(self, store, runner, adapter):
        caller = _chat(store, "caller")
        target = _chat(store, "target")
        store.update_session(target.id, turn_in_flight=True)
        with _calling_session(caller.id):
            result = session_tools.session_fyi(message="heads up", session_id=target.id)

        assert result["status"] == "held"
        assert _fyis(store, target.id) == []
        store.update_session(target.id, turn_in_flight=False)
        runner._flush_deferred_deliveries(target.id)
        assert len(_fyis(store, target.id)) == 1
        assert adapter.handle_message.await_count == 0

    def test_a_finished_target_is_not_delivered(self, store, runner, adapter):
        caller = _chat(store, "caller")
        target = _chat(store, "target")
        store.update_session(target.id, status=SessionStatus.COMPLETED.value)
        with _calling_session(caller.id):
            result = session_tools.session_fyi(message="too late", session_id=target.id)

        assert result == {"session_id": target.id, "status": "not_delivered"}
        assert _fyis(store, target.id) == []
        assert adapter.handle_message.await_count == 0

    def test_rejects_the_callers_own_session(self, store, runner):
        chat = _chat(store)
        with _calling_session(chat.id), pytest.raises(ValueError, match="another session"):
            session_tools.session_fyi(message="note to self", session_id=chat.id)
        assert _fyis(store, chat.id) == []

    def test_reaches_a_session_by_alias(self, store, runner):
        lead = _chat(store, "lead")
        store.set_alias(lead.id, "lead")
        with _calling_session(None):
            result = session_tools.session_fyi(message="progress", session_id="name:lead")

        assert result["session_id"] == lead.id
        assert "from_session" not in _fyis(store, lead.id)[0]


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
