import asyncio
import contextvars
import json

import pytest
from tsugite_daemon.agent_store import Agent
from tsugite_daemon.attention_store import SOURCE_ASK, SOURCE_ROOM
from tsugite_daemon.gateway import terminal_state_callback
from tsugite_daemon.job_store import Job, JobState, JobStore
from tsugite_daemon.jobs_orchestrator import JobsOrchestrator
from tsugite_daemon.room_store import DEFAULT_MAX_ROUNDS
from tsugite_daemon.rooms import RoomConflictError
from tsugite_daemon.session_runner import SessionRunner, get_current_session_id, report_send_failure
from tsugite_daemon.session_store import Session, SessionStatus, SessionStore
from tsugite_pty.terminal_store import TerminalSession, TerminalSessionStore

from tsugite.cancellation import is_cancelled
from tsugite.history.ui_events import split_injected_context
from tsugite.interaction import NonInteractiveBackend, get_interaction_backend, set_interaction_backend
from tsugite.tools import jobs as jobs_tools
from tsugite.tools import sessions as sessions_tools


class _Adapter:
    def __init__(self, store):
        self.store = store
        self.calls: list[tuple[str, str]] = []
        self.hooks: dict[str, object] = {}

    def agent_of(self, session_id):
        return self.store.agents.effective_id(self.store.get_session(session_id).agent_id)

    async def handle_message(self, user_id, message, channel_context, custom_logger=None):
        session_id = channel_context.metadata["conv_id_override"]
        self.store.begin_turn(session_id)
        try:
            self.store.append_event(session_id, {"type": "user_input", "text": message})
            self.calls.append((self.agent_of(session_id), message))
            hook = self.hooks.pop(session_id, None)
            if hook is not None:
                await hook()
            return f"{self.agent_of(session_id)} replies"
        finally:
            self.store.end_turn(session_id)


@pytest.fixture
def store(tmp_path):
    store = SessionStore(tmp_path / "state" / "store.json")
    store.agents.ensure_default(tmp_path / "main", "default")
    for agent_id in ("homelab", "research"):
        store.agents.create(Agent(id=agent_id, name=agent_id.title(), workspace=str(tmp_path), template="default"))
    return store


@pytest.fixture
def adapter(store):
    return _Adapter(store)


class _Bus:
    def __init__(self):
        self.frames: list[tuple[str, dict]] = []

    def emit(self, kind, payload):
        self.frames.append((kind, payload))

    def room_updates(self):
        return [p["room"] for kind, p in self.frames if kind == "session_update" and p["action"] == "room_updated"]


@pytest.fixture
def bus():
    return _Bus()


@pytest.fixture
def runner(store, adapter, bus):
    return SessionRunner(store, adapter, event_bus=bus)


@pytest.fixture
def rooms(runner):
    return runner.rooms


@pytest.fixture
def lead_chat(store):
    return store.create_session(Session(id="", user_id="web-anonymous"))


@pytest.fixture
def room(rooms, lead_chat):
    room = rooms.create("PROJ-123", user_id="web-anonymous", lead_session_id=lead_chat.id)
    rooms.invite(room.id, "homelab", brief="Disks are filling up on the NAS.")
    rooms.invite(room.id, "research")
    return room


async def _settle(rooms, room_id):
    while (driver := rooms._drivers.get(room_id)) is not None and not driver.done():
        await asyncio.wait_for(asyncio.shield(driver), 5)


def _messages(store, room_id):
    return [(e["author"], e["content"]) for e in store.read_events(room_id) if e["type"] == "room_message"]


def _context(prompt):
    blocks, _rest = split_injected_context(prompt)
    return next(b["body"] for b in blocks if b["tag"] == "room_context")


async def _handoff_from_lead_turn(rooms, store, lead_chat, room, handoffs, mode="sequential"):
    store.begin_turn(lead_chat.id)
    result = rooms.handoff(room.id, handoffs, mode)
    store.end_turn(lead_chat.id)
    await _settle(rooms, room.id)
    return result


class TestHandoff:
    @pytest.mark.asyncio
    async def test_sequential_runs_in_order_then_wakes_the_lead_once(self, rooms, store, adapter, lead_chat, room):
        await _handoff_from_lead_turn(
            rooms,
            store,
            lead_chat,
            room,
            [{"agent": "homelab", "message": "check the disks"}, {"agent": "research", "message": "find the docs"}],
        )

        assert [agent for agent, _ in adapter.calls] == ["homelab", "research", "main"]
        assert "homelab replies" in _context(adapter.calls[1][1])
        assert _messages(store, room.id) == [
            ("main", "@homelab check the disks"),
            ("main", "@research find the docs"),
            ("homelab", "homelab replies"),
            ("research", "research replies"),
            ("main", "main replies"),
        ]
        assert store.rooms.get(room.id).rounds_used == 2

    @pytest.mark.asyncio
    async def test_parallel_turns_start_from_the_same_transcript(self, rooms, store, adapter, lead_chat, room):
        await _handoff_from_lead_turn(
            rooms,
            store,
            lead_chat,
            room,
            [{"agent": "homelab", "message": "a"}, {"agent": "research", "message": "b"}],
            mode="parallel",
        )

        prompts = dict(adapter.calls)
        assert "replies" not in _context(prompts["homelab"])
        assert "replies" not in _context(prompts["research"])
        lead_context = _context(prompts["main"])
        assert "homelab replies" in lead_context and "research replies" in lead_context
        assert [agent for agent, _ in adapter.calls][-1] == "main"

    @pytest.mark.asyncio
    async def test_reply_posts_with_a_private_ref_to_the_participant_turn(self, rooms, store, lead_chat, room):
        await _handoff_from_lead_turn(rooms, store, lead_chat, room, [{"agent": "homelab", "message": "go"}])

        reply = next(e for e in store.read_events(room.id) if e.get("author") == "homelab")
        homelab_session = rooms.get(room.id).participant("homelab")
        turn = next(e for e in store.read_events(homelab_session.session_id) if e["id"] == reply["private_ref"])
        assert turn["type"] == "user_input"
        assert reply["participant_session_id"] == homelab_session.session_id

    @pytest.mark.asyncio
    async def test_lead_handoff_from_a_router_turn_continues_the_drive(self, rooms, store, adapter, lead_chat, room):
        async def lead_hands_off_again():
            rooms.handoff(room.id, [{"agent": "research", "message": "now you"}], "sequential")

        adapter.hooks[lead_chat.id] = lead_hands_off_again
        await _handoff_from_lead_turn(rooms, store, lead_chat, room, [{"agent": "homelab", "message": "go"}])

        assert [agent for agent, _ in adapter.calls] == ["homelab", "main", "research", "main"]

    @pytest.mark.parametrize(
        "handoffs,mode,error",
        [
            ([{"agent": "ghost", "message": "x"}], "sequential", "not in room"),
            ([{"agent": "main", "message": "x"}], "sequential", "lead"),
            ([{"agent": "homelab", "message": ""}], "sequential", "message"),
            ([], "sequential", "at least one"),
            ([{"agent": "homelab", "message": "x"}], "round-robin", "mode"),
        ],
    )
    def test_rejects_a_bad_handoff(self, rooms, lead_chat, room, handoffs, mode, error):
        with pytest.raises(ValueError, match=error):
            rooms.handoff(room.id, handoffs, mode)

    def test_a_muted_participant_takes_no_handoff(self, rooms, lead_chat, room):
        rooms.set_muted(room.id, "homelab", True)

        with pytest.raises(ValueError, match="muted"):
            rooms.handoff(room.id, [{"agent": "homelab", "message": "x"}], "sequential")


class TestUserMessages:
    @pytest.mark.asyncio
    async def test_unaddressed_message_goes_to_the_lead(self, rooms, store, adapter, room):
        rooms.post_user_message(room.id, "status?")
        await _settle(rooms, room.id)

        assert [agent for agent, _ in adapter.calls] == ["main"]
        assert "status?" in _context(adapter.calls[0][1])
        assert store.rooms.get(room.id).rounds_used == 0

    @pytest.mark.asyncio
    async def test_mention_runs_that_participant_once_then_the_lead(self, rooms, store, adapter, room):
        rooms.post_user_message(room.id, "@research what do you think?")
        await _settle(rooms, room.id)

        assert [agent for agent, _ in adapter.calls] == ["research", "main"]
        assert store.rooms.get(room.id).rounds_used == 1

    @pytest.mark.asyncio
    async def test_mention_runs_a_muted_participant(self, rooms, adapter, room):
        rooms.set_muted(room.id, "homelab", True)

        rooms.post_user_message(room.id, "@homelab you there?")
        await _settle(rooms, room.id)

        assert [agent for agent, _ in adapter.calls] == ["homelab", "main"]

    @pytest.mark.asyncio
    async def test_user_message_mid_batch_runs_ahead_of_queued_handoffs(self, rooms, store, adapter, lead_chat, room):
        homelab = rooms.get(room.id).participant("homelab")

        async def user_interrupts():
            rooms.post_user_message(room.id, "hold on")

        adapter.hooks[homelab.session_id] = user_interrupts
        await _handoff_from_lead_turn(
            rooms,
            store,
            lead_chat,
            room,
            [{"agent": "homelab", "message": "a"}, {"agent": "research", "message": "b"}],
        )

        assert [agent for agent, _ in adapter.calls] == ["homelab", "main", "research", "main"]


class TestRoomContext:
    @pytest.mark.asyncio
    async def test_context_holds_roster_untrusted_messages_and_identity(self, rooms, adapter, room):
        rooms.post_user_message(room.id, "@homelab look at sda")
        await _settle(rooms, room.id)

        context = _context(adapter.calls[0][1])
        assert 'agent="research"' in context and 'role="lead"' in context
        assert "Disks are filling up on the NAS." in context
        assert '<message id="' in context and 'author="user"' in context
        assert "untrusted" in context
        identity = "You are @homelab in room 'PROJ-123'. Reply only as yourself. The lead routes the next turn."
        assert context.rstrip().endswith(f"<identity>{identity}</identity>")

    @pytest.mark.asyncio
    async def test_other_agents_are_marked_untrusted_and_the_user_is_not(self, rooms, adapter, room):
        rooms.post_user_message(room.id, "@homelab first")
        await _settle(rooms, room.id)
        rooms.post_user_message(room.id, "@research second")
        await _settle(rooms, room.id)

        context = _context(adapter.calls[2][1])
        assert 'author="homelab" untrusted="true"' in context
        assert 'author="user" untrusted' not in context

    @pytest.mark.asyncio
    async def test_cursor_advances_and_the_brief_shows_once(self, rooms, adapter, room):
        rooms.post_user_message(room.id, "@homelab first")
        await _settle(rooms, room.id)
        rooms.post_user_message(room.id, "@homelab second")
        await _settle(rooms, room.id)

        second = _context(adapter.calls[2][1])
        assert "second" in second
        assert "first" not in second
        assert "Disks are filling up" not in second

    def test_join_shows_only_the_last_messages(self, rooms, store, room, monkeypatch):
        monkeypatch.setattr("tsugite_daemon.rooms.JOIN_HISTORY", 2)
        for text in ("one", "two", "three"):
            rooms.post_message(room.id, text)
        store.agents.create(Agent(id="late", name="Late", workspace="/w", template="default"))

        rooms.invite(room.id, "late")

        late = rooms.get(room.id).participant("late")
        shown = [
            e["content"] for e in store.read_events(room.id) if e["type"] == "room_message" and e["id"] > late.cursor
        ]
        assert shown == ["two", "three"]


class TestRoundCap:
    @pytest.mark.asyncio
    async def test_cap_skips_the_participant_and_still_wakes_the_lead(self, rooms, store, adapter, room):
        store.rooms.update(room.id, max_rounds=1)

        rooms.post_user_message(room.id, "@homelab @research hi")
        await _settle(rooms, room.id)

        assert [agent for agent, _ in adapter.calls] == ["homelab", "main"]
        [card] = [e for e in store.read_events(room.id) if e["type"] == "room_card"]
        assert card["kind"] == "round_cap"
        [record] = store.attention.open_records(room.id)
        assert (record.source, record.kind) == (SOURCE_ROOM, "round_cap")

    @pytest.mark.asyncio
    async def test_raising_the_cap_clears_the_attention(self, rooms, store, room):
        store.rooms.update(room.id, max_rounds=1)
        rooms.post_user_message(room.id, "@homelab @research hi")
        await _settle(rooms, room.id)

        rooms.set_max_rounds(room.id, 5)

        assert store.rooms.get(room.id).max_rounds == 5
        assert store.attention.open_records(room.id) == []

    @pytest.mark.asyncio
    async def test_a_user_message_resets_the_rounds_and_clears_the_attention(self, rooms, store, adapter, bus, room):
        store.rooms.update(room.id, max_rounds=1)
        rooms.post_user_message(room.id, "@homelab @research hi")
        await _settle(rooms, room.id)
        adapter.calls.clear()

        rooms.post_user_message(room.id, "@research go on")

        assert bus.room_updates()[-1]["rounds_used"] == 0
        assert store.attention.open_records(room.id) == []
        await _settle(rooms, room.id)
        assert [agent for agent, _ in adapter.calls] == ["research", "main"]
        assert store.rooms.get(room.id).rounds_used == 1

    def test_cap_must_be_positive(self, rooms, room):
        with pytest.raises(ValueError, match="max_rounds"):
            rooms.set_max_rounds(room.id, 0)


class TestStop:
    @pytest.mark.asyncio
    async def test_stop_cancels_the_running_turn_and_queued_handoffs(self, rooms, store, adapter, lead_chat, room):
        started = asyncio.Event()
        seen = {}

        async def block():
            seen["session"] = get_current_session_id()
            started.set()
            try:
                await asyncio.sleep(10)
            finally:
                seen["cancel_bound"] = is_cancelled()

        homelab = rooms.get(room.id).participant("homelab")
        adapter.hooks[homelab.session_id] = block
        store.begin_turn(lead_chat.id)
        rooms.handoff(
            room.id,
            [{"agent": "homelab", "message": "a"}, {"agent": "research", "message": "b"}],
            "sequential",
        )
        store.end_turn(lead_chat.id)
        await asyncio.wait_for(started.wait(), 5)

        rooms.stop(room.id)
        await asyncio.sleep(0)
        await asyncio.gather(rooms._drivers.get(room.id) or asyncio.sleep(0), return_exceptions=True)

        assert [agent for agent, _ in adapter.calls] == ["homelab"]
        assert seen == {"session": homelab.session_id, "cancel_bound": True}
        assert not rooms._steps.get(room.id)


class TestRoomTools:
    @pytest.fixture(autouse=True)
    def bound(self, runner, monkeypatch, lead_chat):
        monkeypatch.setattr(sessions_tools, "_session_runner", runner)
        monkeypatch.setattr(sessions_tools, "_call", lambda fn, *a, **kw: fn(*a, **kw))
        monkeypatch.setattr(sessions_tools, "get_current_session_id", lambda: lead_chat.id)

    def test_open_makes_the_calling_session_the_lead(self, store, lead_chat):
        result = sessions_tools.room_open("PROJ-9", lead_self=True)

        room = store.rooms.get(result["id"])
        [lead] = room.participants
        assert (lead.agent_id, lead.session_id, lead.lead) == ("main", lead_chat.id, True)

    def test_open_sets_the_round_cap(self, store):
        default = store.rooms.get(sessions_tools.room_open("PROJ-9", lead_self=True)["id"])
        capped = store.rooms.get(sessions_tools.room_open("PROJ-10", lead_self=True, max_rounds=40)["id"])

        assert (default.max_rounds, capped.max_rounds) == (DEFAULT_MAX_ROUNDS, 40)

    def test_invite_handoff_and_close_resolve_the_room_from_the_session(self, rooms, store, lead_chat):
        room_id = sessions_tools.room_open("PROJ-9", lead_self=True)["id"]

        sessions_tools.room_invite("homelab", "Check the NAS.")
        store.begin_turn(lead_chat.id)
        queued = sessions_tools.room_handoff([{"agent": "homelab", "message": "go"}])
        closed = sessions_tools.room_close()

        assert {p["agent_id"]: p["working"] for p in queued["participants"]}["homelab"] is True
        assert rooms.get(room_id).participant("homelab").brief == "Check the NAS."
        assert closed["status"] == SessionStatus.COMPLETED.value
        assert store.get_session(lead_chat.id).status == SessionStatus.ACTIVE.value
        assert not rooms._steps.get(room_id)

    def test_tools_need_a_room(self):
        with pytest.raises(ValueError, match="room"):
            sessions_tools.room_handoff([{"agent": "homelab", "message": "go"}])

    def test_only_the_lead_invites(self, rooms, monkeypatch):
        room_id = sessions_tools.room_open("PROJ-9", lead_self=True)["id"]
        sessions_tools.room_invite("homelab", "hi")
        homelab = rooms.get(room_id).participant("homelab")
        monkeypatch.setattr(sessions_tools, "get_current_session_id", lambda: homelab.session_id)

        with pytest.raises(ValueError, match="lead"):
            sessions_tools.room_invite("research", "hi")


def _agents_called(adapter):
    return [agent for agent, _ in adapter.calls]


def _cards(store, room_id):
    return [e for e in store.read_events(room_id) if e["type"] == "room_card"]


def _session_of(rooms, room, agent_id):
    return rooms.get(room.id).participant(agent_id).session_id


def _bind_tools(monkeypatch, runner):
    """`_call` runs on the daemon loop, outside the turn's context."""
    monkeypatch.setattr(sessions_tools, "_session_runner", runner)
    monkeypatch.setattr(sessions_tools, "_call", lambda fn, *a, **kw: contextvars.Context().run(fn, *a, **kw))


class TestBusySessions:
    @pytest.mark.asyncio
    async def test_lead_turn_waits_for_the_lead_chat_turn_to_end(self, rooms, store, adapter, lead_chat, room):
        store.begin_turn(lead_chat.id)
        rooms.post_user_message(room.id, "status?")
        await asyncio.sleep(0.05)
        assert adapter.calls == []

        store.end_turn(lead_chat.id)
        await _settle(rooms, room.id)

        assert _agents_called(adapter) == ["main"]

    @pytest.mark.asyncio
    async def test_participant_turn_waits_for_its_session(self, rooms, store, adapter, room):
        homelab = _session_of(rooms, room, "homelab")
        store.begin_turn(homelab)
        rooms.post_user_message(room.id, "@homelab hi")
        await asyncio.sleep(0.05)
        assert adapter.calls == []

        store.end_turn(homelab)
        await _settle(rooms, room.id)

        assert _agents_called(adapter) == ["homelab", "main"]

    @pytest.mark.asyncio
    async def test_queued_chat_messages_go_first(self, rooms, store, adapter, lead_chat, room):
        store.queue_message(lead_chat.id, {"id": "q-1", "text": "queued"})
        rooms.post_user_message(room.id, "status?")
        await asyncio.sleep(0.05)
        assert adapter.calls == []

        store.drop_queued_message(lead_chat.id, "q-1")
        await _settle(rooms, room.id)

        assert _agents_called(adapter) == ["main"]


class TestRouterTurnContext:
    @pytest.mark.asyncio
    async def test_turns_do_not_inherit_the_kicking_context(self, rooms, adapter, room):
        seen = {}

        async def record_backend():
            seen["backend"] = get_interaction_backend()

        adapter.hooks[_session_of(rooms, room, "homelab")] = record_backend
        set_interaction_backend(object())
        try:
            rooms.post_user_message(room.id, "@homelab hi")
        finally:
            set_interaction_backend(None)
        await _settle(rooms, room.id)

        assert isinstance(seen["backend"], NonInteractiveBackend)

    @pytest.mark.asyncio
    async def test_tools_use_the_room_of_the_router_turn(self, rooms, runner, adapter, lead_chat, room, monkeypatch):
        _bind_tools(monkeypatch, runner)
        rooms.remove(room.id, "research")
        newer = rooms.create("Newer", user_id="web-anonymous", lead_session_id=lead_chat.id)

        async def lead_invites():
            sessions_tools.room_invite("research", "Read up.")

        adapter.hooks[lead_chat.id] = lead_invites
        rooms.post_user_message(room.id, "status?")
        await _settle(rooms, room.id)

        assert rooms.get(room.id).participant("research") is not None
        assert rooms.get(newer.id).participant("research") is None
        assert f'<room_context id="{room.id}"' in adapter.calls[0][1]


class TestTurnOutcomes:
    @pytest.mark.asyncio
    async def test_a_failing_turn_posts_a_card_and_the_batch_continues(self, rooms, store, adapter, lead_chat, room):
        async def boom():
            raise RuntimeError("disk on fire")

        adapter.hooks[_session_of(rooms, room, "homelab")] = boom
        await _handoff_from_lead_turn(
            rooms,
            store,
            lead_chat,
            room,
            [{"agent": "homelab", "message": "a"}, {"agent": "research", "message": "b"}],
        )

        [card] = _cards(store, room.id)
        assert (card["kind"], card["agent_id"], card["text"]) == ("turn_failed", "homelab", "disk on fire")
        assert _agents_called(adapter) == ["homelab", "research", "main"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("empty", ["", "  \n"])
    async def test_an_empty_turn_posts_a_card_instead_of_a_message(
        self, rooms, store, adapter, lead_chat, room, monkeypatch, empty
    ):
        handle_message = adapter.handle_message

        async def silent_homelab(user_id, message, channel_context, custom_logger=None):
            reply = await handle_message(user_id, message, channel_context, custom_logger)
            return empty if adapter.calls[-1][0] == "homelab" else reply

        monkeypatch.setattr(adapter, "handle_message", silent_homelab)
        await _handoff_from_lead_turn(rooms, store, lead_chat, room, [{"agent": "homelab", "message": "a"}])

        [card] = _cards(store, room.id)
        assert (card["kind"], card["agent_id"], card["text"]) == (
            "turn_failed",
            "homelab",
            "the turn produced no message",
        )
        assert "homelab" not in [author for author, _ in _messages(store, room.id)]
        assert _agents_called(adapter) == ["homelab", "main"]

    @pytest.mark.asyncio
    async def test_a_lead_turn_that_only_hands_off_posts_no_card(
        self, rooms, store, adapter, lead_chat, room, monkeypatch
    ):
        handle_message = adapter.handle_message

        async def silent_lead(user_id, message, channel_context, custom_logger=None):
            reply = await handle_message(user_id, message, channel_context, custom_logger)
            return "" if adapter.calls[-1][0] == "main" else reply

        async def lead_hands_off():
            rooms.handoff(room.id, [{"agent": "homelab", "message": "a"}], "sequential")

        monkeypatch.setattr(adapter, "handle_message", silent_lead)
        adapter.hooks[lead_chat.id] = lead_hands_off
        rooms.post_user_message(room.id, "check the NAS")
        await _settle(rooms, room.id)

        assert _cards(store, room.id) == []
        assert _agents_called(adapter) == ["main", "homelab", "main"]

    @pytest.mark.asyncio
    async def test_a_finished_participant_session_posts_a_card(self, rooms, store, adapter, room):
        store.update_session(_session_of(rooms, room, "homelab"), status=SessionStatus.COMPLETED.value)

        rooms.post_user_message(room.id, "@homelab hi")
        await _settle(rooms, room.id)

        assert [c["kind"] for c in _cards(store, room.id)] == ["turn_failed"]
        assert _agents_called(adapter) == ["main"]

    @pytest.mark.asyncio
    async def test_a_finished_background_lead_is_revived(self, rooms, store, adapter):
        lead = store.create_session(Session(id="", source="background", status=SessionStatus.COMPLETED.value))
        room = rooms.create("Bg", user_id=None, lead_session_id=lead.id)

        rooms.post_user_message(room.id, "status?")
        await _settle(rooms, room.id)

        assert _agents_called(adapter) == ["main"]
        assert _messages(store, room.id)[-1] == ("main", "main replies")

    @pytest.mark.asyncio
    async def test_cancel_turn_reaches_a_router_turn(self, rooms, runner, store, adapter, room):
        started = asyncio.Event()
        seen = {}

        async def block():
            started.set()
            try:
                await asyncio.sleep(10)
            finally:
                seen["cancelled"] = is_cancelled()

        homelab = _session_of(rooms, room, "homelab")
        adapter.hooks[homelab] = block
        rooms.post_user_message(room.id, "@homelab hi")
        await asyncio.wait_for(started.wait(), 5)

        assert runner.cancel_turn(homelab) is True
        await _settle(rooms, room.id)

        assert seen == {"cancelled": True}
        assert ("homelab", "homelab replies") not in _messages(store, room.id)

    @pytest.mark.asyncio
    async def test_handoff_text_reaches_the_participant_once(self, rooms, store, adapter, lead_chat, room):
        await _handoff_from_lead_turn(rooms, store, lead_chat, room, [{"agent": "homelab", "message": "check sda"}])

        prompt = adapter.calls[0][1]
        assert "check sda" not in _context(prompt)
        assert prompt.endswith("check sda")


class TestMembershipAtRunTime:
    @pytest.mark.asyncio
    async def test_muted_after_the_handoff_skips_the_turn(self, rooms, store, adapter, lead_chat, room):
        store.begin_turn(lead_chat.id)
        rooms.handoff(room.id, [{"agent": "homelab", "message": "a"}], "sequential")
        rooms.set_muted(room.id, "homelab", True)
        store.end_turn(lead_chat.id)
        await _settle(rooms, room.id)

        assert _agents_called(adapter) == ["main"]
        assert [(c["kind"], c["agent_id"]) for c in _cards(store, room.id)] == [("turn_skipped", "homelab")]
        assert store.rooms.get(room.id).rounds_used == 0

    @pytest.mark.asyncio
    async def test_removed_while_its_handoff_is_queued_skips_the_turn(self, rooms, store, adapter, lead_chat, room):
        store.begin_turn(lead_chat.id)
        rooms.handoff(room.id, [{"agent": "homelab", "message": "a"}], "sequential")
        rooms.remove(room.id, "homelab")
        store.end_turn(lead_chat.id)
        await _settle(rooms, room.id)

        assert _agents_called(adapter) == ["main"]
        assert [c["kind"] for c in _cards(store, room.id)] == ["turn_skipped"]

    @pytest.mark.asyncio
    async def test_removing_a_running_participant_cancels_it_and_drops_its_reply(self, rooms, store, adapter, room):
        started = asyncio.Event()
        homelab = _session_of(rooms, room, "homelab")

        async def block():
            started.set()
            await asyncio.sleep(10)

        adapter.hooks[homelab] = block
        rooms.post_user_message(room.id, "@homelab hi")
        await asyncio.wait_for(started.wait(), 5)

        rooms.remove(room.id, "homelab")
        await _settle(rooms, room.id)

        assert ("homelab", "homelab replies") not in _messages(store, room.id)
        assert _agents_called(adapter) == ["homelab", "main"]


class TestRoundCapRules:
    def test_handoff_over_the_cap_is_rejected_with_the_rounds_left(self, rooms, store, room):
        store.rooms.update(room.id, max_rounds=2, rounds_used=1)
        handoffs = [{"agent": "homelab", "message": "a"}, {"agent": "research", "message": "b"}]

        with pytest.raises(ValueError, match="Only 1 round"):
            rooms.handoff(room.id, handoffs, "parallel")
        assert _messages(store, room.id) == []

    def test_raising_the_cap_posts_a_card_and_broadcasts_the_room(self, rooms, store, bus, room):
        rooms.set_max_rounds(room.id, 20)

        card = _cards(store, room.id)[-1]
        assert (card["kind"], card["max_rounds"]) == ("cap_raised", 20)
        assert bus.room_updates()[-1]["max_rounds"] == 20

    @pytest.mark.asyncio
    async def test_spent_rounds_broadcast_the_room(self, rooms, bus, room):
        rooms.post_user_message(room.id, "@homelab hi")
        await _settle(rooms, room.id)

        assert bus.room_updates()[-1]["rounds_used"] == 1


class TestQueuedTurns:
    @pytest.mark.asyncio
    async def test_queued_turns_are_broadcast_until_they_start(self, rooms, store, bus, lead_chat, room):
        store.begin_turn(lead_chat.id)
        rooms.handoff(
            room.id,
            [{"agent": "homelab", "message": "a"}, {"agent": "research", "message": "b"}],
            "sequential",
        )

        assert bus.room_updates()[-1]["queued"] == ["homelab", "research", "main"]
        store.end_turn(lead_chat.id)
        await _settle(rooms, room.id)
        assert [r["queued"] for r in bus.room_updates()][-3:] == [["research", "main"], ["main"], []]

    @pytest.mark.asyncio
    async def test_a_user_message_broadcasts_its_queued_turns(self, rooms, bus, room):
        rooms.post_user_message(room.id, "@homelab hi")

        assert bus.room_updates()[-1]["queued"] == ["homelab", "main"]
        await _settle(rooms, room.id)

    def test_stop_broadcasts_no_queued_turns(self, rooms, store, bus, lead_chat, room):
        store.begin_turn(lead_chat.id)
        rooms.handoff(room.id, [{"agent": "homelab", "message": "a"}], "sequential")

        rooms.stop(room.id)

        assert bus.room_updates()[-1]["queued"] == []


class TestCloseFromALeadTurn:
    @pytest.mark.asyncio
    async def test_room_close_keeps_the_callers_turn_and_posts_its_reply(
        self, rooms, runner, store, adapter, lead_chat, room, monkeypatch
    ):
        _bind_tools(monkeypatch, runner)
        seen = {}

        async def lead_closes():
            sessions_tools.room_close()
            await asyncio.sleep(0)
            seen["cancelled"] = is_cancelled()

        adapter.hooks[lead_chat.id] = lead_closes
        rooms.post_user_message(room.id, "wrap up")
        await _settle(rooms, room.id)

        assert seen == {"cancelled": False}
        assert _messages(store, room.id)[-1] == ("main", "main replies")
        assert rooms.detail(rooms.get(room.id))["status"] == SessionStatus.COMPLETED.value


class TestBootRecovery:
    def test_open_room_that_was_routing_gets_an_interrupted_card(self, store, adapter, room):
        store.rooms.update(room.id, routing=True)

        SessionRunner(store, adapter, event_bus=None).rooms.recover_interrupted()

        assert [c["kind"] for c in _cards(store, room.id)] == ["interrupted"]
        assert store.rooms.get(room.id).routing is False


class TestRoomToolGates:
    @pytest.fixture(autouse=True)
    def bound(self, runner, monkeypatch):
        _bind_tools(monkeypatch, runner)

    @staticmethod
    def _as(monkeypatch, session_id):
        monkeypatch.setattr(sessions_tools, "get_current_session_id", lambda: session_id)

    def test_non_lead_cannot_close(self, rooms, room, monkeypatch):
        self._as(monkeypatch, _session_of(rooms, room, "homelab"))

        with pytest.raises(ValueError, match="lead"):
            sessions_tools.room_close(room_id=room.id)

    def test_non_lead_cannot_hand_off(self, rooms, room, monkeypatch):
        self._as(monkeypatch, _session_of(rooms, room, "homelab"))

        with pytest.raises(ValueError, match="lead"):
            sessions_tools.room_handoff([{"agent": "research", "message": "x"}], room_id=room.id)

    def test_closed_room_rejects_invite_handoff_and_posts(self, rooms, lead_chat, room, monkeypatch):
        rooms.close(room.id)
        self._as(monkeypatch, lead_chat.id)

        with pytest.raises(ValueError, match="closed"):
            sessions_tools.room_invite("research", "x", room_id=room.id)
        with pytest.raises(ValueError, match="closed"):
            sessions_tools.room_handoff([{"agent": "homelab", "message": "x"}], room_id=room.id)
        with pytest.raises(ValueError, match="closed"):
            rooms.post_user_message(room.id, "late")


_CRITERIA = ["The NAS has 20% free space", "No data was deleted"]


@pytest.fixture
def orchestrator(tmp_path, runner, rooms, monkeypatch):
    orchestrator = JobsOrchestrator(JobStore(tmp_path / "jobs.json"), runner)
    monkeypatch.setattr(jobs_tools, "_jobs_orchestrator", orchestrator)
    return orchestrator


def _job(orchestrator, parent_session_id, **fields):
    job = orchestrator._jobs.add(
        Job(id="", parent_session_id=parent_session_id, prompt="free up disk", acceptance_criteria=_CRITERIA, **fields)
    )
    orchestrator.register_worker(job.id, "worker-1", timeout_minutes=30)
    orchestrator._jobs.update_state(job.id, "verifying")
    return orchestrator.get_job(job.id)


def _verdict(passed: bool) -> str:
    return json.dumps({"ac_results": [{"ac_text": c, "pass": passed, "reason": "checked"} for c in _CRITERIA]})


def _cards(store, room_id, kind=None):
    return [e for e in store.read_events(room_id) if e["type"] == "room_card" and (kind is None or e["kind"] == kind)]


class TestLinkJob:
    def test_unknown_job_is_rejected(self, rooms, store, lead_chat, orchestrator):
        with pytest.raises(ValueError, match="Unknown job: nope"):
            rooms.create("PROJ-1", user_id="u", lead_session_id=lead_chat.id, job_id="nope")
        with pytest.raises(ValueError, match="Unknown job: nope"):
            rooms.link_job(rooms.create("PROJ-1", user_id="u", lead_session_id=lead_chat.id).id, "nope")

        assert all(r.job_id is None for r in store.rooms.list())

    @pytest.mark.asyncio
    async def test_a_finished_job_cannot_be_linked(self, rooms, room, lead_chat, orchestrator):
        job = _job(orchestrator, lead_chat.id)
        orchestrator._finalize(job, JobState.CANCELLED)

        with pytest.raises(RoomConflictError, match="cancelled"):
            rooms.link_job(room.id, job.id)
        with pytest.raises(RoomConflictError):
            rooms.create("PROJ-2", user_id="u", lead_agent_id="research", job_id=job.id)

    @pytest.mark.asyncio
    async def test_room_open_tool_links_a_job(self, runner, store, lead_chat, orchestrator, monkeypatch):
        job = _job(orchestrator, lead_chat.id)
        monkeypatch.setattr(sessions_tools, "_session_runner", runner)
        monkeypatch.setattr(sessions_tools, "_call", lambda fn, *a, **kw: fn(*a, **kw))
        monkeypatch.setattr(sessions_tools, "get_current_session_id", lambda: lead_chat.id)

        result = sessions_tools.room_open("PROJ-1", lead_self=True, job_id=job.id)

        assert result["job"]["id"] == job.id
        assert store.rooms.get(result["id"]).job_id == job.id

    @pytest.mark.asyncio
    async def test_context_shows_the_job_criteria_and_state(self, rooms, room, lead_chat, orchestrator):
        job = _job(orchestrator, lead_chat.id)
        rooms.link_job(room.id, job.id)
        room = rooms.get(room.id)

        context, _cursor = rooms._render_context(room, room.participant("homelab"))

        assert f'<job id="{job.id}" state="verifying">' in context
        assert all(c in context for c in _CRITERIA)


class TestJobVerification:
    @pytest.mark.asyncio
    async def test_verified_job_posts_a_done_card_and_closes_the_room(
        self, rooms, store, room, lead_chat, orchestrator
    ):
        job = _job(orchestrator, lead_chat.id)
        rooms.link_job(room.id, job.id)

        await orchestrator._handle_verifier_complete(job, "verifier-1", _verdict(True))

        [card] = _cards(store, room.id, "job_done")
        assert card["job_id"] == job.id
        assert store.get_session(room.id).status == SessionStatus.COMPLETED.value

    @pytest.mark.asyncio
    async def test_failed_verification_posts_a_card_and_wakes_the_lead(
        self, rooms, store, adapter, room, lead_chat, orchestrator
    ):
        job = _job(orchestrator, lead_chat.id, max_attempts=1)
        rooms.link_job(room.id, job.id)

        await orchestrator._handle_verifier_complete(job, "verifier-1", _verdict(False))
        await _settle(rooms, room.id)

        [card] = _cards(store, room.id, "job_failed")
        assert (card["job_id"], card["state"]) == (job.id, "stuck")
        assert store.get_session(room.id).status != SessionStatus.COMPLETED.value
        [(agent, prompt)] = adapter.calls
        assert agent == "main"
        assert job.id in prompt and 'state="stuck"' in _context(prompt)
        assert "Verifier failed after max attempts" in prompt

    @pytest.mark.asyncio
    async def test_a_re_emitted_job_state_posts_nothing_new(self, rooms, store, adapter, room, lead_chat, orchestrator):
        job = _job(orchestrator, lead_chat.id, max_attempts=1)
        rooms.link_job(room.id, job.id)
        await orchestrator._handle_verifier_complete(job, "verifier-1", _verdict(False))
        await _settle(rooms, room.id)

        orchestrator._emit_job_event(orchestrator.get_job(job.id))
        await _settle(rooms, room.id)

        assert len(_cards(store, room.id, "job_failed")) == 1
        assert len(adapter.calls) == 1

    @pytest.mark.asyncio
    async def test_participant_jobs_are_mirrored_as_status_cards(self, rooms, store, room, orchestrator):
        homelab = rooms.get(room.id).participant("homelab")

        job = _job(orchestrator, homelab.session_id)

        assert [(c["kind"], c["job_id"], c["state"], c["agent_id"]) for c in _cards(store, room.id)[-1:]] == [
            ("job_status", job.id, "running", "homelab")
        ]
        assert "prompt" not in _cards(store, room.id)[-1]

        orchestrator._emit_job_event(orchestrator.get_job(job.id))
        orchestrator._emit_job_event(orchestrator.get_job(job.id))

        assert [c["state"] for c in _cards(store, room.id, "job_status")] == ["running", "verifying"]

    @pytest.mark.asyncio
    async def test_a_participant_job_finishing_leaves_an_unlinked_room_open(self, rooms, store, room, orchestrator):
        homelab = rooms.get(room.id).participant("homelab")
        job = _job(orchestrator, homelab.session_id)

        await orchestrator._handle_verifier_complete(job, "verifier-1", _verdict(True))

        assert _cards(store, room.id)[-1]["kind"] == "job_status"
        assert store.get_session(room.id).status != SessionStatus.COMPLETED.value

    @pytest.mark.asyncio
    @pytest.mark.parametrize("state", ["errored", "cancelled"])
    async def test_errored_and_cancelled_jobs_count_as_failures(
        self, rooms, store, adapter, room, lead_chat, orchestrator, state
    ):
        job = _job(orchestrator, lead_chat.id)
        rooms.link_job(room.id, job.id)

        orchestrator._finalize(job, JobState(state))
        await _settle(rooms, room.id)

        assert [c["state"] for c in _cards(store, room.id, "job_failed")] == [state]
        assert [agent for agent, _ in adapter.calls] == ["main"]

    @pytest.mark.asyncio
    async def test_an_unlinked_job_no_longer_closes_the_room(self, rooms, store, room, lead_chat, orchestrator):
        job = _job(orchestrator, lead_chat.id)
        rooms.link_job(room.id, job.id)
        rooms.link_job(room.id, None)

        await orchestrator._handle_verifier_complete(job, "verifier-1", _verdict(True))

        assert _cards(store, room.id, "job_done") == []
        assert store.get_session(room.id).status != SessionStatus.COMPLETED.value

    @pytest.mark.asyncio
    async def test_a_job_linked_to_two_rooms_closes_both(self, rooms, store, room, lead_chat, orchestrator):
        job = _job(orchestrator, lead_chat.id)
        other = rooms.create("PROJ-2", user_id="u", lead_agent_id="research", job_id=job.id)
        rooms.link_job(room.id, job.id)

        await orchestrator._handle_verifier_complete(job, "verifier-1", _verdict(True))

        assert {store.get_session(r).status for r in (room.id, other.id)} == {SessionStatus.COMPLETED.value}

    @pytest.mark.asyncio
    async def test_a_cancelled_job_wakes_the_lead_without_blaming_the_criteria(
        self, rooms, store, adapter, room, lead_chat, orchestrator
    ):
        job = _job(orchestrator, lead_chat.id)
        rooms.link_job(room.id, job.id)

        await orchestrator.cancel_job(job.id, reason="no longer needed")
        await _settle(rooms, room.id)

        [(_agent, prompt)] = adapter.calls
        assert "is cancelled" in prompt and "no longer needed" in prompt
        assert "did not verify" not in prompt

    @pytest.mark.asyncio
    async def test_boot_recovery_posts_a_restart_card_and_leaves_the_lead_asleep(
        self, rooms, store, adapter, room, lead_chat, orchestrator
    ):
        job = _job(orchestrator, lead_chat.id)
        rooms.link_job(room.id, job.id)

        orchestrator.recover_orphaned_jobs()
        await _settle(rooms, room.id)

        assert _cards(store, room.id, "job_failed") == []
        [card] = _cards(store, room.id, "interrupted")
        assert card["job_id"] == job.id and "restarted" in card["text"]
        assert adapter.calls == []

    @pytest.mark.asyncio
    async def test_job_done_lets_the_running_turn_post_before_closing(
        self, rooms, store, adapter, room, lead_chat, orchestrator
    ):
        job = _job(orchestrator, lead_chat.id)
        rooms.link_job(room.id, job.id)
        started, release = asyncio.Event(), asyncio.Event()

        async def hold():
            started.set()
            await release.wait()

        adapter.hooks[lead_chat.id] = hold
        rooms.post_user_message(room.id, "status?")
        await started.wait()

        await orchestrator._handle_verifier_complete(job, "verifier-1", _verdict(True))
        assert store.get_session(room.id).status != SessionStatus.COMPLETED.value
        with pytest.raises(RoomConflictError):
            rooms.post_user_message(room.id, "more?")

        release.set()
        await _settle(rooms, room.id)

        assert _messages(store, room.id)[-1] == ("main", "main replies")
        assert store.get_session(room.id).status == SessionStatus.COMPLETED.value

    @pytest.mark.asyncio
    async def test_a_failing_rooms_hook_leaves_the_job_transition_intact(
        self, rooms, store, room, lead_chat, orchestrator, monkeypatch
    ):
        job = _job(orchestrator, lead_chat.id)
        frames = []
        orchestrator._event_bus = type("Bus", (), {"emit": lambda _self, kind, data: frames.append(kind)})()
        monkeypatch.setattr(rooms, "on_job_event", lambda *a, **kw: 1 / 0)

        await orchestrator._handle_verifier_complete(job, "verifier-1", _verdict(True))

        assert orchestrator.get_job(job.id).state == "done"
        assert "job_update" in frames


class TestAttention:
    def test_participant_attention_is_mirrored_and_counted(self, rooms, runner, store, bus, room):
        homelab = rooms.get(room.id).participant("homelab")

        runner.open_attention(homelab.session_id, source=SOURCE_ASK, ref_id="ask-1", kind="ask")

        [card] = _cards(store, room.id, "attention")
        [record] = store.attention.open_records(homelab.session_id)
        assert (card["attention_id"], card["session_id"], card["agent_id"]) == (
            record.id,
            homelab.session_id,
            "homelab",
        )
        detail = rooms.detail(rooms.get(room.id))
        assert detail["needs_attention"] is True
        assert [r["id"] for r in detail["attention"]] == [record.id]
        assert bus.room_updates()[-1]["needs_attention"] is True

        runner.clear_attention_ref(SOURCE_ASK, "ask-1")

        assert rooms.detail(rooms.get(room.id))["attention"] == []
        assert bus.room_updates()[-1]["needs_attention"] is False

    def test_send_failures_and_needs_ack_deliveries_are_mirrored(self, rooms, store, bus, room):
        homelab = rooms.get(room.id).participant("homelab")

        report_send_failure(store, bus, homelab.session_id, ref_id="send-1", error="boom")
        store.record_delivery(
            homelab.session_id,
            {"type": "delivery", "delivery_id": "d-1", "kind": "needs_ack", "message": "approve?"},
            needs_ack=True,
        )

        cards = _cards(store, room.id, "attention")
        assert [c["attention_kind"] for c in cards] == ["send_failed", "needs_ack"]
        assert "approve?" not in str(cards)
        assert len(bus.room_updates()[-1]["attention"]) == 2

    def test_a_failing_rooms_hook_leaves_attention_intact(self, rooms, runner, store, room, monkeypatch):
        homelab = rooms.get(room.id).participant("homelab")
        monkeypatch.setattr(rooms, "on_attention_opened", lambda record: 1 / 0)
        monkeypatch.setattr(rooms, "on_attention_changed", lambda session_id: 1 / 0)
        store.attention.on_open = rooms.on_attention_opened

        runner.open_attention(homelab.session_id, source=SOURCE_ASK, ref_id="ask-9", kind="ask")
        assert len(store.attention.open_records(homelab.session_id)) == 1

        runner.clear_attention_ref(SOURCE_ASK, "ask-9")
        assert store.attention.open_records(homelab.session_id) == []

    def test_attention_outside_rooms_posts_nothing(self, rooms, runner, store, room, lead_chat):
        other = store.create_session(type(lead_chat)(id="", user_id="u"))

        runner.open_attention(other.id, source=SOURCE_ASK, ref_id="ask-2", kind="ask")

        assert _cards(store, room.id, "attention") == []


class TestTerminals:
    def test_participant_terminal_state_is_mirrored(self, rooms, store, bus, room, tmp_path):
        homelab = rooms.get(room.id).participant("homelab")

        terminals = TerminalSessionStore(tmp_path / "terminals.json")
        terminals.add(TerminalSession(id="term-1", cmd="make", cwd=str(tmp_path), parent_session_id=homelab.session_id))

        terminal_state_callback(bus, terminals, rooms)("term-1", "failed")

        assert ("terminal_state", {"terminal_id": "term-1", "state": "failed"}) in bus.frames

        [card] = _cards(store, room.id, "terminal")
        assert (card["terminal_id"], card["state"], card["agent_id"]) == ("term-1", "failed", "homelab")


class TestAttentionBroadcasts:
    def test_runner_open_broadcasts_the_room_once(self, rooms, runner, bus, room):
        homelab = rooms.get(room.id).participant("homelab")
        before = len(bus.room_updates())

        runner.open_attention(homelab.session_id, source=SOURCE_ASK, ref_id="ask-1", kind="ask")

        assert len(bus.room_updates()) == before + 1

    def test_needs_ack_delivery_broadcasts_the_room_once(self, rooms, runner, bus, room):
        homelab = rooms.get(room.id).participant("homelab")
        before = len(bus.room_updates())

        runner.deliver_to_session(homelab.session_id, "approve?", source="test", kind="needs_ack")

        assert len(bus.room_updates()) == before + 1


class TestBootRecoveryWithJobs:
    @pytest.mark.asyncio
    async def test_open_room_whose_job_is_done_closes_at_boot(self, rooms, store, room, lead_chat, orchestrator):
        job = _job(orchestrator, lead_chat.id)
        rooms.link_job(room.id, job.id)
        orchestrator._jobs.update_state(job.id, "done")

        rooms.recover_interrupted()

        assert store.get_session(room.id).status == SessionStatus.COMPLETED.value


@pytest.fixture
def opener(store):
    return store.create_session(Session(id="", user_id="web-anonymous"))


@pytest.fixture
def opened(rooms, opener):
    room = rooms.create("PROJ-7", user_id="web-anonymous", lead_agent_id="main", opener_session_id=opener.id)
    rooms.invite(room.id, "homelab")
    rooms.invite(room.id, "research")
    return rooms.get(room.id)


async def _reports(rooms, store, session_id):
    """The report turns that woke `session_id`, once every pending wake has run."""
    while rooms._wakes:
        await asyncio.wait(list(rooms._wakes.values()))
        await asyncio.sleep(0)
    return [e["text"] for e in store.read_events(session_id) if e["type"] == "user_input"]


def _as_caller(monkeypatch, runner, session_id):
    _bind_tools(monkeypatch, runner)
    monkeypatch.setattr(sessions_tools, "get_current_session_id", lambda: session_id)


def _lead_session(room):
    return room.participant("main").session_id


class TestRoomOpenFreshLead:
    @pytest.fixture(autouse=True)
    def bound(self, runner, monkeypatch, opener):
        _as_caller(monkeypatch, runner, opener.id)

    @pytest.mark.asyncio
    async def test_a_fresh_lead_of_the_callers_agent_is_briefed_and_started(self, rooms, store, adapter, opener):
        result = sessions_tools.room_open("PROJ-7", brief="Free up space on the NAS.")

        room = store.rooms.get(result["id"])
        [lead] = room.participants
        assert lead.lead and lead.agent_id == "main" and lead.session_id != opener.id
        assert room.opener_session_id == opener.id
        assert store.get_session(lead.session_id).metadata["room_id"] == room.id
        assert result["participants"][0]["working"] is True
        await _settle(rooms, room.id)
        [(agent, prompt)] = adapter.calls
        assert agent == "main"
        assert "Free up space on the NAS." in _context(prompt)
        assert store.read_events(opener.id) == []

    @pytest.mark.asyncio
    async def test_lead_agent_picks_the_leads_agent(self, rooms, store):
        result = sessions_tools.room_open("PROJ-7", brief="Read up.", lead_agent="research")

        await _settle(rooms, result["id"])
        assert [p.agent_id for p in store.rooms.get(result["id"]).participants] == ["research"]

    def test_a_fresh_lead_needs_a_brief(self):
        with pytest.raises(ValueError, match="brief"):
            sessions_tools.room_open("PROJ-7")

    def test_lead_self_keeps_the_caller_as_lead(self, store, adapter, opener):
        result = sessions_tools.room_open("PROJ-7", lead_self=True)

        room = store.rooms.get(result["id"])
        [lead] = room.participants
        assert (lead.session_id, room.opener_session_id) == (opener.id, opener.id)
        assert adapter.calls == []

    def test_notify_when_takes_a_bare_event_name(self, store):
        result = sessions_tools.room_open("PROJ-7", lead_self=True, notify_when="closed")

        assert store.rooms.get(result["id"]).notify_when == ["closed"]

    @pytest.mark.parametrize("notify_when,error", [(["x"], "notify_when takes"), (5, "notify_when must be a list")])
    def test_a_bad_notify_when_is_rejected(self, notify_when, error):
        with pytest.raises(ValueError, match=error):
            sessions_tools.room_open("PROJ-7", lead_self=True, notify_when=notify_when)


class TestOpenerReports:
    @pytest.mark.asyncio
    async def test_a_user_close_wakes_the_opener_with_the_leads_final_message(self, rooms, store, opener, opened):
        rooms.post_user_message(opened.id, "wrap up")
        await _settle(rooms, opened.id)

        rooms.close(opened.id)

        [report] = await _reports(rooms, store, opener.id)
        assert f"Room 'PROJ-7' ({opened.id}): Closed by the user." in report
        assert f"(#chats?sessionId={opened.id})" in report
        assert "@main's final message:\nmain replies" in report

    @pytest.mark.asyncio
    async def test_a_close_from_the_leads_turn_reports_the_reply_it_posts(
        self, rooms, runner, store, adapter, opener, opened, monkeypatch
    ):
        _as_caller(monkeypatch, runner, _lead_session(opened))

        async def lead_closes():
            sessions_tools.room_close()

        adapter.hooks[_lead_session(opened)] = lead_closes
        rooms.post_user_message(opened.id, "wrap up")
        await _settle(rooms, opened.id)

        [report] = await _reports(rooms, store, opener.id)
        assert "Closed by the lead." in report and "main replies" in report

    @pytest.mark.asyncio
    async def test_the_opener_closes_its_room(self, rooms, runner, store, opener, opened, monkeypatch):
        _as_caller(monkeypatch, runner, opener.id)

        result = sessions_tools.room_close(room_id=opened.id)

        assert result["status"] == SessionStatus.COMPLETED.value
        [report] = await _reports(rooms, store, opener.id)
        assert "Closed by its opener." in report

    @pytest.mark.asyncio
    async def test_the_final_message_skips_the_openers_later_post(self, rooms, store, opener, opened):
        rooms.post_user_message(opened.id, "wrap up")
        await _settle(rooms, opened.id)
        rooms._post(opened.id, "thanks all", "main", opener.id, None, role="opener")

        rooms.close(opened.id)

        [report] = await _reports(rooms, store, opener.id)
        assert "main replies" in report and "thanks all" not in report

    @pytest.mark.asyncio
    async def test_a_verified_job_closes_the_room_once_after_the_running_turn(
        self, rooms, store, adapter, opener, opened, orchestrator
    ):
        job = _job(orchestrator, opener.id)
        rooms.link_job(opened.id, job.id)
        started, release = asyncio.Event(), asyncio.Event()

        async def hold():
            started.set()
            await release.wait()

        adapter.hooks[_lead_session(opened)] = hold
        rooms.post_user_message(opened.id, "status?")
        await started.wait()
        await orchestrator._handle_verifier_complete(job, "verifier-1", _verdict(True))
        release.set()
        await _settle(rooms, opened.id)

        [report] = await _reports(rooms, store, opener.id)
        assert f"Closed after job {job.id} verified (done)." in report and "main replies" in report

    @pytest.mark.asyncio
    async def test_boot_recovery_says_why_it_closed_the_room(self, rooms, store, opener, opened, orchestrator):
        job = _job(orchestrator, opener.id)
        rooms.link_job(opened.id, job.id)
        orchestrator._jobs.update_state(job.id, "done")

        rooms.recover_interrupted()

        [report] = await _reports(rooms, store, opener.id)
        assert f"Closed at boot recovery, after job {job.id} verified (done)." in report

    @pytest.mark.asyncio
    async def test_round_cap_is_reported_once_while_its_attention_is_open(self, rooms, store, opener, opened):
        store.rooms.update(opened.id, max_rounds=1)
        rooms.post_user_message(opened.id, "@homelab @research hi")
        await _settle(rooms, opened.id)

        rooms._cap(rooms.get(opened.id))

        [report] = await _reports(rooms, store, opener.id)
        assert "Round cap reached (1/1)." in report

    @pytest.mark.asyncio
    async def test_attention_is_reported(self, rooms, runner, store, opener, opened):
        runner.open_attention(opened.participant("homelab").session_id, source=SOURCE_ASK, ref_id="a", kind="ask")

        [report] = await _reports(rooms, store, opener.id)
        assert "@homelab needs you (ask)." in report

    @pytest.mark.asyncio
    async def test_failed_turns_in_one_step_are_reported_once(self, rooms, store, adapter, opener, opened):
        async def boom():
            raise RuntimeError("provider down")

        for agent_id in ("homelab", "research"):
            adapter.hooks[opened.participant(agent_id).session_id] = boom
        store.begin_turn(_lead_session(opened))
        rooms.handoff(
            opened.id, [{"agent": "homelab", "message": "a"}, {"agent": "research", "message": "b"}], "parallel"
        )
        store.end_turn(_lead_session(opened))
        await _settle(rooms, opened.id)

        [report] = await _reports(rooms, store, opener.id)
        assert "@homelab's turn failed: provider down. @research's turn failed: provider down." in report

    @pytest.mark.asyncio
    async def test_room_turns_do_not_reach_the_opener(self, rooms, store, adapter, opener, opened):
        rooms.post_user_message(opened.id, "@homelab hi")
        await _settle(rooms, opened.id)

        assert _agents_called(adapter) == ["homelab", "main"]
        assert await _reports(rooms, store, opener.id) == []

    @pytest.mark.asyncio
    async def test_reports_during_the_openers_turn_wake_it_once_after(self, rooms, runner, store, opener, opened):
        store.begin_turn(opener.id)
        for agent_id in ("homelab", "research"):
            runner.open_attention(
                opened.participant(agent_id).session_id, source=SOURCE_ASK, ref_id=agent_id, kind="ask"
            )
        await asyncio.sleep(0)
        assert not [e for e in store.read_events(opener.id) if e["type"] == "user_input"]
        assert opener.id in rooms._wakes

        store.end_turn(opener.id)

        [report] = await _reports(rooms, store, opener.id)
        assert "@homelab needs you" in report and "@research needs you" in report

    @pytest.mark.asyncio
    async def test_reports_wait_for_the_openers_queued_messages(self, rooms, runner, store, opener, opened):
        store.queue_message(opener.id, {"id": "q-1", "text": "hi"})
        runner.open_attention(opened.participant("homelab").session_id, source=SOURCE_ASK, ref_id="a", kind="ask")
        await asyncio.sleep(0)
        assert not [e for e in store.read_events(opener.id) if e["type"] == "user_input"]

        store.drop_queued_message(opener.id, "q-1")

        assert len(await _reports(rooms, store, opener.id)) == 1

    @pytest.mark.asyncio
    async def test_a_compacted_opener_gets_the_reports(self, rooms, runner, store, opener, opened):
        successor = store.compact_session(opener.id)

        runner.open_attention(opened.participant("homelab").session_id, source=SOURCE_ASK, ref_id="a", kind="ask")

        assert len(await _reports(rooms, store, successor.id)) == 1

    @pytest.mark.asyncio
    async def test_notify_when_narrows_the_reports(self, rooms, runner, store, opener):
        room = rooms.create(
            "PROJ-8", user_id="u", lead_agent_id="main", opener_session_id=opener.id, notify_when=["closed"]
        )
        room = rooms.invite(room.id, "homelab")

        runner.open_attention(room.participant("homelab").session_id, source=SOURCE_ASK, ref_id="a", kind="ask")
        assert await _reports(rooms, store, opener.id) == []
        rooms.close(room.id)

        [report] = await _reports(rooms, store, opener.id)
        assert "Closed by the user." in report

    @pytest.mark.asyncio
    async def test_no_reports_when_the_opener_leads_even_after_compaction(self, rooms, runner, store, opener):
        room = rooms.create("PROJ-8", user_id="u", lead_session_id=opener.id, opener_session_id=opener.id)
        room = rooms.invite(room.id, "homelab")
        successor = store.compact_session(opener.id)

        runner.open_attention(room.participant("homelab").session_id, source=SOURCE_ASK, ref_id="a", kind="ask")
        rooms.close(room.id)

        assert await _reports(rooms, store, successor.id) == []


class TestRoomPost:
    @pytest.fixture(autouse=True)
    def bound(self, runner, monkeypatch, opener):
        _as_caller(monkeypatch, runner, opener.id)

    @pytest.mark.asyncio
    async def test_unaddressed_post_goes_to_the_lead_and_keeps_the_rounds(self, rooms, store, adapter, opened):
        store.rooms.update(opened.id, rounds_used=3)

        sessions_tools.room_post(opened.id, "how is it going?")
        await _settle(rooms, opened.id)

        assert _agents_called(adapter) == ["main"]
        assert 'author="main" role="opener" untrusted="true">how is it going?' in _context(adapter.calls[0][1])
        assert store.rooms.get(opened.id).rounds_used == 3

    @pytest.mark.asyncio
    async def test_mention_runs_that_participant_then_the_lead_and_spends_a_round(self, rooms, store, adapter, opened):
        store.rooms.update(opened.id, rounds_used=3)

        sessions_tools.room_post(opened.id, "@homelab check sda")
        await _settle(rooms, opened.id)

        assert _agents_called(adapter) == ["homelab", "main"]
        assert 'author="main" role="opener" untrusted="true">@homelab check sda' in _context(adapter.calls[0][1])
        assert store.rooms.get(opened.id).rounds_used == 4

    @pytest.mark.asyncio
    async def test_a_compacted_opener_still_posts(self, rooms, store, adapter, opener, opened, runner, monkeypatch):
        _as_caller(monkeypatch, runner, store.compact_session(opener.id).id)

        sessions_tools.room_post(opened.id, "still there?")
        await _settle(rooms, opened.id)

        assert _agents_called(adapter) == ["main"]

    def test_the_lead_is_pointed_to_room_handoff(self, store, opened, runner, monkeypatch):
        _as_caller(monkeypatch, runner, store.compact_session(_lead_session(opened)).id)

        with pytest.raises(ValueError, match="room_handoff"):
            sessions_tools.room_post(opened.id, "@homelab go")

    def test_other_sessions_cannot_post(self, opened, runner, monkeypatch):
        _as_caller(monkeypatch, runner, opened.participant("homelab").session_id)

        with pytest.raises(ValueError, match="opened"):
            sessions_tools.room_post(opened.id, "hi")

    def test_a_muted_participant_is_refused(self, rooms, opened):
        rooms.set_muted(opened.id, "homelab", True)

        with pytest.raises(ValueError, match="muted"):
            sessions_tools.room_post(opened.id, "@homelab go")
        assert _messages(rooms._store, opened.id) == []

    @pytest.mark.asyncio
    async def test_a_mention_at_the_cap_is_refused_without_a_report(self, rooms, store, opener, opened):
        store.rooms.update(opened.id, max_rounds=1, rounds_used=1)

        with pytest.raises(ValueError, match="Only the user can raise the cap"):
            sessions_tools.room_post(opened.id, "@homelab status?")
        assert await _reports(rooms, store, opener.id) == []


class TestRoomListAndStatus:
    @pytest.fixture(autouse=True)
    def bound(self, runner, monkeypatch, opener):
        _as_caller(monkeypatch, runner, opener.id)

    def test_list_defaults_to_open_rooms_this_session_opened(self, rooms, store, opener, opened, room, monkeypatch):
        closed = rooms.create("PROJ-old", user_id="u", lead_agent_id="main", opener_session_id=opener.id)
        rooms.close(closed.id)

        assert [r["id"] for r in sessions_tools.room_list()] == [opened.id]
        assert {r["id"] for r in sessions_tools.room_list(all=True)} == {opened.id, room.id}
        monkeypatch.setattr(sessions_tools, "get_current_session_id", lambda: store.compact_session(opener.id).id)
        assert [r["id"] for r in sessions_tools.room_list()] == [opened.id]

    @pytest.mark.asyncio
    async def test_status_reports_the_room_without_its_transcript(
        self, rooms, runner, store, opened, orchestrator, opener
    ):
        job = _job(orchestrator, opener.id)
        rooms.link_job(opened.id, job.id)
        rooms.set_muted(opened.id, "research", True)
        rooms.post_message(opened.id, "x" * 2000)
        store.rooms.update(opened.id, rounds_used=2)
        homelab = opened.participant("homelab")
        runner.open_attention(homelab.session_id, source=SOURCE_ASK, ref_id="a", kind="ask")
        store.begin_turn(homelab.session_id)

        status = sessions_tools.room_status(opened.id)

        assert (status["id"], status["title"], status["status"]) == (opened.id, "PROJ-7", "active")
        assert (status["rounds_used"], status["max_rounds"]) == (2, DEFAULT_MAX_ROUNDS)
        assert status["participants"] == [
            {"agent_id": "main", "name": "main", "lead": True, "muted": False, "working": False},
            {"agent_id": "homelab", "name": "Homelab", "lead": False, "muted": False, "working": True},
            {"agent_id": "research", "name": "Research", "lead": False, "muted": True, "working": False},
        ]
        assert status["last_message"] == {"author": "user", "content": "x" * 500 + "…"}
        assert status["attention"] == [{"kind": "ask", "agent_id": "homelab"}]
        assert status["job"] == {"id": job.id, "state": "verifying"}

    def test_the_last_message_is_found_beyond_one_page(self, rooms, opened, monkeypatch):
        monkeypatch.setattr("tsugite_daemon.rooms._PAGE_EVENTS", 2)
        rooms.post_message(opened.id, "the last word")
        for _ in range(3):
            rooms.set_muted(opened.id, "homelab", True)
            rooms.set_muted(opened.id, "homelab", False)

        assert sessions_tools.room_status(opened.id)["last_message"]["content"] == "the last word"
