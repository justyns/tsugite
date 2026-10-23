import asyncio
import contextvars

import pytest
from tsugite_daemon.agent_store import Agent
from tsugite_daemon.attention_store import SOURCE_ROOM
from tsugite_daemon.session_runner import SessionRunner, get_current_session_id
from tsugite_daemon.session_store import Session, SessionStatus, SessionStore

from tsugite.cancellation import is_cancelled
from tsugite.history.ui_events import split_injected_context
from tsugite.interaction import NonInteractiveBackend, get_interaction_backend, set_interaction_backend
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
        store.rooms.update(room.id, max_rounds=1, rounds_used=1)

        rooms.post_user_message(room.id, "@homelab hi")
        await _settle(rooms, room.id)

        assert [agent for agent, _ in adapter.calls] == ["main"]
        [card] = [e for e in store.read_events(room.id) if e["type"] == "room_card"]
        assert card["kind"] == "round_cap"
        [record] = store.attention.open_records(room.id)
        assert (record.source, record.kind) == (SOURCE_ROOM, "round_cap")

    @pytest.mark.asyncio
    async def test_raising_the_cap_clears_the_attention(self, rooms, store, room):
        store.rooms.update(room.id, max_rounds=1, rounds_used=1)
        rooms.post_user_message(room.id, "@homelab hi")
        await _settle(rooms, room.id)

        rooms.set_max_rounds(room.id, 5)

        assert store.rooms.get(room.id).max_rounds == 5
        assert store.attention.open_records(room.id) == []

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
        result = sessions_tools.room_open("PROJ-9")

        room = store.rooms.get(result["id"])
        [lead] = room.participants
        assert (lead.agent_id, lead.session_id, lead.lead) == ("main", lead_chat.id, True)

    def test_invite_handoff_and_close_resolve_the_room_from_the_session(self, rooms, store, lead_chat):
        room_id = sessions_tools.room_open("PROJ-9")["id"]

        sessions_tools.room_invite("homelab", "Check the NAS.")
        store.begin_turn(lead_chat.id)
        queued = sessions_tools.room_handoff([{"agent": "homelab", "message": "go"}])
        closed = sessions_tools.room_close()

        assert queued["queued"] == 1
        assert rooms.get(room_id).participant("homelab").brief == "Check the NAS."
        assert closed["status"] == SessionStatus.COMPLETED.value
        assert store.get_session(lead_chat.id).status == SessionStatus.ACTIVE.value
        assert not rooms._steps.get(room_id)

    def test_tools_need_a_room(self):
        with pytest.raises(ValueError, match="room"):
            sessions_tools.room_handoff([{"agent": "homelab", "message": "go"}])

    def test_only_the_lead_invites(self, rooms, monkeypatch):
        room_id = sessions_tools.room_open("PROJ-9")["id"]
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
