"""Rooms: a shared transcript plus one session per participating agent."""

import re
from unittest.mock import patch

import pytest
from starlette.testclient import TestClient
from tsugite_daemon.adapters.base import BaseAdapter, ChannelContext
from tsugite_daemon.adapters.http import HTTPAgentAdapter, HTTPServer
from tsugite_daemon.agent_store import Agent
from tsugite_daemon.auth import TokenStore
from tsugite_daemon.config import HTTPConfig, RuntimeDefaults
from tsugite_daemon.room_store import DEFAULT_MAX_ROUNDS, Participant, Room, RoomStore
from tsugite_daemon.session_runner import SessionRunner
from tsugite_daemon.session_store import SessionSource, SessionStatus, SessionStore, room_transcript_error
from tsugite_daemon.webhook_store import WebhookStore


def _workspace(path, *agent_names):
    path.mkdir()
    for name in agent_names:
        (path / f"{name}.md").write_text(f"---\nname: {name}\n---\n\nHi.\n")
    return path


@pytest.fixture
def default_ws(tmp_path):
    return _workspace(tmp_path / "main", "default")


@pytest.fixture
def store(tmp_path, default_ws):
    store = SessionStore(tmp_path / "state" / "store.json")
    store.agents.ensure_default(default_ws, "default")
    for agent_id in ("homelab", "research"):
        ws = _workspace(tmp_path / agent_id, "default")
        store.agents.create(Agent(id=agent_id, name=agent_id.title(), workspace=str(ws), template="default"))
    return store


@pytest.fixture
def server(store, default_ws, tmp_path):
    with patch("tsugite.workspace.Workspace") as mock_ws_cls:
        from tsugite.workspace import WorkspaceNotFoundError

        mock_ws_cls.load.side_effect = WorkspaceNotFoundError("not found")
        mock_ws_cls.try_load.return_value = None
        adapter = HTTPAgentAdapter(
            runtime=RuntimeDefaults(workspace_dir=default_ws, agent_file="default"), session_store=store
        )
    srv = HTTPServer(
        config=HTTPConfig(enabled=True, host="127.0.0.1", port=8378),
        adapter=adapter,
        webhook_store=WebhookStore(tmp_path / "webhooks.json"),
        token_store=TokenStore(tmp_path / "tokens.json"),
    )
    srv.session_runner = SessionRunner(store=store, adapter=None, event_bus=srv.event_bus)
    return srv


@pytest.fixture
def frames(server, monkeypatch):
    sent = []
    monkeypatch.setattr(server.event_bus, "emit", lambda kind, payload: sent.append((kind, payload)))
    return sent


@pytest.fixture
def client(server):
    return TestClient(server.app)


@pytest.fixture
def auth(server):
    _st, raw = server._token_store.create_admin_token(name="test-token")
    return {"Authorization": f"Bearer {raw}"}


@pytest.fixture
def room(client, auth):
    resp = client.post("/api/rooms", headers=auth, json={"title": "PROJ-123", "lead_agent_id": "homelab"})
    assert resp.status_code == 201
    return resp.json()


def _events(store, room_id, event_type):
    return [e for e in store.read_events(room_id) if e["type"] == event_type]


class TestRoomStore:
    def test_rooms_persist_across_reload(self, tmp_path):
        store = RoomStore(tmp_path / "store.json")
        store.create(Room(id="r1", participants=[Participant(agent_id="a", session_id="s1", lead=True)]))
        store.update("r1", rounds_used=3)

        reloaded = RoomStore(tmp_path / "store.json").get("r1")

        assert reloaded.participants == [Participant(agent_id="a", session_id="s1", lead=True)]
        assert (reloaded.max_rounds, reloaded.rounds_used) == (DEFAULT_MAX_ROUNDS, 3)

    @pytest.mark.parametrize("bad", [{"stray": 1}, {"participants": [{"agent_id": "a", "session_id": "s", "x": 1}]}])
    def test_load_rejects_unknown_room_and_participant_keys(self, tmp_path, bad):
        store = RoomStore(tmp_path / "store.json")
        store.create(Room(id="r1"))
        store._storage.upsert("r1", {"id": "r1", "participants": [], **bad})

        with pytest.raises(TypeError):
            RoomStore(tmp_path / "store.json")


class TestRoomsApi:
    @pytest.mark.parametrize(
        "method,path",
        [
            ("GET", "/api/rooms"),
            ("POST", "/api/rooms"),
            ("GET", "/api/rooms/r1"),
            ("POST", "/api/rooms/r1/close"),
            ("POST", "/api/rooms/r1/participants"),
            ("PATCH", "/api/rooms/r1/participants/homelab"),
            ("DELETE", "/api/rooms/r1/participants/homelab"),
            ("POST", "/api/rooms/r1/messages"),
        ],
    )
    def test_requires_auth(self, client, method, path):
        assert client.request(method, path, json={}).status_code == 401

    def test_create_makes_a_room_transcript_and_a_lead_session(self, room, store):
        transcript = store.get_session(room["id"])
        assert (transcript.source, transcript.title, transcript.status) == ("room", "PROJ-123", "active")
        assert room["status"] == "active"
        assert room["max_rounds"] == DEFAULT_MAX_ROUNDS
        [lead] = room["participants"]
        assert (lead["agent_id"], lead["lead"], lead["muted"]) == ("homelab", True, False)
        session = store.get_session(lead["session_id"])
        assert (session.agent_id, session.metadata["room_id"]) == ("homelab", room["id"])
        assert [e["agent_id"] for e in _events(store, room["id"], "room_member")] == ["homelab"]

    def test_create_defaults_the_lead_to_the_default_agent(self, client, auth):
        resp = client.post("/api/rooms", headers=auth, json={"title": "T"})

        assert [p["agent_id"] for p in resp.json()["participants"]] == ["main"]

    @pytest.mark.parametrize(
        "body,error",
        [
            ({}, "title"),
            ({"title": "T", "lead_agent_id": "ghost"}, "unknown agent"),
            ({"title": "T", "max_rounds": 0}, "max_rounds"),
        ],
    )
    def test_create_rejects_a_bad_body(self, client, auth, body, error):
        resp = client.post("/api/rooms", headers=auth, json=body)

        assert resp.status_code == 400
        assert error in resp.json()["error"]

    def test_participant_session_keeps_its_room_through_compaction(self, room, store):
        successor = store.compact_session(room["participants"][0]["session_id"])

        assert successor.metadata["room_id"] == room["id"]
        with pytest.raises(ValueError, match="read-only"):
            store.set_metadata_bulk(successor.id, {"room_id": "elsewhere"})

    def test_list_details_matches_the_list_route(self, client, auth, room, server):
        assert server.session_runner.rooms.list_details() == client.get("/api/rooms", headers=auth).json()["rooms"]

    def test_list_and_get(self, client, auth, room):
        assert [r["id"] for r in client.get("/api/rooms", headers=auth).json()["rooms"]] == [room["id"]]
        assert client.get(f"/api/rooms/{room['id']}", headers=auth).json()["title"] == "PROJ-123"
        assert client.get("/api/rooms/nope", headers=auth).status_code == 404

    def test_invite_creates_the_agent_room_session(self, client, auth, room, store):
        resp = client.post(f"/api/rooms/{room['id']}/participants", headers=auth, json={"agent_id": "research"})

        assert resp.status_code == 201
        joined = resp.json()["participants"][1]
        assert (joined["agent_id"], joined["lead"]) == ("research", False)
        session = store.get_session(joined["session_id"])
        assert (session.agent_id, session.metadata["room_id"], session.source) == (
            "research",
            room["id"],
            SessionSource.INTERACTIVE.value,
        )
        [_, join] = _events(store, room["id"], "room_member")
        assert (join["action"], join["agent_id"], join["participant_session_id"]) == ("join", "research", session.id)

    @pytest.mark.parametrize(
        "body,status",
        [({"agent_id": "ghost"}, 400), ({}, 400), ({"agent_id": "homelab"}, 409)],
    )
    def test_invite_rejects_unknown_missing_and_duplicate_agents(self, client, auth, room, body, status):
        assert client.post(f"/api/rooms/{room['id']}/participants", headers=auth, json=body).status_code == status

    @pytest.mark.parametrize(
        "body,error",
        [
            ({}, "agent_id must be a string"),
            ({"agent_id": 5}, "agent_id must be a string"),
            ({"agent_id": "ghost"}, "unknown agent: ghost"),
        ],
    )
    def test_invite_error_messages(self, client, auth, room, body, error):
        resp = client.post(f"/api/rooms/{room['id']}/participants", headers=auth, json=body)

        assert (resp.status_code, resp.json()["error"]) == (400, error)

    def test_closed_room_rejects_mute_and_remove(self, client, auth, room):
        base = f"/api/rooms/{room['id']}/participants"
        client.post(base, headers=auth, json={"agent_id": "research"})
        client.post(f"/api/rooms/{room['id']}/close", headers=auth)

        assert client.patch(f"{base}/research", headers=auth, json={"muted": True}).status_code == 409
        assert client.delete(f"{base}/research", headers=auth).status_code == 409

    def test_invite_to_an_unknown_room_is_404(self, client, auth):
        assert (
            client.post("/api/rooms/nope/participants", headers=auth, json={"agent_id": "research"}).status_code == 404
        )

    def test_mute_and_unmute(self, client, auth, room, store):
        client.post(f"/api/rooms/{room['id']}/participants", headers=auth, json={"agent_id": "research"})

        muted = client.patch(f"/api/rooms/{room['id']}/participants/research", headers=auth, json={"muted": True})
        assert muted.json()["participants"][1]["muted"] is True
        client.patch(f"/api/rooms/{room['id']}/participants/research", headers=auth, json={"muted": False})

        assert [e["action"] for e in _events(store, room["id"], "room_member")][-2:] == ["mute", "unmute"]
        assert store.rooms.get(room["id"]).participants[1].muted is False

    def test_the_lead_cannot_be_muted_or_removed(self, client, auth, room):
        base = f"/api/rooms/{room['id']}/participants/homelab"

        assert client.patch(base, headers=auth, json={"muted": True}).status_code == 400
        assert client.delete(base, headers=auth).status_code == 400

    def test_mute_needs_a_boolean_and_a_participant(self, client, auth, room):
        base = f"/api/rooms/{room['id']}/participants"
        client.post(base, headers=auth, json={"agent_id": "research"})

        assert client.patch(f"{base}/research", headers=auth, json={"muted": "yes"}).status_code == 400
        assert client.patch(f"{base}/main", headers=auth, json={"muted": True}).status_code == 404

    def test_remove_ends_the_participant_session(self, client, auth, room, store):
        invited = client.post(f"/api/rooms/{room['id']}/participants", headers=auth, json={"agent_id": "research"})
        session_id = invited.json()["participants"][1]["session_id"]

        resp = client.delete(f"/api/rooms/{room['id']}/participants/research", headers=auth)

        assert [p["agent_id"] for p in resp.json()["participants"]] == ["homelab"]
        assert store.get_session(session_id).status == SessionStatus.COMPLETED.value
        assert _events(store, room["id"], "room_member")[-1]["action"] == "leave"
        assert client.delete(f"/api/rooms/{room['id']}/participants/research", headers=auth).status_code == 404

    def test_user_message_lands_in_the_transcript_events(self, client, auth, room):
        resp = client.post(f"/api/rooms/{room['id']}/messages", headers=auth, json={"content": "hello all"})

        assert resp.status_code == 201
        events = client.get(f"/api/sessions/{room['id']}/events", headers=auth).json()["events"]
        [message] = [e for e in events if e["type"] == "room_message"]
        assert (message["author"], message["content"]) == ("user", "hello all")

    def test_empty_message_is_400(self, client, auth, room):
        assert client.post(f"/api/rooms/{room['id']}/messages", headers=auth, json={"content": " "}).status_code == 400

    def test_close_ends_the_room_and_its_sessions(self, client, auth, room, store):
        resp = client.post(f"/api/rooms/{room['id']}/close", headers=auth)

        assert resp.json()["status"] == SessionStatus.COMPLETED.value
        assert store.get_session(room["participants"][0]["session_id"]).status == SessionStatus.COMPLETED.value
        posted = client.post(f"/api/rooms/{room['id']}/messages", headers=auth, json={"content": "late"})
        assert posted.status_code == 409
        invited = client.post(f"/api/rooms/{room['id']}/participants", headers=auth, json={"agent_id": "research"})
        assert invited.status_code == 409


class TestRoomFrames:
    def test_live_room_events_carry_the_stored_event_id(self, client, auth, room, store, frames):
        client.post(f"/api/rooms/{room['id']}/messages", headers=auth, json={"content": "hi"})

        [live] = [p for kind, p in frames if kind == "session_event" and p["event_type"] == "room_message"]
        [stored] = _events(store, room["id"], "room_message")
        assert live["id"] == stored["id"]
        assert live["session_id"] == room["id"]

    @pytest.mark.parametrize(
        "method,suffix,body",
        [
            ("POST", "/participants", {"agent_id": "research"}),
            ("POST", "/close", None),
        ],
    )
    def test_room_changes_broadcast_the_room(self, client, auth, room, frames, method, suffix, body):
        resp = client.request(method, f"/api/rooms/{room['id']}{suffix}", headers=auth, json=body)

        [update] = [p for kind, p in frames if kind == "session_update" and p["action"] == "room_updated"]
        assert update["id"] == room["id"]
        assert update["room"] == resp.json()

    def test_mute_and_remove_broadcast_the_room(self, client, auth, room, frames):
        base = f"/api/rooms/{room['id']}/participants"
        client.post(base, headers=auth, json={"agent_id": "research"})
        client.patch(f"{base}/research", headers=auth, json={"muted": True})
        removed = client.delete(f"{base}/research", headers=auth).json()

        updates = [p["room"] for kind, p in frames if kind == "session_update" and p["action"] == "room_updated"]
        assert [u["participants"][-1]["muted"] for u in updates[:2]] == [False, True]
        assert updates[-1] == removed


class TestRoomTranscriptIsNotAChat:
    def test_session_lists_leave_out_room_transcripts_unless_asked(self, client, auth, room, store):
        assert room["id"] not in {s.id for s in store.list_sessions()}
        assert [s.id for s in store.list_sessions(source="room")] == [room["id"]]
        listed = {s["id"] for s in client.get("/api/sessions", headers=auth).json()["sessions"]}
        assert room["id"] not in listed

    def test_sessions_command_leaves_out_room_transcripts(self, server, room):
        import asyncio

        from tsugite_daemon.commands import cmd_sessions

        lines = asyncio.run(cmd_sessions(server.adapter)).splitlines()

        assert len(lines) == 1
        assert room["participants"][0]["session_id"][:12] in lines[0]

    @pytest.mark.parametrize("body", [{"title": "x"}, {"status": "completed"}])
    def test_patch_cannot_rename_or_end_a_room_transcript(self, client, auth, room, body):
        resp = client.patch(f"/api/sessions/{room['id']}", headers=auth, json=body)

        assert (resp.status_code, resp.json()["error"]) == (
            400,
            "A room transcript is renamed and ended through the rooms API.",
        )

    def test_patch_can_still_pin_a_room_transcript(self, client, auth, room):
        assert client.patch(f"/api/sessions/{room['id']}", headers=auth, json={"pinned": True}).status_code == 200

    def test_chat_list_leaves_out_room_transcripts_unless_asked(self, client, auth, room):
        def ids(query=""):
            return {s["id"] for s in client.get(f"/api/chat/sessions{query}", headers=auth).json()["sessions"]}

        assert room["id"] not in ids()
        assert room["participants"][0]["session_id"] in ids()
        assert ids("?source=room") == {room["id"]}

    def test_chat_to_a_room_transcript_is_409(self, client, auth, room):
        resp = client.post("/api/chat", headers=auth, json={"session_id": room["id"], "message": "hi"})

        assert resp.status_code == 409
        assert resp.json() == {"error": room_transcript_error(room["id"]), "code": "room_transcript"}

    def test_chat_search_leaves_out_room_transcripts_unless_asked(self, client, auth, room):
        def ids(query):
            return {s["id"] for s in client.get(f"/api/chat/sessions{query}", headers=auth).json()["sessions"]}

        assert room["id"] not in ids("?q=PROJ")
        assert ids("?q=PROJ&source=room") == {room["id"]}

    @pytest.mark.asyncio
    async def test_no_turn_runs_on_a_room_transcript(self, store, room, default_ws):
        class _Adapter(BaseAdapter):
            async def start(self):
                pass

            async def stop(self):
                pass

        adapter = _Adapter(RuntimeDefaults(workspace_dir=default_ws, agent_file="default"), store)
        context = ChannelContext(
            source="session", channel_id=None, user_id="u", reply_to="x", metadata={"conv_id_override": room["id"]}
        )

        with pytest.raises(ValueError, match=re.escape(room_transcript_error(room["id"]))):
            await adapter.handle_message("u", "hi", context)
        assert store.get_session(room["id"]).turn_in_flight is False
