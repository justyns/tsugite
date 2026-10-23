"""Room membership and the shared transcript."""

from dataclasses import asdict, replace
from datetime import datetime, timezone
from typing import Optional

from tsugite.history import generate_session_id
from tsugite_daemon.agent_store import UnknownAgentError
from tsugite_daemon.room_store import DEFAULT_MAX_ROUNDS, Participant, Room
from tsugite_daemon.session_store import (
    FINISHED_STATUSES,
    METADATA_ROOM_ID,
    SESSION_ID_LABEL,
    Session,
    SessionSource,
    SessionStatus,
    SessionStore,
)

USER_AUTHOR = "user"


class RoomLookupError(LookupError):
    pass


class RoomConflictError(ValueError):
    pass


class Rooms:
    def __init__(self, store: SessionStore, event_bus):
        self._store = store
        self._event_bus = event_bus

    def get(self, room_id: str) -> Room:
        room = self._store.rooms.get(room_id)
        if room is None:
            raise RoomLookupError(f"Unknown room: {room_id}")
        return room

    def list_details(self) -> list[dict]:
        return [self.detail(room) for room in self._store.rooms.list()]

    def detail(self, room: Room) -> dict:
        transcript = self._store.get_session(room.id)
        return {
            **asdict(room),
            "title": transcript.title,
            "status": transcript.status,
            "created_at": transcript.created_at,
        }

    def create(
        self,
        title: str,
        *,
        user_id: Optional[str],
        lead_agent_id: str,
        max_rounds: int = DEFAULT_MAX_ROUNDS,
    ) -> Room:
        self._require_agent(lead_agent_id)
        room = Room(id=generate_session_id("room"), max_rounds=max_rounds)
        self._store.create_session(Session(id=room.id, source=SessionSource.ROOM.value, user_id=user_id, title=title))
        self._store.rooms.create(room)
        self._emit_update("created", room.id)
        return self._join(room, lead_agent_id, lead=True)

    def invite(self, room_id: str, agent_id: str) -> Room:
        room = self._open_room(room_id)
        self._require_agent(agent_id)
        if room.participant(agent_id):
            raise RoomConflictError(f"{agent_id} is already in room {room_id}")
        return self._join(room, agent_id, lead=False)

    def remove(self, room_id: str, agent_id: str) -> Room:
        room = self._open_room(room_id)
        participant = self._non_lead(room, agent_id, "removed")
        room = self._store.rooms.update(room_id, participants=[p for p in room.participants if p is not participant])
        self._end_session(participant.session_id)
        self._record_member(room_id, "leave", participant)
        return self._updated(room)

    def set_muted(self, room_id: str, agent_id: str, muted: bool) -> Room:
        room = self._open_room(room_id)
        participant = self._non_lead(room, agent_id, "muted")
        updated = replace(participant, muted=muted)
        room = self._store.rooms.update(
            room_id, participants=[updated if p is participant else p for p in room.participants]
        )
        self._record_member(room_id, "mute" if muted else "unmute", updated)
        return self._updated(room)

    def post_message(
        self,
        room_id: str,
        content: str,
        *,
        author: str = USER_AUTHOR,
        participant_session_id: Optional[str] = None,
        private_ref: Optional[int] = None,
    ) -> None:
        self._open_room(room_id)
        self._record(
            room_id,
            "room_message",
            {
                "author": author,
                "participant_session_id": participant_session_id,
                "content": content,
                "private_ref": private_ref,
            },
        )
        self._store.update_session(room_id)

    def close(self, room_id: str) -> Room:
        room = self._open_room(room_id)
        for participant in room.participants:
            self._end_session(participant.session_id)
        self._end_session(room_id)
        return self._updated(room)

    def _open_room(self, room_id: str) -> Room:
        room = self.get(room_id)
        if self._store.get_session(room_id).status in FINISHED_STATUSES:
            raise RoomConflictError(f"Room {room_id} is closed")
        return room

    def _require_agent(self, agent_id: str) -> None:
        if self._store.agents.get(agent_id) is None:
            raise UnknownAgentError(f"unknown agent: {agent_id}")

    @staticmethod
    def _non_lead(room: Room, agent_id: str, verb: str) -> Participant:
        participant = room.participant(agent_id)
        if participant is None:
            raise RoomLookupError(f"{agent_id} is not in room {room.id}")
        if participant.lead:
            raise ValueError(f"The lead cannot be {verb}")
        return participant

    def _join(self, room: Room, agent_id: str, *, lead: bool) -> Room:
        session = self._store.create_session(
            Session(
                id=generate_session_id(SESSION_ID_LABEL),
                agent_id=agent_id,
                title=self._store.get_session(room.id).title,
                metadata={METADATA_ROOM_ID: room.id},
            )
        )
        self._emit_update("created", session.id)
        participant = Participant(agent_id=agent_id, session_id=session.id, lead=lead)
        room = self._store.rooms.update(room.id, participants=[*room.participants, participant])
        self._record_member(room.id, "join", participant)
        return self._updated(room)

    def _end_session(self, session_id: str) -> None:
        live = self._store.resolve_live(session_id)
        if live is not None and live.status not in FINISHED_STATUSES:
            self._store.update_session(live.id, status=SessionStatus.COMPLETED.value)
            self._emit_update("completed", live.id)

    def _record_member(self, room_id: str, action: str, participant: Participant) -> None:
        self._record(
            room_id,
            "room_member",
            {"action": action, "agent_id": participant.agent_id, "participant_session_id": participant.session_id},
        )

    def _record(self, room_id: str, event_type: str, data: dict) -> None:
        event_id = self._store.append_event(
            room_id, {"type": event_type, "timestamp": datetime.now(timezone.utc).isoformat(), **data}
        )
        if self._event_bus:
            self._event_bus.emit(
                "session_event", {"session_id": room_id, "event_type": event_type, "id": event_id, **data}
            )

    def _updated(self, room: Room) -> Room:
        if self._event_bus:
            self._event_bus.emit("session_update", {"action": "room_updated", "id": room.id, "room": self.detail(room)})
        return room

    def _emit_update(self, action: str, session_id: str) -> None:
        if self._event_bus:
            self._event_bus.emit("session_update", {"action": action, "id": session_id})
