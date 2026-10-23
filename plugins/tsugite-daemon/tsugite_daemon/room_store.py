"""Rooms: group chats between agents, persisted in daemon.db (`rooms` collection).

A room's id is also the id of its transcript, a `SessionSource.ROOM` session.
"""

import threading
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Optional

from tsugite.core.record_store import SqliteCollectionStorage

DEFAULT_MAX_ROUNDS = 12


@dataclass
class Participant:
    agent_id: str
    session_id: str
    lead: bool = False
    muted: bool = False


@dataclass
class Room:
    id: str
    participants: list[Participant] = field(default_factory=list)
    max_rounds: int = DEFAULT_MAX_ROUNDS
    rounds_used: int = 0

    def __post_init__(self):
        if self.max_rounds < 1:
            raise ValueError("max_rounds must be at least 1")

    def participant(self, agent_id: str) -> Optional[Participant]:
        return next((p for p in self.participants if p.agent_id == agent_id), None)


class RoomStore:
    def __init__(self, path: Path):
        self._storage = SqliteCollectionStorage.for_state_file(path, "rooms")
        self._lock = threading.Lock()
        self._rooms: dict[str, Room] = {}
        for entry in self._storage.load_all():
            entry["participants"] = [Participant(**p) for p in entry["participants"]]
            room = Room(**entry)
            self._rooms[room.id] = room

    def list(self) -> list[Room]:
        with self._lock:
            return list(self._rooms.values())

    def get(self, room_id: str) -> Optional[Room]:
        with self._lock:
            return self._rooms.get(room_id)

    def create(self, room: Room) -> Room:
        with self._lock:
            self._save(room)
        return room

    def update(self, room_id: str, **fields) -> Room:
        with self._lock:
            room = self._rooms.get(room_id)
            if room is None:
                raise ValueError(f"Unknown room: {room_id}")
            room = replace(room, **fields)
            self._save(room)
        return room

    def _save(self, room: Room) -> None:
        self._storage.upsert(room.id, asdict(room))
        self._rooms[room.id] = room
