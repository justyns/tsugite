"""Per-event session history."""

from .base import HistoryBackend, Session
from .models import Event
from .reconstruction import events_to_messages, last_index_of
from .registry import get_history_backend, reset_history_backend, set_history_backend
from .storage import (
    JsonlHistoryBackend,
    SessionStorage,
    SessionSummary,
    generate_session_id,
    get_history_dir,
    get_machine_name,
    list_session_files,
)

__all__ = [
    "Event",
    "HistoryBackend",
    "JsonlHistoryBackend",
    "Session",
    "SessionStorage",
    "SessionSummary",
    "events_to_messages",
    "generate_session_id",
    "get_history_backend",
    "get_history_dir",
    "get_machine_name",
    "last_index_of",
    "list_session_files",
    "reset_history_backend",
    "set_history_backend",
]
