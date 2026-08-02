"""Per-event session history."""

from .base import HistoryBackend, Session
from .models import Event, SessionSummary
from .reconstruction import events_to_messages, last_index_of, reconstruct_raw_turns
from .registry import get_history_backend, reset_history_backend, set_history_backend
from .sqlite_backend import SqliteHistoryBackend, generate_session_id, get_history_dir
from .ui_events import event_to_ui_dict

__all__ = [
    "Event",
    "HistoryBackend",
    "Session",
    "SessionSummary",
    "SqliteHistoryBackend",
    "event_to_ui_dict",
    "events_to_messages",
    "reconstruct_raw_turns",
    "generate_session_id",
    "get_history_backend",
    "get_history_dir",
    "last_index_of",
    "reset_history_backend",
    "set_history_backend",
]
