"""Conversation history management."""

from .index import (
    get_conversation_metadata,
    query_index,
    rebuild_index,
    update_index,
)
from .models import ConversationMetadata, IndexEntry, Turn
from .storage import (
    generate_conversation_id,
    get_history_dir,
    load_conversation,
    save_turn_to_history,
)

__all__ = [
    "ConversationMetadata",
    "IndexEntry",
    "Turn",
    "generate_conversation_id",
    "get_conversation_metadata",
    "get_history_dir",
    "load_conversation",
    "query_index",
    "rebuild_index",
    "save_turn_to_history",
    "update_index",
]
