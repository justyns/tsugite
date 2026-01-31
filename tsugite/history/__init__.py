"""Session storage V2 - conversation history management."""

from .models import (
    AttachmentRef,
    CompactionSummary,
    ContextSnapshot,
    ContextUpdate,
    SessionMeta,
    SessionRecord,
    Turn,
)
from .reconstruction import (
    apply_cache_control_to_messages,
    dereference_cached_content,
    get_current_context,
    get_turns,
    load_and_apply_history,
    reconstruct_messages,
)
from .storage import (
    SessionStorage,
    generate_session_id,
    get_history_dir,
    get_machine_name,
    list_session_files,
)

__all__ = [
    # Models
    "AttachmentRef",
    "CompactionSummary",
    "ContextSnapshot",
    "ContextUpdate",
    "SessionMeta",
    "SessionRecord",
    "Turn",
    # Storage
    "SessionStorage",
    "generate_session_id",
    "get_history_dir",
    "get_machine_name",
    "list_session_files",
    # Reconstruction
    "apply_cache_control_to_messages",
    "dereference_cached_content",
    "get_current_context",
    "get_turns",
    "load_and_apply_history",
    "reconstruct_messages",
]
