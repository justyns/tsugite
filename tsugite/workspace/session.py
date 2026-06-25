"""Workspace session management, backed by the global history store.

A workspace's sessions are ordinary history sessions tagged with the workspace name;
the "current" session is simply the most recent one. This replaces the legacy
per-workspace ``session.jsonl`` marker file - workspace conversations now live in the
same store (and search/branch/compaction machinery) as every other mode.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

from tsugite.history import get_history_backend

from .models import DEFAULT_COMPACTION_THRESHOLD, Workspace

MAX_CONTEXT_TOKENS = 200000


@dataclass
class SessionInfo:
    """Information about a workspace session."""

    conversation_id: Optional[str] = None
    message_count: int = 0
    token_estimate: int = 0
    last_agent: Optional[str] = None
    last_updated: Optional[datetime] = None


class WorkspaceSession:
    """The current/archived sessions for a workspace, resolved from the history backend."""

    def __init__(self, workspace: Workspace):
        self.workspace = workspace

    def get_conversation_id(self) -> Optional[str]:
        """Most recent session for this workspace, or None."""
        ids = get_history_backend().list_sessions(workspace=self.workspace.name, limit=1)
        return ids[0] if ids else None

    def start_new(self) -> str:
        """Create a fresh session tagged with this workspace; it becomes the current one."""
        session = get_history_backend().create(agent_name="default", model="unknown", workspace=self.workspace.name)
        return session.session_id

    def should_compact(self, threshold: float = DEFAULT_COMPACTION_THRESHOLD) -> bool:
        cid = self.get_conversation_id()
        if not cid:
            return False
        try:
            summary = get_history_backend().load(cid).summary()
            return summary.total_tokens >= (MAX_CONTEXT_TOKENS * threshold)
        except Exception:
            return False

    def compact(self) -> str:
        """Rotate to a fresh session (the prior one stays in history)."""
        return self.start_new()

    def get_info(self) -> SessionInfo:
        cid = self.get_conversation_id()
        if not cid:
            return SessionInfo()
        try:
            summary = get_history_backend().load(cid).summary()
            return SessionInfo(
                conversation_id=cid,
                message_count=summary.turn_count,
                token_estimate=summary.total_tokens,
                last_agent=summary.agent,
                last_updated=summary.created_at,
            )
        except Exception:
            return SessionInfo()

    def list_archived(self) -> List[str]:
        """Prior sessions for this workspace (most recent first), excluding the current one."""
        ids = get_history_backend().list_sessions(workspace=self.workspace.name)
        return ids[1:]


__all__ = ["WorkspaceSession", "SessionInfo"]
