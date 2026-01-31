"""Session management for workspaces."""

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from tsugite.history import SessionStorage

from .models import DEFAULT_COMPACTION_THRESHOLD, Workspace

MAX_CONTEXT_TOKENS = 200000


@dataclass
class SessionInfo:
    """Information about a workspace session."""

    conversation_id: Optional[str]
    message_count: int
    token_estimate: int
    last_agent: Optional[str]
    last_updated: Optional[datetime]


class WorkspaceSession:
    """Manages persistent session for a workspace with file locking."""

    def __init__(self, workspace: Workspace):
        """Initialize session manager.

        Args:
            workspace: Workspace to manage sessions for
        """
        self.workspace = workspace

    def get_conversation_id(self) -> Optional[str]:
        """Get current session ID without creating new one.

        Returns:
            Session ID if exists, None otherwise
        """
        if not self.workspace.session_path.exists():
            return None
        return self._load_session_id()

    def start_new(self) -> str:
        """Start a new session (archives current if exists).

        Returns:
            New session ID
        """
        if self.workspace.session_path.exists():
            self._archive_current_session()
        return self._create_session()

    def should_compact(self, threshold: float = DEFAULT_COMPACTION_THRESHOLD) -> bool:
        """Check if session should be compacted.

        Args:
            threshold: Token usage threshold (0.0-1.0)

        Returns:
            True if session should be compacted
        """
        if not self.workspace.session_path.exists():
            return False

        try:
            storage = SessionStorage.load(self.workspace.session_path)
            return storage.total_tokens >= (MAX_CONTEXT_TOKENS * threshold)
        except Exception:
            return False

    def compact(self) -> str:
        """Compact session using session storage V2.

        Returns:
            New session ID
        """
        if not self.workspace.session_path.exists():
            return self._create_session()

        self._archive_current_session()
        return self._create_session()

    def get_info(self) -> SessionInfo:
        """Get information about current session.

        Returns:
            Session information
        """
        if not self.workspace.session_path.exists():
            return SessionInfo(
                conversation_id=None,
                message_count=0,
                token_estimate=0,
                last_agent=None,
                last_updated=None,
            )

        try:
            storage = SessionStorage.load(self.workspace.session_path)
            return SessionInfo(
                conversation_id=storage.session_id,
                message_count=storage.turn_count,
                token_estimate=storage.total_tokens,
                last_agent=storage.agent,
                last_updated=storage.created_at,
            )
        except Exception:
            return SessionInfo(
                conversation_id=None,
                message_count=0,
                token_estimate=0,
                last_agent=None,
                last_updated=None,
            )

    def list_archived(self) -> List[Path]:
        """List archived session files.

        Returns:
            List of archived session paths, sorted by modification time (newest first)
        """
        if not self.workspace.sessions_dir.exists():
            return []

        sessions = list(self.workspace.sessions_dir.glob("*.jsonl"))
        return sorted(sessions, key=lambda p: p.stat().st_mtime, reverse=True)

    def _load_session_id(self) -> Optional[str]:
        """Load session ID from session file (V2 format only)."""
        try:
            with open(self.workspace.session_path) as f:
                first_line = f.readline().strip()
                if first_line:
                    entry = json.loads(first_line)
                    if entry.get("type") == "session_meta":
                        return self.workspace.session_path.stem
        except Exception:
            pass
        return None

    def _create_session(self) -> str:
        """Create new session.

        Returns:
            New session ID
        """
        storage = SessionStorage.create(
            agent_name="default",
            model="unknown",
            workspace=self.workspace.name,
            session_path=self.workspace.session_path,
        )

        return storage.session_id

    def _archive_current_session(self) -> None:
        """Archive current session to sessions directory."""
        if not self.workspace.session_path.exists():
            return

        self.workspace.sessions_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        archive_path = self.workspace.sessions_dir / f"{timestamp}.jsonl"

        self.workspace.session_path.rename(archive_path)


__all__ = ["WorkspaceSession", "SessionInfo"]
