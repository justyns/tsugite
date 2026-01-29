"""Session management for workspaces."""

import json
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import portalocker

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

        total_tokens = 0

        try:
            with open(self.workspace.session_path) as f:
                for line in f:
                    entry = json.loads(line.strip())
                    if "usage" in entry:
                        total_tokens += entry["usage"].get("total_tokens", 0)

            return total_tokens >= (MAX_CONTEXT_TOKENS * threshold)
        except Exception:
            return False

    def compact(self) -> str:
        """Compact session using memory extraction agent.

        TODO: Implement memory extraction agent integration.
        For now, just archive and create new session.

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

        conversation_id = None
        message_count = 0
        token_estimate = 0
        last_agent = None
        last_updated = None

        try:
            with open(self.workspace.session_path) as f:
                for line in f:
                    entry = json.loads(line.strip())
                    metadata = entry.get("metadata", {})

                    if "conversation_id" in entry:
                        conversation_id = entry["conversation_id"]

                    if "role" in entry:
                        message_count += 1

                    if "usage" in entry:
                        token_estimate += entry["usage"].get("total_tokens", 0)

                    if "agent" in metadata:
                        last_agent = metadata["agent"]

                    if "timestamp" in metadata:
                        last_updated = datetime.fromisoformat(metadata["timestamp"])

        except Exception:
            pass

        return SessionInfo(
            conversation_id=conversation_id,
            message_count=message_count,
            token_estimate=token_estimate,
            last_agent=last_agent,
            last_updated=last_updated,
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
        """Load session ID from first line of session file."""
        try:
            with open(self.workspace.session_path) as f:
                first_line = f.readline().strip()
                if first_line:
                    entry = json.loads(first_line)
                    return entry.get("conversation_id")
        except Exception:
            pass
        return None

    def _create_session(self) -> str:
        """Create new session and write header.

        Returns:
            New session ID
        """
        session_id = str(uuid.uuid4())

        header = {
            "conversation_id": session_id,
            "workspace": self.workspace.name,
            "created_at": datetime.now().isoformat(),
        }

        with portalocker.Lock(self.workspace.session_path, "w", timeout=5) as f:
            f.write(json.dumps(header) + "\n")

        return session_id

    def _archive_current_session(self) -> None:
        """Archive current session to sessions directory."""
        if not self.workspace.session_path.exists():
            return

        self.workspace.sessions_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        archive_path = self.workspace.sessions_dir / f"{timestamp}.jsonl"

        self.workspace.session_path.rename(archive_path)


__all__ = ["WorkspaceSession", "SessionInfo"]
