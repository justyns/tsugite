"""Session management for daemon - maps users to sessions."""

import json
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

from tsugite.history import SessionStorage, Turn, generate_session_id, get_history_dir


@dataclass
class SessionInfo:
    """Session tracking information."""

    conversation_id: str
    cumulative_tokens: int = 0
    message_count: int = 0


class SessionManager:
    """Manages per-user conversation sessions with context-based compaction.

    Each agent has its own SessionManager instance.
    Sessions are persisted to disk so they survive daemon restarts.
    """

    def __init__(self, agent_name: str, workspace_dir: Path, context_limit: int = 128000):
        """Initialize session manager.

        Args:
            agent_name: Name of the agent this manager is for
            workspace_dir: Agent's workspace directory
            context_limit: Maximum context window in tokens (default 128k)
        """
        self.agent_name = agent_name
        self.workspace_dir = workspace_dir
        self.context_limit = context_limit
        self.compaction_threshold = int(context_limit * 0.8)  # 80%
        self.sessions: Dict[str, SessionInfo] = {}  # user_id → session info
        self._lock = threading.Lock()

        # Session storage directory
        self.sessions_dir = workspace_dir / "daemon_sessions"
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

    def _get_session_file(self, user_id: str) -> Path:
        """Get path to session file for a user."""
        safe_id = user_id.replace(":", "_")
        return self.sessions_dir / f"{safe_id}.json"

    def _load_session_file(self, user_id: str) -> Optional[dict]:
        """Load session data from disk."""
        path = self._get_session_file(user_id)
        if path.exists():
            try:
                return json.loads(path.read_text())
            except (json.JSONDecodeError, OSError):
                return None
        return None

    def _save_session_file(self, user_id: str, conv_id: str, compaction_count: int = 0) -> None:
        """Save session data to disk."""
        path = self._get_session_file(user_id)
        data = {
            "conversation_id": conv_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "compaction_count": compaction_count,
        }
        path.write_text(json.dumps(data))

    def _estimate_tokens_from_storage(self, session_id: str) -> tuple[int, int]:
        """Estimate token count from session storage.

        Returns:
            Tuple of (estimated_tokens, message_count)
        """
        try:
            session_path = get_history_dir() / f"{session_id}.jsonl"
            if not session_path.exists():
                return 0, 0

            storage = SessionStorage.load(session_path)
            turns = [r for r in storage.load_records() if isinstance(r, Turn)]
            last_tokens = turns[-1].tokens if turns and turns[-1].tokens else 0
            return last_tokens, storage.turn_count
        except Exception:
            return 0, 0

    def get_or_create_session(self, user_id: str) -> str:
        """Get existing session or create new one.

        Loads session from disk if not in memory, allowing persistence
        across daemon restarts.

        Args:
            user_id: Platform user ID (e.g., Discord user ID)

        Returns:
            Session ID for history system
        """
        with self._lock:
            if user_id in self.sessions:
                return self.sessions[user_id].conversation_id

            # Try to load from file
            session_data = self._load_session_file(user_id)
            if session_data:
                conv_id = session_data["conversation_id"]
                initial_tokens, initial_messages = self._estimate_tokens_from_storage(conv_id)
            else:
                conv_id = f"daemon_{self.agent_name}_{user_id}"
                self._save_session_file(user_id, conv_id)
                initial_tokens = 0
                initial_messages = 0

            self.sessions[user_id] = SessionInfo(
                conversation_id=conv_id,
                cumulative_tokens=initial_tokens,
                message_count=initial_messages,
            )
            return conv_id

    def update_context_limit(self, limit: int) -> None:
        """Update context limit and recalculate compaction threshold."""
        self.context_limit = limit
        self.compaction_threshold = int(limit * 0.8)

    def update_token_count(self, user_id: str, tokens_used: int) -> None:
        """Update cumulative token count for session.

        Args:
            user_id: Platform user ID
            tokens_used: Number of tokens used in this turn
        """
        with self._lock:
            if user_id in self.sessions:
                self.sessions[user_id].cumulative_tokens = max(self.sessions[user_id].cumulative_tokens, tokens_used)
                self.sessions[user_id].message_count += 1

    def needs_compaction(self, user_id: str) -> bool:
        """Check if session needs compaction (>80% of context limit).

        Args:
            user_id: Platform user ID

        Returns:
            True if session should be compacted
        """
        with self._lock:
            if user_id not in self.sessions:
                return False
            return self.sessions[user_id].cumulative_tokens >= self.compaction_threshold

    def compact_session(self, user_id: str) -> str:
        """Compact session: create new session and update persistent storage.

        Args:
            user_id: Platform user ID

        Returns:
            New session ID
        """
        with self._lock:
            session_data = self._load_session_file(user_id) or {}
            compaction_count = session_data.get("compaction_count", 0) + 1

            new_conv_id = generate_session_id(self.agent_name)

            self._save_session_file(user_id, new_conv_id, compaction_count)

            self.sessions[user_id] = SessionInfo(conversation_id=new_conv_id)
            return new_conv_id
