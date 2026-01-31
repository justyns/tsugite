"""Session management for daemon - maps users to conversations."""

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional


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

        # Session storage directory
        self.sessions_dir = workspace_dir / "daemon_sessions"
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

    def _get_session_file(self, user_id: str) -> Path:
        """Get path to session file for a user."""
        return self.sessions_dir / f"{user_id}.json"

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

    def _estimate_tokens_from_history(self, conv_id: str) -> tuple[int, int]:
        """Estimate token count from conversation history.

        Returns:
            Tuple of (estimated_tokens, message_count)
        """
        try:
            from tsugite.history import load_conversation

            turns = load_conversation(conv_id)
            tokens = sum((len(t.user or "") + len(t.assistant or "")) // 4 for t in turns if hasattr(t, "user"))
            return tokens, len(turns)
        except Exception:
            return 0, 0

    def get_or_create_session(self, user_id: str) -> str:
        """Get existing session or create new one.

        Loads session from disk if not in memory, allowing persistence
        across daemon restarts.

        Args:
            user_id: Platform user ID (e.g., Discord user ID)

        Returns:
            Conversation ID for tsugite history system
        """
        if user_id in self.sessions:
            return self.sessions[user_id].conversation_id

        # Try to load from file
        session_data = self._load_session_file(user_id)
        if session_data:
            conv_id = session_data["conversation_id"]
            # Estimate tokens from history to restore state
            initial_tokens, initial_messages = self._estimate_tokens_from_history(conv_id)
        else:
            # New session - use deterministic ID
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

    def update_token_count(self, user_id: str, tokens_used: int) -> None:
        """Update cumulative token count for session.

        Args:
            user_id: Platform user ID
            tokens_used: Number of tokens used in this turn
        """
        if user_id in self.sessions:
            self.sessions[user_id].cumulative_tokens += tokens_used
            self.sessions[user_id].message_count += 1

    def needs_compaction(self, user_id: str) -> bool:
        """Check if session needs compaction (>80% of context limit).

        Args:
            user_id: Platform user ID

        Returns:
            True if session should be compacted
        """
        if user_id not in self.sessions:
            return False
        return self.sessions[user_id].cumulative_tokens >= self.compaction_threshold

    def compact_session(self, user_id: str) -> str:
        """Compact session: create new conversation and update persistent storage.

        Args:
            user_id: Platform user ID

        Returns:
            New conversation ID
        """
        from tsugite.history import generate_conversation_id

        # Load current session data to get compaction count
        session_data = self._load_session_file(user_id) or {}
        compaction_count = session_data.get("compaction_count", 0) + 1

        # Generate new conversation ID
        new_conv_id = generate_conversation_id(self.agent_name)

        # Update persistent storage
        self._save_session_file(user_id, new_conv_id, compaction_count)

        # Update in-memory state
        self.sessions[user_id] = SessionInfo(conversation_id=new_conv_id)
        return new_conv_id
