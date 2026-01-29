"""Session management for daemon - maps users to conversations."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict


@dataclass
class SessionInfo:
    """Session tracking information."""

    conversation_id: str
    cumulative_tokens: int = 0
    message_count: int = 0


class SessionManager:
    """Manages per-user conversation sessions with context-based compaction.

    Each agent has its own SessionManager instance.
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

    def get_or_create_session(self, user_id: str) -> str:
        """Get existing session or create new one.

        Args:
            user_id: Platform user ID (e.g., Discord user ID)

        Returns:
            Conversation ID for tsugite history system
        """
        if user_id in self.sessions:
            return self.sessions[user_id].conversation_id

        from tsugite.history import generate_conversation_id

        conv_id = generate_conversation_id(self.agent_name)
        self.sessions[user_id] = SessionInfo(conversation_id=conv_id)
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
        """Compact session: create new session.

        Args:
            user_id: Platform user ID

        Returns:
            New conversation ID
        """
        from tsugite.history import generate_conversation_id

        new_conv_id = generate_conversation_id(self.agent_name)
        self.sessions[user_id] = SessionInfo(conversation_id=new_conv_id)
        return new_conv_id
