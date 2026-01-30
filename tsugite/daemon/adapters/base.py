"""Base adapter for platform integrations."""

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Protocol

from tsugite.agent_runner import run_agent_async
from tsugite.daemon.config import AgentConfig
from tsugite.daemon.session import SessionManager
from tsugite.options import ExecutionOptions


class HasUIHandler(Protocol):
    """Protocol for objects with a ui_handler attribute."""

    ui_handler: Any


@dataclass
class ChannelContext:
    """Channel routing context for replies.

    Contains information about where a message came from and where replies
    should be sent. Allows multi-channel conversation continuity.

    Attributes:
        source: Platform identifier (discord, cli, slack, etc.)
        channel_id: Platform-specific channel identifier
        user_id: User identifier
        reply_to: Formatted reply target (e.g., "discord:123456789")
        metadata: Additional platform-specific metadata
    """

    source: str
    channel_id: Optional[str]
    user_id: str
    reply_to: str
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for history storage.

        Returns:
            Dictionary with channel metadata including timestamp
        """
        result = {
            "source": self.source,
            "channel_id": self.channel_id,
            "user_id": self.user_id,
            "reply_to": self.reply_to,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if self.metadata:
            result.update(self.metadata)
        # Set is_daemon_managed after merge to prevent override
        result["is_daemon_managed"] = True
        return result


class BaseAdapter(ABC):
    """Base class for platform adapters.

    Each adapter instance is tied to a specific agent.
    """

    def __init__(self, agent_name: str, agent_config: AgentConfig, session_manager: SessionManager):
        """Initialize base adapter.

        Args:
            agent_name: Name of the agent this adapter is for
            agent_config: Agent configuration
            session_manager: Session manager for this agent
        """
        self.agent_name = agent_name
        self.agent_config = agent_config
        self.session_manager = session_manager

        # Build workspace attachments once at startup
        from tsugite.daemon.memory import get_workspace_attachments
        from tsugite.utils import resolve_attachments

        workspace_files = get_workspace_attachments(
            workspace_dir=agent_config.workspace_dir,
            memory_enabled=agent_config.memory_enabled,
            inject_days=agent_config.memory_inject_days,
        )
        self.workspace_attachments = resolve_attachments(workspace_files) if workspace_files else []

    @abstractmethod
    async def start(self) -> None:
        """Start the adapter."""
        pass

    @abstractmethod
    async def stop(self) -> None:
        """Stop the adapter."""
        pass

    async def handle_message(
        self, user_id: str, message: str, channel_context: ChannelContext, custom_logger: Optional[HasUIHandler] = None
    ) -> str:
        """Common message handling with session compaction support.

        Args:
            user_id: Platform user ID
            message: User's message
            channel_context: Channel routing information
            custom_logger: Optional custom logger with ui_handler for UI events

        Returns:
            Agent's response
        """
        if self.session_manager.needs_compaction(user_id):
            await self._compact_session(user_id)

        conv_id = self.session_manager.get_or_create_session(user_id)

        metadata = channel_context.to_dict()
        metadata["daemon_agent"] = self.agent_name

        agent_path = self.agent_config.workspace_dir / self.agent_config.agent_file
        result = await run_agent_async(
            agent_path=agent_path,
            prompt=message,
            continue_conversation_id=conv_id,
            attachments=self.workspace_attachments,
            exec_options=ExecutionOptions(
                return_token_usage=True,  # Need tokens for tracking
            ),
            channel_metadata=metadata,
            custom_logger=custom_logger,
        )

        if hasattr(result, "token_usage") and result.token_usage:
            self.session_manager.update_token_count(user_id, result.token_usage.total_tokens)

        session_info = self.session_manager.sessions.get(user_id)
        if (
            session_info
            and self.agent_config.memory_enabled
            and session_info.message_count % self.agent_config.memory_extraction_interval == 0
        ):
            # Run memory extraction in background (don't await)
            memory_agent_path = self.agent_config.workspace_dir / "memory_extraction.md"
            if memory_agent_path.exists():
                from tsugite.daemon.memory import extract_memories
                from tsugite.history import load_conversation

                history = load_conversation(conv_id)
                asyncio.create_task(extract_memories(history, self.agent_config.workspace_dir, memory_agent_path))

        return str(result)

    async def _compact_session(self, user_id: str) -> None:
        """Compact session when approaching context limit.

        1. Load conversation history
        2. Summarize conversation
        3. Extract memories (background agent)
        4. Create new session with summary as first message

        Args:
            user_id: Platform user ID
        """
        from tsugite.daemon.memory import extract_memories, summarize_session
        from tsugite.history import load_conversation
        from tsugite.history.models import Turn

        old_conv_id = self.session_manager.sessions[user_id].conversation_id
        session_info = self.session_manager.sessions[user_id]

        history = load_conversation(old_conv_id)
        summary = await summarize_session(history)

        if self.agent_config.memory_enabled:
            memory_agent_path = self.agent_config.workspace_dir / "memory_extraction.md"
            if memory_agent_path.exists():
                await extract_memories(history, self.agent_config.workspace_dir, memory_agent_path)

        new_conv_id = self.session_manager.compact_session(user_id)

        from tsugite.history.storage import save_turn_to_history

        summary_turn = Turn(
            timestamp=datetime.now(timezone.utc),
            user="[Session Summary]",
            assistant=f"Previous session summary:\n\n{summary}",
            metadata={
                "type": "compaction_summary",
                "source_session": old_conv_id,
                "compacted_at": datetime.now(timezone.utc).isoformat(),
                "original_message_count": session_info.message_count,
                "is_daemon_managed": True,
                "daemon_agent": self.agent_name,
            },
        )

        save_turn_to_history(new_conv_id, summary_turn)
