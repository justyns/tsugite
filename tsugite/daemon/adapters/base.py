"""Base adapter for platform integrations."""

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Protocol

from tsugite.agent_inheritance import find_agent_file
from tsugite.agent_runner import run_agent
from tsugite.daemon.config import AgentConfig
from tsugite.daemon.session import SessionManager
from tsugite.options import ExecutionOptions


class HasUIHandler(Protocol):
    """Protocol for objects with a ui_handler attribute."""

    ui_handler: Any


def resolve_agent_path(agent_file: str, workspace_dir, workspace=None):
    """Resolve agent file reference to absolute path.

    Args:
        agent_file: Agent file name or path (e.g., "default", "default.md", "path/to/agent.md")
        workspace_dir: Workspace directory for search context
        workspace: Optional Workspace object for workspace-aware resolution

    Returns:
        Resolved path to agent file, or None if not found
    """
    agent_ref = agent_file
    if agent_ref.endswith(".md") and "/" not in agent_ref:
        agent_ref = agent_ref[:-3]

    return find_agent_file(agent_ref, current_agent_dir=workspace_dir, workspace=workspace)


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

        # Load workspace for agent resolution
        from tsugite.workspace import Workspace, WorkspaceNotFoundError

        try:
            self._workspace = Workspace.load(agent_config.workspace_dir)
        except WorkspaceNotFoundError:
            self._workspace = None

        # Build workspace attachments once at startup
        from tsugite.workspace.context import build_workspace_attachments

        if self._workspace:
            self.workspace_attachments = build_workspace_attachments(self._workspace)
        else:
            self.workspace_attachments = []

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

        agent_path = resolve_agent_path(
            self.agent_config.agent_file,
            self.agent_config.workspace_dir,
            self._workspace,
        )
        if not agent_path:
            raise ValueError(f"Agent not found: {self.agent_config.agent_file}")

        # Run agent in thread pool to isolate from Discord's event loop
        import concurrent.futures

        from tsugite.cli.helpers import PathContext

        # Set up path context so agent knows its workspace
        workspace_dir = self.agent_config.workspace_dir
        path_context = PathContext(
            invoked_from=workspace_dir,
            workspace_dir=workspace_dir,
            effective_cwd=workspace_dir,
        )

        def run_in_workspace():
            """Run agent in workspace directory (thread-safe via executor isolation)."""
            import os

            original_cwd = os.getcwd()
            try:
                os.chdir(str(workspace_dir))
                return run_agent(
                    agent_path=agent_path,
                    prompt=message,
                    continue_conversation_id=conv_id,
                    attachments=self.workspace_attachments,
                    exec_options=ExecutionOptions(
                        return_token_usage=True,
                    ),
                    path_context=path_context,
                )
            finally:
                os.chdir(original_cwd)

        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            result = await loop.run_in_executor(executor, run_in_workspace)

        # Save conversation history (daemon was missing this, unlike CLI)
        try:
            from tsugite.agent_runner.history_integration import save_run_to_history
            from tsugite.agent_runner.validation import get_agent_info

            agent_info = get_agent_info(agent_path)
            save_run_to_history(
                agent_path=agent_path,
                agent_name=self.agent_name,
                prompt=message,
                result=str(result),
                model=agent_info.get("model", "unknown"),
                token_count=result.token_count if hasattr(result, "token_count") else None,
                cost=result.cost if hasattr(result, "cost") else None,
                execution_steps=result.execution_steps if hasattr(result, "execution_steps") else None,
                continue_conversation_id=conv_id,
                system_prompt=result.system_message if hasattr(result, "system_message") else None,
                attachments=result.attachments if hasattr(result, "attachments") else None,
                channel_metadata=metadata,
            )
        except Exception as e:
            import sys

            print(f"Warning: Failed to save daemon history: {e}", file=sys.stderr)

        if hasattr(result, "token_usage") and result.token_usage:
            self.session_manager.update_token_count(user_id, result.token_usage.total_tokens)

        return str(result)

    async def _compact_session(self, user_id: str) -> None:
        """Compact session when approaching context limit.

        1. Load session and reconstruct messages
        2. Summarize conversation
        3. Create new session with compaction summary

        Args:
            user_id: Platform user ID
        """
        from tsugite.daemon.memory import summarize_session
        from tsugite.history import SessionStorage, get_history_dir, reconstruct_messages

        old_conv_id = self.session_manager.sessions[user_id].conversation_id
        session_info = self.session_manager.sessions[user_id]
        old_session_path = get_history_dir() / f"{old_conv_id}.jsonl"

        messages = reconstruct_messages(old_session_path)
        summary = await summarize_session(messages)

        new_conv_id = self.session_manager.compact_session(user_id)

        new_session_path = get_history_dir() / f"{new_conv_id}.jsonl"
        new_storage = SessionStorage.create(
            agent_name=self.agent_name,
            model=self.agent_config.agent_file,
            compacted_from=old_conv_id,
            session_path=new_session_path,
        )

        new_storage.record_compaction_summary(summary, session_info.message_count)
