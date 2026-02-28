"""Base adapter for platform integrations."""

import asyncio
import contextvars
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Protocol

from tsugite.agent_inheritance import find_agent_file
from tsugite.agent_runner import run_agent
from tsugite.daemon.config import AgentConfig
from tsugite.daemon.session import SessionManager
from tsugite.options import ExecutionOptions

logger = logging.getLogger(__name__)


class HasUIHandler(Protocol):
    """Protocol for objects with a ui_handler attribute."""

    ui_handler: Any


def resolve_agent_path(agent_file: str, workspace_dir: Path, workspace: Any = None) -> Optional[Path]:
    """Resolve agent file reference to absolute path.

    Args:
        agent_file: Agent file name or path (e.g., "default", "+default", "default.md", "path/to/agent.md")
        workspace_dir: Workspace directory for search context
        workspace: Optional Workspace object for workspace-aware resolution

    Returns:
        Resolved path to agent file, or None if not found
    """
    agent_ref = agent_file.lstrip("+")
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

    def __init__(
        self,
        agent_name: str,
        agent_config: AgentConfig,
        session_manager: SessionManager,
        identity_map: Optional[Dict[str, str]] = None,
    ):
        """Initialize base adapter.

        Args:
            agent_name: Name of the agent this adapter is for
            agent_config: Agent configuration
            session_manager: Session manager for this agent
            identity_map: Reverse lookup from "source:platform_id" to canonical name
        """
        self.agent_name = agent_name
        self.agent_config = agent_config
        self.session_manager = session_manager
        self._identity_map = identity_map or {}

        from tsugite.workspace import Workspace, WorkspaceNotFoundError
        from tsugite.workspace.context import build_workspace_attachments

        try:
            self._workspace = Workspace.load(agent_config.workspace_dir)
        except WorkspaceNotFoundError:
            self._workspace = None

        self.workspace_attachments = build_workspace_attachments(self._workspace) if self._workspace else []

    def _resolve_agent_path(self, agent_file: Optional[str] = None) -> Optional[Path]:
        """Resolve an agent file to an absolute path.

        Args:
            agent_file: Agent file to resolve. Defaults to the configured agent_file.
        """
        return resolve_agent_path(
            agent_file or self.agent_config.agent_file,
            self.agent_config.workspace_dir,
            self._workspace,
        )

    def resolve_model(self) -> str:
        """Resolve the effective model name, returning 'unknown' on failure.

        Prefers daemon config model override, then falls back to the agent file's model.
        """
        if self.agent_config.model:
            return self.agent_config.model

        from tsugite.agent_runner.validation import get_agent_info

        agent_path = self._resolve_agent_path()
        if not agent_path:
            return "unknown"
        try:
            return get_agent_info(agent_path).get("model", "unknown")
        except Exception:
            return "unknown"

    @abstractmethod
    async def start(self) -> None:
        """Start the adapter."""

    @abstractmethod
    async def stop(self) -> None:
        """Stop the adapter."""

    @staticmethod
    def _emit_ui(custom_logger: Optional[HasUIHandler], event_type: str) -> None:
        """Emit a UI event if a custom logger with ui_handler is available."""
        if custom_logger and hasattr(custom_logger, "ui_handler"):
            custom_logger.ui_handler._emit(event_type, {})

    def _build_agent_context(self, channel_context: ChannelContext) -> Dict[str, Any]:
        """Build context dict for agent template rendering."""
        ctx: Dict[str, Any] = {"is_daemon": True, "is_scheduled": False, "schedule_id": "", "has_notify_tool": False}
        meta = channel_context.metadata or {}
        if channel_context.source == "scheduler":
            ctx["is_scheduled"] = True
            ctx["schedule_id"] = meta.get("schedule_id", "")
            ctx["has_notify_tool"] = meta.get("notify_tool", False)
        ctx["running_tasks"] = meta.get("running_tasks", [])
        return ctx

    def _build_message_context(self, message: str, channel_context: ChannelContext, user_id: str) -> str:
        """Prepend per-message dynamic context to the user prompt.

        Keeps dynamic metadata in the user message turn (not the cached
        attachment context turn) for better cache efficiency.
        """
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        session = self.session_manager.sessions.get(user_id)
        tokens_used = session.cumulative_tokens if session else 0

        return f"""<message_context>
  <datetime>{timestamp}</datetime>
  <working_directory>{self.agent_config.workspace_dir}</working_directory>
  <source>{channel_context.source}</source>
  <user_id>{channel_context.user_id}</user_id>
  <context_tokens_used>{tokens_used}</context_tokens_used>
  <context_limit>{self.agent_config.context_limit}</context_limit>
</message_context>

{message}"""

    def resolve_user(self, user_id: str, channel_context: ChannelContext) -> str:
        """Resolve platform user ID to canonical identity.

        Group chats stay isolated (keyed by source:channel:user). DMs resolve
        via identity_map, falling back to bare user_id for backward compat.
        """
        is_group = channel_context.metadata and channel_context.metadata.get("guild_id")
        if is_group:
            return f"{channel_context.source}:{channel_context.channel_id}:{user_id}"
        return self._identity_map.get(f"{channel_context.source}:{user_id}", user_id)

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
        user_id = self.resolve_user(user_id, channel_context)

        if self.session_manager.needs_compaction(user_id):
            self._emit_ui(custom_logger, "compacting")
            await self._compact_session(user_id)
            self._emit_ui(custom_logger, "compacted")

        conv_id = self.session_manager.get_or_create_session(user_id)

        metadata = channel_context.to_dict()
        metadata["daemon_agent"] = self.agent_name

        agent_path = self._resolve_agent_path()
        if channel_context.metadata and channel_context.metadata.get("agent_file_override"):
            override = Path(channel_context.metadata["agent_file_override"])
            if override.exists():
                agent_path = override
        if not agent_path:
            raise ValueError(f"Agent not found: {self.agent_config.agent_file}")

        enriched_prompt = self._build_message_context(message, channel_context, user_id)

        from tsugite.cli.helpers import PathContext

        workspace_dir = self.agent_config.workspace_dir
        path_context = PathContext(
            invoked_from=workspace_dir,
            workspace_dir=workspace_dir,
            effective_cwd=workspace_dir,
        )

        def run_in_workspace():
            """Run agent in workspace directory (thread-safe via executor isolation)."""
            original_cwd = os.getcwd()
            try:
                os.chdir(str(workspace_dir))
                return run_agent(
                    agent_path=agent_path,
                    prompt=enriched_prompt,
                    continue_conversation_id=conv_id,
                    attachments=self.workspace_attachments,
                    # TODO: Support sandbox options from daemon agent config
                    # (sandbox, allow_domains, no_network) — see ROADMAP.md
                    exec_options=ExecutionOptions(
                        return_token_usage=True,
                        model_override=(channel_context.metadata or {}).get("model_override")
                        or self.agent_config.model,
                        max_turns_override=self.agent_config.max_turns,
                    ),
                    path_context=path_context,
                    custom_logger=custom_logger,
                    context=self._build_agent_context(channel_context),
                )
            finally:
                os.chdir(original_cwd)

        ctx = contextvars.copy_context()
        result = await asyncio.to_thread(ctx.run, run_in_workspace)

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
                token_count=getattr(result, "token_count", None),
                cost=getattr(result, "cost", None),
                execution_steps=getattr(result, "execution_steps", None),
                continue_conversation_id=conv_id,
                system_prompt=getattr(result, "system_message", None),
                attachments=getattr(result, "attachments", None),
                channel_metadata=metadata,
            )
        except Exception as e:
            logger.warning("Failed to save daemon history: %s", e)

        if result.token_count:
            self.session_manager.update_token_count(user_id, result.token_count)

        return str(result)

    async def _compact_session(self, user_id: str) -> None:
        """Compact session when approaching context limit.

        1. Load session and reconstruct messages
        2. Summarize conversation
        3. Create new session with compaction summary

        Args:
            user_id: Platform user ID
        """
        from tsugite.daemon.memory import infer_compaction_model, summarize_session
        from tsugite.history import SessionStorage, get_history_dir, reconstruct_messages

        model = self.agent_config.compaction_model or infer_compaction_model(self.resolve_model())

        session_info = self.session_manager.sessions[user_id]
        old_conv_id = session_info.conversation_id
        old_session_path = get_history_dir() / f"{old_conv_id}.jsonl"

        logger.info("[%s] Compacting session (%d messages)...", self.agent_name, session_info.message_count)
        messages = reconstruct_messages(old_session_path)
        summary = await summarize_session(messages, model=model, max_context_tokens=self.agent_config.context_limit)
        logger.info("[%s] Session compacted", self.agent_name)

        new_conv_id = self.session_manager.compact_session(user_id)

        new_session_path = get_history_dir() / f"{new_conv_id}.jsonl"
        new_storage = SessionStorage.create(
            agent_name=self.agent_name,
            model=self.agent_config.agent_file,
            compacted_from=old_conv_id,
            session_path=new_session_path,
        )

        new_storage.record_compaction_summary(summary, session_info.message_count)
