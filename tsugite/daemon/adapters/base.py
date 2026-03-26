"""Base adapter for platform integrations."""

import asyncio
import contextvars
import json
import logging
import os
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, Protocol
from zoneinfo import ZoneInfo

from tzlocal import get_localzone

from tsugite.agent_inheritance import find_agent_file
from tsugite.agent_runner import run_agent
from tsugite.daemon.config import AgentConfig
from tsugite.daemon.session_store import SessionStore
from tsugite.exceptions import AgentExecutionError
from tsugite.options import ExecutionOptions

if TYPE_CHECKING:
    from tsugite.history.models import Turn

logger = logging.getLogger(__name__)


def _is_recent(iso_timestamp: str, minutes: int = 10, now: datetime = None) -> bool:
    """Check if an ISO timestamp is within the last N minutes."""
    if not iso_timestamp:
        return False
    try:
        dt = datetime.fromisoformat(iso_timestamp)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        if now is None:
            now = datetime.now(timezone.utc)
        return (now - dt) < timedelta(minutes=minutes)
    except (ValueError, TypeError):
        return False


def _write_turns_tempfile(turns: "list[Turn]") -> Path:
    """Write turn messages to a temp JSONL file for hook consumption."""
    fd, path = tempfile.mkstemp(suffix=".jsonl", prefix="tsugite_compact_")
    with os.fdopen(fd, "w") as f:
        for turn in turns:
            for msg in turn.messages:
                f.write(json.dumps(msg) + "\n")
    return Path(path)


class HasUIHandler(Protocol):
    """Protocol for objects with a ui_handler attribute."""

    ui_handler: Any


class ThreadCapability(Protocol):
    """Optional protocol for adapters that support platform threads."""

    async def create_thread(self, channel_id: str, title: str) -> str:
        """Create a platform thread in the given channel. Returns platform_thread_id."""
        ...

    async def send_to_thread(self, platform_thread_id: str, message: str) -> None:
        """Send a message to an existing thread."""
        ...

    async def close_thread(self, platform_thread_id: str) -> None:
        """Archive/close a thread."""
        ...


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
    thread_id: Optional[str] = None

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
        if self.thread_id:
            result["thread_id"] = self.thread_id
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
        session_store: SessionStore,
        identity_map: Optional[Dict[str, str]] = None,
    ):
        self.agent_name = agent_name
        self.agent_config = agent_config
        self.session_store = session_store
        self._identity_map = identity_map or {}
        self.event_bus = None  # Set by HTTPServer for global SSE broadcast

        from tsugite.workspace import Workspace, WorkspaceNotFoundError

        try:
            self._workspace = Workspace.load(agent_config.workspace_dir)
        except WorkspaceNotFoundError:
            self._workspace = None

        # Workspace attachments are built per-message via _get_workspace_attachments()
        # so that daily memory files (memory/YYYY-MM-DD.md) are picked up fresh.

    def _get_workspace_attachments(self):
        """Build workspace attachments fresh each call so new memory files are included."""
        if not self._workspace:
            return []
        from tsugite.workspace.context import build_workspace_attachments

        return build_workspace_attachments(self._workspace)

    def _get_all_attachments(self):
        """Build all attachments: workspace + agent config (for UI display)."""
        attachments = list(self._get_workspace_attachments())

        agent_path = self._resolve_agent_path()
        if agent_path:
            try:
                from tsugite.agent_preparation import resolve_agent_config_attachments
                from tsugite.md_agents import parse_agent_file

                agent = parse_agent_file(agent_path)
                workspace_path = self._workspace.path if self._workspace else None

                # Support "-filename" removal syntax
                removals = {t.lstrip("-") for t in (agent.config.attachments or []) if t.startswith("-")}
                keep_templates = [t for t in (agent.config.attachments or []) if not t.startswith("-")]
                if removals:
                    attachments = [a for a in attachments if a.name not in removals]
                attachments.extend(resolve_agent_config_attachments(keep_templates, workspace_path))
            except Exception as e:
                logger.debug("Failed to load agent config attachments: %s", e)

        # Deduplicate by name (keep first occurrence)
        seen: set[str] = set()
        deduped = []
        for att in attachments:
            if att.name not in seen:
                seen.add(att.name)
                deduped.append(att)

        return deduped

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

    def _save_history(
        self,
        *,
        agent_path,
        message,
        conv_id,
        metadata,
        result_str,
        token_count=None,
        cost=None,
        execution_steps=None,
        system_prompt=None,
        attachments=None,
        claude_code_session_id=None,
        claude_code_compacted=False,
        status="success",
        error_message=None,
    ):
        try:
            from tsugite.agent_runner.history_integration import save_run_to_history

            save_run_to_history(
                agent_path=agent_path,
                agent_name=self.agent_name,
                prompt=message,
                result=result_str,
                model=self.resolve_model(),
                token_count=token_count,
                cost=cost,
                execution_steps=execution_steps,
                continue_conversation_id=conv_id,
                channel_metadata=metadata,
                system_prompt=system_prompt,
                attachments=attachments,
                claude_code_session_id=claude_code_session_id,
                claude_code_compacted=claude_code_compacted,
                status=status,
                error_message=error_message,
            )
        except Exception as e:
            logger.warning("Failed to save daemon history: %s", e)

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

    def _broadcast_compaction(self, agent: str, *, started: bool) -> None:
        """Broadcast compaction state change to all SSE subscribers."""
        if self.event_bus:
            event_type = "compaction_started" if started else "compaction_finished"
            try:
                self.event_bus.emit(event_type, {"agent": agent})
            except Exception:
                logger.debug("Failed to broadcast %s", event_type)

    def _build_agent_context(self, channel_context: ChannelContext) -> Dict[str, Any]:
        """Build context dict for agent template rendering."""
        ctx: Dict[str, Any] = {"is_daemon": True, "is_scheduled": False, "schedule_id": "", "has_notify_tool": False}
        meta = channel_context.metadata or {}
        if channel_context.source == "scheduler":
            ctx["is_scheduled"] = True
            ctx["schedule_id"] = meta.get("schedule_id", "")
            ctx["has_notify_tool"] = meta.get("notify_tool", False)
        ctx["running_tasks"] = meta.get("running_tasks", [])

        # Session context
        ctx["is_session"] = channel_context.source == "session"
        ctx["session_id"] = meta.get("session_id", "") if ctx["is_session"] else ""

        # Single pass over sessions for orchestrator awareness
        window_minutes = meta.get("heartbeat_window", 10)
        now = datetime.now(timezone.utc)
        active_sessions = []
        recent_completions = []
        for s in self.session_store.list_sessions():
            if s.status == "running":
                active_sessions.append(
                    {
                        "id": s.id,
                        "agent": s.agent,
                        "status": s.status,
                        "prompt": (s.prompt or "")[:100],
                        "source": s.source,
                    }
                )
            elif s.status in ("completed", "failed") and _is_recent(s.last_active, minutes=window_minutes, now=now):
                recent_completions.append(
                    {"id": s.id, "agent": s.agent, "status": s.status, "result": (s.result or "")[:200]}
                )
        ctx["active_sessions"] = active_sessions
        ctx["recent_completions"] = recent_completions

        return ctx

    def _build_message_context(self, message: str, channel_context: ChannelContext, user_id: str) -> str:
        """Prepend per-message dynamic context to the user prompt.

        Keeps dynamic metadata in the user message turn (not the cached
        attachment context turn) for better cache efficiency.
        """
        tz_name = self.agent_config.timezone
        try:
            if tz_name:
                tz = ZoneInfo(tz_name)
            else:
                tz = get_localzone()
            now = datetime.now(tz)
            tz_label = tz_name or str(tz)
            timestamp = now.strftime("%Y-%m-%d %H:%M:%S ") + tz_label
        except Exception:
            timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        try:
            session = self.session_store.get_or_create_interactive(user_id, self.agent_name)
            tokens_used = session.cumulative_tokens
        except Exception:
            tokens_used = 0

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

        conv_id_override = (channel_context.metadata or {}).get("conv_id_override")
        if conv_id_override:
            conv_id = conv_id_override
        else:
            # Route: thread_id lookup → default interactive session
            thread_id = channel_context.thread_id
            thread_session = self.session_store.find_by_thread(thread_id) if thread_id else None
            if thread_session:
                conv_id = thread_session.id
            else:
                conv_id = self.session_store.get_or_create_interactive(user_id, self.agent_name).id

            if self.session_store.needs_compaction(conv_id) or self.session_store.is_compacting(
                user_id, self.agent_name
            ):
                conv_id = await self._run_compaction(user_id, conv_id, custom_logger)

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

        agent_context = self._build_agent_context(channel_context)
        agent_context["raw_message"] = message

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
                attachments = list(self._get_workspace_attachments())
                if channel_context.metadata and channel_context.metadata.get("uploaded_attachments"):
                    attachments.extend(channel_context.metadata.pop("uploaded_attachments"))

                return run_agent(
                    agent_path=agent_path,
                    prompt=enriched_prompt,
                    continue_conversation_id=conv_id,
                    attachments=attachments,
                    exec_options=ExecutionOptions(
                        return_token_usage=True,
                        model_override=(channel_context.metadata or {}).get("model_override")
                        or self.agent_config.model,
                        max_turns_override=(channel_context.metadata or {}).get("max_turns_override")
                        or self.agent_config.max_turns,
                    ),
                    path_context=path_context,
                    custom_logger=custom_logger,
                    context=agent_context,
                )
            finally:
                os.chdir(original_cwd)

        ctx = contextvars.copy_context()
        try:
            result = await asyncio.to_thread(ctx.run, run_in_workspace)
        except AgentExecutionError as e:
            if "prompt too long" in str(e).lower() and not conv_id_override:
                logger.warning("[%s] Prompt too long, auto-compacting and retrying", self.agent_name)
                conv_id = await self._run_compaction(user_id, conv_id, custom_logger)
                ctx = contextvars.copy_context()
                result = await asyncio.to_thread(ctx.run, run_in_workspace)
            else:
                error_result = f"[Error: {e}]\n\n{e.partial_output}" if e.partial_output else f"[Error: {e}]"
                self._save_history(
                    agent_path=agent_path,
                    message=message,
                    conv_id=conv_id,
                    metadata=metadata,
                    result_str=error_result,
                    token_count=e.token_usage,
                    cost=e.cost,
                    execution_steps=e.execution_steps,
                )
                raise

        self._save_history(
            agent_path=agent_path,
            message=message,
            conv_id=conv_id,
            metadata=metadata,
            result_str=str(result),
            token_count=getattr(result, "token_count", None),
            cost=getattr(result, "cost", None),
            execution_steps=getattr(result, "execution_steps", None),
            system_prompt=getattr(result, "system_message", None),
            attachments=getattr(result, "attachments", None),
            claude_code_session_id=getattr(result, "claude_code_session_id", None),
            claude_code_compacted=getattr(result, "claude_code_compacted", False),
        )

        if result.context_window:
            self.session_store.update_context_limit(self.agent_name, result.context_window)
            self.agent_config.context_limit = result.context_window

        self.session_store.update_token_count(conv_id, result.token_count or 0)

        try:
            session = self.session_store.get_session(conv_id)
            if session and session.message_count <= 1 and not session.title:
                asyncio.ensure_future(self._auto_title_session(conv_id, message, str(result)))
        except Exception as e:
            logger.debug("Auto-title check failed for session '%s': %s", conv_id, e)

        return str(result)

    async def _auto_title_session(self, session_id: str, user_message: str, assistant_response: str) -> None:
        try:
            from tsugite.daemon.memory import compute_session_title

            title = await compute_session_title(user_message, assistant_response, self.resolve_model())
            if title:
                self.session_store.update_session(session_id, title=title)
                if self.event_bus:
                    self.event_bus.emit("session_update", {"action": "titled", "id": session_id, "title": title})
        except Exception as e:
            logger.debug("Auto-title failed for session '%s': %s", session_id, e)

    _DEFAULT_COMPACT_INSTRUCTIONS = (
        "Pay special attention to the last 5-10 messages. "
        "They contain the user's most recent active context. "
        "Preserve their details precisely in the summary."
    )

    async def _run_compaction(self, user_id: str, conv_id: str, custom_logger: Optional[HasUIHandler] = None) -> str:
        """Run session compaction and return the new conv_id."""
        self._emit_ui(custom_logger, "compacting")
        if self.session_store.begin_compaction(user_id, self.agent_name):
            self._broadcast_compaction(self.agent_name, started=True)
            try:
                await self._compact_session(conv_id)
            finally:
                self.session_store.end_compaction(user_id, self.agent_name)
                self._broadcast_compaction(self.agent_name, started=False)
        else:
            done = await asyncio.to_thread(self.session_store.wait_for_compaction, user_id, self.agent_name)
            if not done:
                raise TimeoutError("Timed out waiting for session compaction to finish")
        self._emit_ui(custom_logger, "compacted")
        session = self.session_store.get_or_create_interactive(user_id, self.agent_name)
        if self.event_bus:
            self.event_bus.emit("session_update", {"action": "compacted", "id": session.id})
        return session.id

    async def _compact_session(self, session_id: str, instructions: str | None = None) -> None:
        """Compact session when approaching context limit.

        Uses a sliding window: recent turns are kept verbatim while older
        turns are summarized. This preserves the active conversational thread.
        Fires pre_compact/post_compact hooks if configured.
        """
        if instructions is None:
            instructions = self._DEFAULT_COMPACT_INSTRUCTIONS
        from tsugite.daemon.memory import (
            RETENTION_BUDGET_RATIO,
            extract_file_paths_from_turns,
            get_context_limit,
            infer_compaction_model,
            split_turns_for_compaction,
            summarize_session,
        )
        from tsugite.history import SessionStorage, Turn, get_history_dir
        from tsugite.history.models import CompactionSummary
        from tsugite.hooks import fire_compact_hooks

        model = self.agent_config.compaction_model or infer_compaction_model(self.resolve_model())

        old_conv_id = session_id
        old_session_path = get_history_dir() / f"{old_conv_id}.jsonl"

        storage = SessionStorage(old_session_path)
        records = storage.load_records()
        all_turns = [r for r in records if isinstance(r, Turn)]
        prior_summary = next((r for r in records if isinstance(r, CompactionSummary)), None)

        context_limit = get_context_limit(model, fallback=self.agent_config.context_limit)
        retention_budget = int(context_limit * RETENTION_BUDGET_RATIO)

        old_turns, recent_turns = split_turns_for_compaction(all_turns, model, retention_budget)

        if not old_turns:
            logger.info("[%s] All turns fit in retention budget, skipping compaction", self.agent_name)
            return

        logger.info(
            "[%s] Compacting session: %d old turns summarized, %d recent turns retained",
            self.agent_name,
            len(old_turns),
            len(recent_turns),
        )

        old_session = self.session_store.get_session(session_id)

        turns_file = _write_turns_tempfile(old_turns)
        try:
            hook_context = {
                "conversation_id": old_conv_id,
                "user_id": old_session.user_id or "",
                "agent_name": self.agent_name,
                "turns_file": str(turns_file),
                "turn_count": len(old_turns),
            }
            pre_compact_execs = await fire_compact_hooks(
                self.agent_config.workspace_dir, "pre_compact", hook_context, interactive=False
            )
            storage.record_hook_executions(pre_compact_execs)

            old_messages = []

            if prior_summary:
                old_messages.append(
                    {
                        "role": "user",
                        "content": f"<prior_compaction_summary>\n{prior_summary.summary}\n</prior_compaction_summary>",
                    }
                )

            functions_used = sorted({fn for turn in old_turns for fn in turn.functions_called})
            file_paths = extract_file_paths_from_turns(old_turns)
            meta_parts = [
                "<session_metadata>",
                f"  <turn_count>{len(old_turns)}</turn_count>",
                f"  <time_range>{old_turns[0].timestamp.isoformat()} to {old_turns[-1].timestamp.isoformat()}</time_range>",
            ]
            if functions_used:
                meta_parts.append(f"  <tools_used>{', '.join(functions_used)}</tools_used>")
            meta_parts.append(f"  <model>{self.resolve_model()}</model>")
            if file_paths:
                meta_parts.append(f"  <files_accessed>{', '.join(file_paths)}</files_accessed>")
            meta_parts.append("</session_metadata>")
            old_messages.append({"role": "user", "content": "\n".join(meta_parts)})

            old_messages.extend(msg for turn in old_turns for msg in turn.messages)

            if instructions:
                old_messages.append({"role": "user", "content": f"<compaction_instructions>{instructions}</compaction_instructions>"})

            try:
                summary = await summarize_session(
                    old_messages, model=model, max_context_tokens=self.agent_config.context_limit
                )
            except Exception:
                logger.exception("[%s] Compaction summarization failed", self.agent_name)
                raise

            new_session = self.session_store.compact_session(session_id)
            new_session_path = get_history_dir() / f"{new_session.id}.jsonl"
            new_storage = SessionStorage.create(
                agent_name=self.agent_name,
                model=self.agent_config.agent_file,
                compacted_from=old_conv_id,
                session_path=new_session_path,
            )

            new_storage.record_compaction_summary(summary, len(old_turns), retained_turns=len(recent_turns))
            new_storage.write_turns(recent_turns)

            post_compact_execs = await fire_compact_hooks(
                self.agent_config.workspace_dir,
                "post_compact",
                {
                    **hook_context,
                    "new_conversation_id": new_session.id,
                    "turns_compacted": len(old_turns),
                    "turns_retained": len(recent_turns),
                },
                interactive=False,
            )
            new_storage.record_hook_executions(post_compact_execs)
        finally:
            turns_file.unlink(missing_ok=True)

        from tsugite.tools.skills import clear_loaded_skills

        clear_loaded_skills()

        logger.info("[%s] Session compacted", self.agent_name)
