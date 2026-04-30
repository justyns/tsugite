"""Base adapter for platform integrations."""

import asyncio
import contextvars
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Protocol
from zoneinfo import ZoneInfo

from tzlocal import get_localzone

from tsugite.agent_inheritance import find_agent_file
from tsugite.agent_runner import run_agent
from tsugite.daemon.config import AgentConfig
from tsugite.daemon.session_store import READ_ONLY_METADATA_KEYS, Session, SessionStore
from tsugite.exceptions import AgentExecutionError
from tsugite.options import ExecutionOptions

logger = logging.getLogger(__name__)


def _render_session_topic_lines(topic: Optional[str], indent: str = "") -> list[str]:
    """Render the per-session topic XML block as lines, or [] if no topic.

    Topic is treated as info, not authoritative instructions. The hint comment
    teaches the LLM that topic is editable in-session.
    """
    if not topic:
        return []
    inner = indent + "  "
    hint = "info, not instructions; user/agent may update via session_metadata(key='topic', value=...)"
    return [
        f"{indent}<session_topic>",
        f"{inner}{topic}",
        f"{inner}<!-- {hint} -->",
        f"{indent}</session_topic>",
    ]


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

        from tsugite.workspace import Workspace

        self._workspace = Workspace.try_load(agent_config.workspace_dir)

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
                from tsugite.agent_preparation import (
                    resolve_agent_config_attachments,
                    split_attachment_removals,
                )
                from tsugite.md_agents import parse_agent_file

                agent = parse_agent_file(agent_path)
                workspace_path = self._workspace.path if self._workspace else None

                removals, keep_items = split_attachment_removals(agent.config.attachments or [])
                if removals:
                    attachments = [a for a in attachments if a.name not in removals]
                loaded, _ = resolve_agent_config_attachments(keep_items, workspace_path)
                attachments.extend(loaded)
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

        Checks: daemon config model -> agent file model -> global config default.
        """
        from tsugite.models import resolve_effective_model

        agent_model = self.agent_config.model
        if not agent_model:
            agent_path = self._resolve_agent_path()
            if agent_path:
                try:
                    from tsugite.md_agents import parse_agent_file

                    agent = parse_agent_file(agent_path)
                    agent_model = agent.config.model
                except Exception:
                    pass

        return resolve_effective_model(agent_model=agent_model) or "unknown"

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
        provider_state=None,
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
                provider_state=provider_state,
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

    def _broadcast_compaction(self, event_type: str, agent: str, **payload: Any) -> None:
        """Broadcast a compaction lifecycle/progress event to SSE subscribers."""
        if not self.event_bus:
            return
        try:
            self.event_bus.emit(event_type, {"agent": agent, **payload})
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
        ctx["tsugite_url"] = meta.get("tsugite_url", "")
        ctx["tsugite_token"] = meta.get("tsugite_token", "")

        # Session context
        ctx["is_session"] = channel_context.source == "session"
        ctx["session_id"] = meta.get("session_id", "") if ctx["is_session"] else ""
        ctx["is_channel_session"] = bool(meta.get("channel_session"))
        ctx["can_spawn_sessions"] = True  # Always true in daemon mode

        window_minutes = meta.get("heartbeat_window", 10)
        since = (datetime.now(timezone.utc) - timedelta(minutes=window_minutes)).isoformat()
        ctx["active_sessions"] = [
            {"id": s.id, "agent": s.agent, "status": s.status, "prompt": (s.prompt or "")[:100], "source": s.source}
            for s in self.session_store.list_sessions(status="running")
        ]
        ctx["recent_completions"] = [
            {"id": s.id, "agent": s.agent, "status": s.status, "result": (s.result or "")[:200]}
            for s in self.session_store.list_sessions(status="completed", updated_since=since)
        ]

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
        session_topic_xml = ""
        session_meta_xml = ""
        scratchpad_xml = ""
        try:
            conv_id_override = (channel_context.metadata or {}).get("conv_id_override")
            if conv_id_override:
                session = self.session_store.get_session(conv_id_override)
            else:
                session = self.session_store.get_or_create_interactive(user_id, self.agent_name)
            tokens_used = session.cumulative_tokens
            if session.metadata:
                topic_lines = _render_session_topic_lines(session.metadata.get("topic"), indent="  ")
                if topic_lines:
                    session_topic_xml = "\n" + "\n".join(topic_lines)
                user_meta = {
                    k: v
                    for k, v in session.metadata.items()
                    if k not in READ_ONLY_METADATA_KEYS and k != "topic"
                }
                if user_meta:
                    entries = "\n".join(f"    {k}={v}" for k, v in user_meta.items())
                    session_meta_xml = f"\n  <session_metadata>\n{entries}\n  </session_metadata>"
            if session.scratchpad:
                scratchpad_xml = f"\n  <scratchpad>\n{session.scratchpad}\n  </scratchpad>"
        except Exception:
            tokens_used = 0

        return f"""<message_context>
  <datetime>{timestamp}</datetime>
  <working_directory>{self.agent_config.workspace_dir}</working_directory>
  <source>{channel_context.source}</source>
  <user_id>{channel_context.user_id}</user_id>
  <context_tokens_used>{tokens_used}</context_tokens_used>
  <context_limit>{self.agent_config.context_limit}</context_limit>{session_topic_xml}{session_meta_xml}{scratchpad_xml}
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
                conv_id = await self._run_compaction(user_id, conv_id, custom_logger, reason="token_threshold")

        from tsugite.daemon.session_runner import get_current_session_id, set_current_session_id

        if get_current_session_id() is None:
            set_current_session_id(conv_id)

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
        # Skip the sort+copy for the common case where nothing was suppressed.
        suppressed = self.session_store.get_suppressed_skills(conv_id)
        if suppressed:
            agent_context["suppressed_skills"] = sorted(suppressed)

        # Sticky skills carried over from prior turns drive TTL tracking.
        sticky_counters = self.session_store.get_sticky_skills(conv_id)
        if sticky_counters:
            agent_context["sticky_skills"] = sticky_counters
        from tsugite.config import load_config as _load_ttl_config

        agent_context["skill_ttl_default"] = _load_ttl_config().skill_ttl_default

        from tsugite.cli.helpers import PathContext, set_workspace_dir

        workspace_dir = self.agent_config.workspace_dir
        path_context = PathContext(
            invoked_from=workspace_dir,
            workspace_dir=workspace_dir,
            effective_cwd=workspace_dir,
        )

        def run_in_workspace():
            """Run agent with workspace bound via task-local ContextVar."""
            set_workspace_dir(workspace_dir)
            attachments = list(self._get_workspace_attachments())
            if channel_context.metadata and channel_context.metadata.get("uploaded_attachments"):
                attachments.extend(channel_context.metadata.pop("uploaded_attachments"))

            meta = channel_context.metadata or {}
            effort_override = meta.get("reasoning_effort_override") or self.session_store.get_reasoning_effort(conv_id)
            model_override = (
                meta.get("model_override") or self.session_store.get_model_override(conv_id) or self.agent_config.model
            )
            return run_agent(
                agent_path=agent_path,
                prompt=enriched_prompt,
                continue_conversation_id=conv_id,
                attachments=attachments,
                exec_options=ExecutionOptions(
                    return_token_usage=True,
                    model_override=model_override,
                    max_turns_override=meta.get("max_turns_override") or self.agent_config.max_turns,
                    reasoning_effort_override=effort_override,
                ),
                path_context=path_context,
                custom_logger=custom_logger,
                context=agent_context,
                user_input_for_history=message,
            )

        code_events_before = self.session_store.count_events_by_type(conv_id, "code_execution")
        ctx = contextvars.copy_context()
        try:
            result = await asyncio.to_thread(ctx.run, run_in_workspace)
        except AgentExecutionError as e:
            if "prompt too long" in str(e).lower() and not conv_id_override:
                code_events_after = self.session_store.count_events_by_type(conv_id, "code_execution")
                if code_events_after > code_events_before:
                    logger.warning(
                        "[%s] Prompt too long after %d code executions - not auto-retrying "
                        "to avoid re-issuing side effects",
                        self.agent_name,
                        code_events_after - code_events_before,
                    )
                    raise
                logger.warning("[%s] Prompt too long, auto-compacting and retrying", self.agent_name)
                conv_id = await self._run_compaction(user_id, conv_id, custom_logger, reason="prompt_too_long")
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
            provider_state=getattr(result, "provider_state", None),
        )

        # Sticky-skill TTL bookkeeping: update session-level counters after this turn.
        self._update_skill_ttl(conv_id, message, result, agent_context)

        ps = getattr(result, "provider_state", None) or {}
        if ps.get("context_window"):
            self.session_store.update_context_limit(self.agent_name, ps["context_window"])
            self.agent_config.context_limit = ps["context_window"]

        last_input = getattr(result, "last_input_tokens", None)
        context_tokens = last_input if isinstance(last_input, int) and last_input > 0 else (result.token_count or 0)
        self.session_store.update_token_count(conv_id, context_tokens)

        try:
            from tsugite.usage import get_usage_store

            get_usage_store().record(
                session_id=conv_id,
                agent=self.agent_name,
                model=self.resolve_model(),
                source=channel_context.source if channel_context else "daemon",
                total_tokens=result.token_count or 0,
                cost_usd=result.cost,
                duration_ms=getattr(result, "duration_ms", None),
                cache_creation_tokens=ps.get("cache_creation_tokens", 0),
                cache_read_tokens=ps.get("cache_read_tokens", 0),
            )
        except Exception as e:
            logger.debug("Failed to record usage: %s", e)

        try:
            session = self.session_store.get_session(conv_id)
            if session and session.message_count <= 1 and not session.title:
                asyncio.create_task(self._auto_title_session(conv_id, message, str(result)))
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

    async def _run_compaction(
        self, user_id: str, conv_id: str, custom_logger: Optional[HasUIHandler] = None, reason: str | None = None
    ) -> str:
        """Run session compaction and return the new conv_id.

        The new id comes from `_compact_session`'s return value (active branch)
        or `old.superseded_by` (waited on another thread). Both are direct
        consequences of the rotation that just happened; rediscovering via
        `get_or_create_interactive` would silently substitute the user's
        default-interactive session for non-default or non-interactive sources.
        """
        self._emit_ui(custom_logger, "compacting")
        new_session: Optional[Session] = None
        if self.session_store.begin_compaction(user_id, self.agent_name):
            self._broadcast_compaction("compaction_started", self.agent_name)

            def progress_cb(payload: Dict[str, Any]) -> None:
                self._broadcast_compaction("compaction_progress", self.agent_name, **payload)

            try:
                new_session = await self._compact_session(conv_id, reason=reason, progress_callback=progress_cb)
            finally:
                self.session_store.end_compaction(user_id, self.agent_name)
                self._broadcast_compaction("compaction_finished", self.agent_name)
        else:
            done = await asyncio.to_thread(self.session_store.wait_for_compaction, user_id, self.agent_name)
            if not done:
                raise TimeoutError("Timed out waiting for session compaction to finish")
            new_session = self.session_store.resolve_compacted_successor(conv_id)

        self._emit_ui(custom_logger, "compacted")
        new_id = new_session.id if new_session else conv_id
        if self.event_bus:
            self.event_bus.emit("session_update", {"action": "compacted", "id": new_id})
        return new_id

    def _update_skill_ttl(self, conv_id: str, user_message: str, result, agent_context: dict) -> None:
        """Advance per-session TTL counters based on what happened this turn.

        - trigger-matched skills become sticky (counter = 0)
        - explicit load_skill() calls reset their sticky counter (renewal)
        - unload_skill() calls drop from sticky entirely
        - skills referenced by name or trigger in user_message + final answer reset
        - all other sticky skills increment by 1
        - skills whose counter now exceeds their effective ttl are dropped and a
          SkillUnloadedEvent is emitted
        """
        try:
            from tsugite.config import load_config as _load_cfg
            from tsugite.events.events import SkillUnloadedEvent
            from tsugite.skill_discovery import find_referenced_skills, scan_skills

            ttl_default = _load_cfg().skill_ttl_default

            # Registry lookup for ttl values and trigger keywords used by the scan.
            registry = {s.name: s for s in scan_skills()}

            # Prune already-expired entries the preparer identified.
            for name in agent_context.get("_expired_sticky_skills") or []:
                self.session_store.drop_sticky(conv_id, name)

            auto_exempt = set(agent_context.get("_auto_loaded_skill_names") or [])

            # New trigger-matched skills become sticky (auto-loaded are exempt).
            for name in agent_context.get("_triggered_skill_names") or []:
                if name in auto_exempt:
                    continue
                self.session_store.mark_sticky(conv_id, name)

            execution_steps = getattr(result, "execution_steps", None) or []

            # Explicit load_skill() calls become sticky (renewal or fresh),
            # and count as references so the counter resets to 0.
            referenced: set[str] = set()
            for step in execution_steps:
                for name in (getattr(step, "loaded_skills", {}) or {}).keys():
                    referenced.add(name)
                    if name not in auto_exempt:
                        self.session_store.mark_sticky(conv_id, name)

            # Drop anything the agent called unload_skill() on — this wins over any
            # other sticky mutation from the same turn.
            for step in execution_steps:
                for name in getattr(step, "unloaded_skills", []) or []:
                    self.session_store.drop_sticky(conv_id, name)
                    referenced.discard(name)

            # Text-scan for skill names / triggers in user message + final answer.
            sticky_after_initial_updates = self.session_store.get_sticky_skills(conv_id)
            if sticky_after_initial_updates:
                sticky_metas = [registry[n] for n in sticky_after_initial_updates if n in registry]
                scan_text = f"{user_message}\n{str(result)}"
                referenced.update(find_referenced_skills(scan_text, sticky_metas))

            self.session_store.bump_unused_counters(conv_id, referenced)

            # Drop anything that exceeded its TTL and notify listeners.
            for name, counter in list(self.session_store.get_sticky_skills(conv_id).items()):
                meta = registry.get(name)
                effective_ttl = meta.ttl if (meta is not None and meta.ttl is not None) else ttl_default
                if effective_ttl > 0 and counter > effective_ttl:
                    self.session_store.drop_sticky(conv_id, name)
                    if self.event_bus:
                        try:
                            self.event_bus.emit(SkillUnloadedEvent(skill_name=name))
                        except Exception:
                            logger.debug("Failed to emit SkillUnloadedEvent for %s", name, exc_info=True)
        except Exception:
            logger.exception("Skill TTL bookkeeping failed for session %s", conv_id)

    async def _compact_session(
        self,
        session_id: str,
        instructions: str | None = None,
        reason: str | None = None,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Optional[Session]:
        """Compact a session by summarizing older events and rotating to a new
        session whose first body event is a `compaction` summary, followed by
        the retained recent events.

        Returns the new `Session` on success, or `None` when nothing was
        rotated (all events already fit in the retention budget). Callers must
        use the returned session for downstream id-keyed work; looking up the
        successor via `_interactive_index` is unreliable for non-default or
        non-interactive sessions.
        """
        if instructions is None:
            instructions = self._DEFAULT_COMPACT_INSTRUCTIONS
        from tsugite.daemon.memory import (
            RETENTION_BUDGET_RATIO,
            extract_file_paths_from_events,
            get_context_limit,
            infer_compaction_model,
            sanitize_for_summary,
            split_events_for_compaction,
            summarize_session,
        )
        from tsugite.history import SessionStorage, SessionSummary, events_to_messages, get_history_dir
        from tsugite.hooks import fire_compact_hooks

        resolved_model = self.resolve_model()
        model = self.agent_config.compaction_model or infer_compaction_model(resolved_model)

        old_conv_id = session_id
        old_session_path = get_history_dir() / f"{old_conv_id}.jsonl"
        storage = SessionStorage.load(old_session_path)
        all_events = storage.load_events()

        prior_summary = next(
            (e.data.get("summary") for e in reversed(all_events) if e.type == "compaction"),
            None,
        )

        context_limit = get_context_limit(model, fallback=self.agent_config.context_limit)
        retention_budget = int(context_limit * RETENTION_BUDGET_RATIO)

        old_events, recent_events = split_events_for_compaction(all_events, model, retention_budget)

        if not old_events:
            logger.info("[%s] All events fit in retention budget, skipping compaction", self.agent_name)
            return None

        old_user_inputs = sum(1 for e in old_events if e.type == "user_input")
        recent_user_inputs = sum(1 for e in recent_events if e.type == "user_input")

        logger.info(
            "[%s] Compacting session: %d old turns summarized, %d recent turns retained",
            self.agent_name,
            old_user_inputs,
            recent_user_inputs,
        )

        if progress_callback:
            try:
                progress_callback(
                    {"phase": "starting", "replaced_count": old_user_inputs, "retained_count": recent_user_inputs}
                )
            except Exception:
                logger.debug("compaction progress_callback raised", exc_info=True)

        old_session = self.session_store.get_session(session_id)

        hook_context = {
            "conversation_id": old_conv_id,
            "user_id": old_session.user_id or "",
            "agent_name": self.agent_name,
            "turn_count": old_user_inputs,
        }
        pre_compact_execs = await fire_compact_hooks(
            self.agent_config.workspace_dir, "pre_compact", hook_context, interactive=False
        )
        for ex in pre_compact_execs:
            storage.record("hook_execution", **ex.model_dump(exclude_none=True))

        old_messages: list[dict] = []
        if prior_summary:
            old_messages.append(
                {
                    "role": "user",
                    "content": f"<prior_compaction_summary>\n{prior_summary}\n</prior_compaction_summary>",
                }
            )

        from tsugite.workspace.models import WORKSPACE_FILES

        functions_used = sorted(SessionSummary.from_events(old_events).functions_called)
        scaffolding_basenames = {f.lower() for f in WORKSPACE_FILES}
        file_paths = [
            p
            for p in extract_file_paths_from_events(old_events)
            if p.rsplit("/", 1)[-1].lower() not in scaffolding_basenames
        ]
        first_user_event = next((e for e in old_events if e.type == "user_input"), None)
        last_response_event = next((e for e in reversed(old_events) if e.type == "model_response"), None)
        time_start = first_user_event.ts.isoformat() if first_user_event else ""
        time_end = (
            (last_response_event or first_user_event).ts.isoformat()
            if (last_response_event or first_user_event)
            else ""
        )

        meta_parts = [
            "<session_metadata>",
            f"  <turn_count>{old_user_inputs}</turn_count>",
            f"  <time_range>{time_start} to {time_end}</time_range>",
        ]
        if functions_used:
            meta_parts.append(f"  <tools_used>{', '.join(functions_used)}</tools_used>")
        meta_parts.append(f"  <model>{resolved_model}</model>")
        if file_paths:
            meta_parts.append(f"  <files_accessed>{', '.join(file_paths)}</files_accessed>")
        meta_parts.append("</session_metadata>")
        topic = (old_session.metadata or {}).get("topic")
        meta_parts.extend(_render_session_topic_lines(topic))
        old_messages.append({"role": "user", "content": "\n".join(meta_parts)})

        old_messages.extend(events_to_messages(old_events))

        if instructions:
            old_messages.append(
                {"role": "user", "content": f"<compaction_instructions>{instructions}</compaction_instructions>"}
            )

        attachment_basenames: set[str] = set(WORKSPACE_FILES)
        try:
            agent_path = self._resolve_agent_path()
            if agent_path:
                from tsugite.agent_preparation import split_attachment_removals
                from tsugite.md_agents import parse_agent_file

                attachments_spec = parse_agent_file(agent_path).config.attachments or []
                _, keep_items = split_attachment_removals(attachments_spec)
                for item in keep_items:
                    path = item if isinstance(item, str) else item.path
                    if path:
                        attachment_basenames.add(Path(path).name)
        except Exception:
            logger.debug("[%s] Failed to enumerate attachment basenames", self.agent_name, exc_info=True)

        old_messages = sanitize_for_summary(
            old_messages, model=model, attachment_basenames=attachment_basenames
        )

        # Snapshot the agent's tracked context limit before summarization so
        # that any mutation during the call (e.g. provider state leakage from
        # a smaller compact model) doesn't corrupt the displayed value or
        # the next compaction-threshold computation.
        saved_session_store_limit = self.session_store.get_context_limit(self.agent_name)
        saved_agent_config_limit = self.agent_config.context_limit
        try:
            summary = await summarize_session(
                old_messages,
                model=model,
                max_context_tokens=self.agent_config.context_limit,
                progress_callback=progress_callback,
            )
        except Exception:
            logger.exception("[%s] Compaction summarization failed", self.agent_name)
            raise
        finally:
            self.session_store.update_context_limit(self.agent_name, saved_session_store_limit)
            self.agent_config.context_limit = saved_agent_config_limit

        new_session = self.session_store.compact_session(session_id)
        new_session_path = get_history_dir() / f"{new_session.id}.jsonl"
        new_storage = SessionStorage.create(
            agent_name=self.agent_name,
            model=resolved_model,
            parent_session=old_conv_id,
            session_path=new_session_path,
        )

        new_storage.record(
            "compaction",
            summary=summary,
            replaced_count=old_user_inputs,
            retained_count=recent_user_inputs,
            reason=reason,
        )
        for event in recent_events:
            new_storage.record(event.type, **event.data)

        post_compact_execs = await fire_compact_hooks(
            self.agent_config.workspace_dir,
            "post_compact",
            {
                **hook_context,
                "new_conversation_id": new_session.id,
                "turns_compacted": old_user_inputs,
                "turns_retained": recent_user_inputs,
            },
            interactive=False,
        )
        for ex in post_compact_execs:
            new_storage.record("hook_execution", **ex.model_dump(exclude_none=True))

        from tsugite.tools.skills import clear_loaded_skills

        clear_loaded_skills()

        try:
            from tsugite.daemon.memory import _count_tokens, _message_text

            new_events = SessionStorage.load(new_session_path).load_events()
            new_messages = events_to_messages(new_events)
            text = "\n".join(_message_text(m) for m in new_messages)
            estimated = _count_tokens(text, resolved_model) if text else 0
            self.session_store.set_cumulative_tokens(new_session.id, estimated)
        except Exception:
            logger.debug("[%s] Failed to seed post-compaction token estimate", self.agent_name, exc_info=True)

        logger.info("[%s] Session compacted", self.agent_name)
        return new_session
