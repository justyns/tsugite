"""Session runner - executes agent sessions in the background."""

import asyncio
import contextvars
import logging
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Coroutine, Optional
from uuid import uuid4
from xml.sax.saxutils import quoteattr

from tsugite.tools.notify import send_notification_nowait
from tsugite.ui.jsonl import JSONLUIHandler
from tsugite_daemon.adapters.base import ChannelContext
from tsugite_daemon.attention_store import OWNER_SESSION, SOURCE_ERROR
from tsugite_daemon.session_store import (
    FINISHED_STATUSES,
    Session,
    SessionSource,
    SessionStatus,
    SessionStore,
    attention_fields,
)

logger = logging.getLogger(__name__)

DELIVERY_KIND_FYI = "fyi"
DELIVERY_KIND_NEEDS_ACK = "needs_ack"
DELIVERY_KINDS = (DELIVERY_KIND_FYI, DELIVERY_KIND_NEEDS_ACK)

# Cuts a cycle of sessions notifying each other.
MAX_CHAIN_DEPTH = 5

_current_session_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("current_session_id", default=None)
_current_chain_depth: contextvars.ContextVar[int] = contextvars.ContextVar("chain_depth", default=0)


def get_current_session_id() -> Optional[str]:
    return _current_session_id.get()


def get_current_chain_depth() -> int:
    return _current_chain_depth.get()


def set_current_session_id(session_id: str) -> None:
    _current_session_id.set(session_id)


def set_current_chain_depth(depth: int) -> None:
    _current_chain_depth.set(depth)


@contextmanager
def chain_depth_scope(depth: int):
    """Run a chained reply at `depth`, restoring what the caller had."""
    previous = _current_chain_depth.get()
    _current_chain_depth.set(depth)
    try:
        yield
    finally:
        _current_chain_depth.set(previous)


# Transient events that should reach live subscribers but not the JSONL event log.
# Persisting them would bloat history with high-frequency heartbeats whose only
# value is real-time UI feedback.
_TRANSIENT_EVENT_TYPES = frozenset({"llm_wait_progress"})


class LoggingProgressHandler(JSONLUIHandler):
    """Wraps SSE event emission to also append events to the session JSONL log and broadcast via SSE."""

    def __init__(self, store: SessionStore, session_id: str, broadcaster=None):
        self._store = store
        self._session_id = session_id
        self._broadcaster = broadcaster

    def _emit(self, event_type: str, data: dict[str, Any]) -> None:
        event = {"type": event_type, "timestamp": datetime.now(timezone.utc).isoformat(), **data}
        if event_type not in _TRANSIENT_EVENT_TYPES:
            self._store.append_event(self._session_id, event)
        if self._broadcaster:
            self._broadcaster.emit(
                "session_event",
                {"session_id": self._session_id, "event_type": event_type, **data},
            )


_COMPLETION_SOURCES = {
    SessionStatus.COMPLETED.value: "session_completion",
    SessionStatus.CANCELLED.value: "session_cancelled",
    SessionStatus.FAILED.value: "session_failed",
}


def build_completion_message(session: Session, status: str, summary: str) -> str:
    from tsugite.prompt_xml import El

    body = ""
    if summary:
        tag = "error" if status == SessionStatus.FAILED.value else "result"
        body = El(tag, [summary]).render() + "\n"
    return (
        f"<session_finished id={quoteattr(session.id)} status={quoteattr(status)}"
        f" title={quoteattr(session.title or '')}>\n{body}</session_finished>"
    )


def report_send_failure(store: SessionStore, event_bus, session_id: str, *, ref_id: str, error: str) -> None:
    """Record a send that failed: an error block in the chat, and an attention
    record the session list can badge.

    A module function because BaseAdapter reports without holding a SessionRunner.
    """
    store.append_event(
        session_id,
        {"type": "error", "timestamp": datetime.now(timezone.utc).isoformat(), "error": error},
    )
    if event_bus:
        event_bus.emit("session_event", {"session_id": session_id, "event_type": "error", "error": error})
    opened = store.attention.open(
        owner_kind=OWNER_SESSION,
        owner_id=session_id,
        source=SOURCE_ERROR,
        ref_id=ref_id,
        kind="send_failed",
    )
    if opened and event_bus:
        records = store.attention.open_records(session_id)
        event_bus.emit("session_update", {"action": "attention", "id": session_id, **attention_fields(records)})


NotifyCallback = Callable[[Session, str], Coroutine[Any, Any, None]]


class SessionRunner:
    """Manages async agent session execution."""

    def __init__(
        self,
        store: SessionStore,
        adapter,
        notify_callback: Optional[NotifyCallback] = None,
        event_bus=None,
        notification_channels: Optional[dict] = None,
    ):
        self._store = store
        self._adapter = adapter
        self._event_bus = event_bus
        self._notification_channels = notification_channels or {}
        store.set_turn_end_listener(self._on_turn_end)
        self._active_tasks: dict[str, asyncio.Task] = {}
        self._completion_listeners: list[NotifyCallback] = []
        if notify_callback:
            self._completion_listeners.append(notify_callback)

    def flush_held_deliveries(self) -> None:
        """Deliver cards held by a turn that died with the last daemon; that turn
        never ends, so nothing else will flush them.

        Call it once the notifier is wired, or the fanout is dropped.
        """
        for session_id in self._store.sessions_holding_deliveries():
            self._flush_deferred_deliveries(session_id)

    def add_completion_listener(self, callback: NotifyCallback) -> None:
        """Register a session-completion listener. Idempotent."""
        if callback not in self._completion_listeners:
            self._completion_listeners.append(callback)

    async def _dispatch_completion(self, session: Session, result_str: str) -> None:
        """Fan completion out to every listener; one listener's failure must not
        starve the others."""
        for callback in list(self._completion_listeners):
            try:
                await callback(session, result_str)
            except Exception as e:
                logger.error("Session '%s' completion listener failed: %s", session.id, e)
                self._report_send_failure(
                    session.id,
                    ref_id=f"{session.id}:completion",
                    error=f"Completion listener failed: {e}",
                )

    @property
    def store(self) -> SessionStore:
        return self._store

    @property
    def runtime(self):
        """The adapter's runtime defaults, or None when no adapter is wired."""
        return getattr(self._adapter, "runtime", None)

    def is_session_running(self, session_id: str) -> bool:
        task = self._active_tasks.get(session_id)
        return task is not None and not task.done()

    def start_session(self, session: Session) -> Session:
        session.status = SessionStatus.RUNNING.value
        if not session.source:
            session.source = SessionSource.BACKGROUND.value
        # Background runs stay open to `session_reply` after they finish; job
        # workers and scheduled runs are one-shot.
        session.resumable = session.source == SessionSource.BACKGROUND.value
        session = self._store.create_session(session)

        if self._event_bus:
            self._event_bus.emit("session_update", {"action": "created", "id": session.id})

        progress = LoggingProgressHandler(self._store, session.id, broadcaster=self._event_bus)
        progress._emit("session_start", {"prompt": session.prompt[:200]})

        loop = asyncio.get_running_loop()
        task = loop.create_task(self._run_session(session, progress))
        self._active_tasks[session.id] = task
        task.add_done_callback(lambda t: self._active_tasks.pop(session.id, None))

        return session

    async def _run_session(self, session: Session, progress: LoggingProgressHandler) -> None:
        adapter = self._adapter

        from tsugite.interaction import NonInteractiveBackend, set_interaction_backend

        custom_logger = SimpleNamespace(ui_handler=progress)

        metadata = {
            "session_id": session.id,
            "conv_id_override": session.id,
            "model_override": session.model,
        }
        if session.agent_file:
            metadata["agent_file_override"] = str(adapter._resolve_agent_path(session.agent_file) or session.agent_file)
        if session.workspace_override:
            metadata["workspace_override"] = session.workspace_override
        # Carry an inherited sandbox policy (stamped by a sandboxed spawner) into
        # the chokepoint so the spawned run stays sandboxed.
        if session.metadata and session.metadata.get("sandbox_override"):
            metadata["sandbox_override"] = session.metadata["sandbox_override"]

        # Delegated files (validated at spawn time) become first-turn attachments
        # here, where the target model is known for the vision gate; non-inlinable
        # ones degrade to a path hint on the message.
        message = session.prompt
        delegation_files = (session.metadata or {}).get("delegation_files")
        if delegation_files:
            from tsugite.attachments.delegation import (
                format_delegation_hint,
                materialize_delegation_attachments,
                partition_delegation_files,
            )
            from tsugite.models import model_supports_vision

            effective_model = session.model or adapter.runtime.model
            supports_vision = model_supports_vision(effective_model) if effective_model else True
            inline_files, hint_files = partition_delegation_files([Path(p) for p in delegation_files], supports_vision)
            uploaded = materialize_delegation_attachments(inline_files)
            if uploaded:
                metadata["uploaded_attachments"] = uploaded
            message += format_delegation_hint(hint_files)

        channel_context = ChannelContext(
            source="session",
            channel_id=None,
            user_id=f"session:{session.id}",
            reply_to=f"session:{session.id}",
            metadata=metadata,
        )

        set_current_session_id(session.id)
        set_interaction_backend(NonInteractiveBackend())

        try:
            result = await adapter.handle_message(
                user_id=f"session:{session.id}",
                message=message,
                channel_context=channel_context,
                custom_logger=custom_logger,
            )
            result_str = str(result)
            updated = self._store.update_session(
                session.id,
                status=SessionStatus.COMPLETED.value,
                result=result_str,
            )
            if not session.title:
                asyncio.create_task(self._auto_title_background_session(session, result_str, adapter))
            progress._emit("session_complete", {"result_preview": result_str[:500]})
            if self._event_bus:
                self._event_bus.emit("session_update", {"action": "completed", "id": session.id})
                self._event_bus.emit("agent_status", {})
            logger.info("Session '%s' completed", session.id)

            await self._dispatch_completion(updated, result_str)

            await self._notify_finished(session, SessionStatus.COMPLETED.value, result_str[:500])

        except asyncio.CancelledError:
            updated = self._store.update_session(session.id, status=SessionStatus.CANCELLED.value)
            progress._emit("session_cancelled", {})
            if self._event_bus:
                self._event_bus.emit("session_update", {"action": "cancelled", "id": session.id})
            logger.info("Session '%s' cancelled", session.id)
            await self._dispatch_completion(updated, "CANCELLED")
            await self._notify_finished(session, SessionStatus.CANCELLED.value, "")
        except Exception as e:
            updated = self._store.update_session(session.id, status=SessionStatus.FAILED.value, error=str(e))
            progress._emit("session_error", {"error": str(e)})
            if self._event_bus:
                self._event_bus.emit("session_update", {"action": "failed", "id": session.id})
            logger.error("Session '%s' failed: %s", session.id, e)
            await self._dispatch_completion(updated, f"FAILED: {str(e)[:500]}")
            await self._notify_finished(session, SessionStatus.FAILED.value, str(e)[:500])

    async def _notify_finished(self, session: Session, status: str, summary: str) -> None:
        listed = ([session.parent_id] if session.parent_id else []) + session.notify_sessions
        targets = list(dict.fromkeys(listed))
        if not targets:
            return

        depth = get_current_chain_depth()
        if depth >= MAX_CHAIN_DEPTH:
            logger.warning(
                "Session '%s' finished at chain depth %d (max %d); not notifying %s",
                session.id,
                depth,
                MAX_CHAIN_DEPTH,
                targets,
            )
            return

        message = build_completion_message(session, status, summary)
        source = _COMPLETION_SOURCES[status]
        with chain_depth_scope(depth + 1):
            for target_id in targets:
                await self._notify_one(session, target_id, message, source)

    async def _notify_one(self, session: Session, target_id: str, message: str, source: str) -> None:
        if target_id == session.id:
            logger.info("Session '%s' lists itself as a notify target; skipping", session.id)
            return
        target = self._store.resolve_live(target_id)
        if target is None:
            logger.info("Session '%s' cannot notify '%s': unknown or pruned", session.id, target_id)
            return
        try:
            await self.reply_to_session(target.id, message, source=source, metadata={"from_session": session.id})
        except Exception as e:
            logger.warning("Session '%s' failed to notify '%s': %s", session.id, target.id, e)
            self._report_send_failure(
                session.id,
                ref_id=f"{session.id}:notify:{target.id}",
                error=f"Failed to notify session '{target.id}': {e}",
            )

    def add_notify_session(self, session_id: str, target_id: str) -> Session:
        session = self._store.add_notify_session(session_id, target_id)
        if self._event_bus:
            self._event_bus.emit(
                "session_update",
                {"action": "notify_target_added", "id": session_id, "target": target_id},
            )
        return session

    def rename_session(self, session_id: str, title: str) -> Session:
        session = self._store.update_session(session_id, title=title)
        if self._event_bus:
            self._event_bus.emit("session_update", {"action": "titled", "id": session_id, "title": title})
        return session

    def set_pin(self, session_id: str, pinned: bool, position: Optional[int] = None) -> Session:
        session = self._store.set_pin(session_id, pinned, position=position)
        if self._event_bus:
            self._event_bus.emit(
                "session_update",
                {"action": "pinned" if session.pinned else "unpinned", "id": session_id},
            )
        return session

    def reorder_pins(self, ordered_ids: list[str]) -> list[Session]:
        ordered = self._store.reorder_pins(ordered_ids)
        if self._event_bus:
            self._event_bus.emit("session_update", {"action": "reordered", "ids": [s.id for s in ordered]})
        return ordered

    def set_primary_session(self, session_id: str) -> Session:
        session = self._store.set_primary_session(session_id)
        if self._event_bus:
            self._event_bus.emit("session_update", {"action": "primary_set", "id": session_id})
        return session

    def clear_primary_session(self, user_id: str) -> Optional[Session]:
        cleared = self._store.clear_primary_session(user_id)
        if self._event_bus:
            self._event_bus.emit(
                "session_update",
                {"action": "primary_cleared", "id": cleared.id if cleared else None},
            )
        return cleared

    def deliver_to_session(
        self,
        session_id: str,
        message: str,
        *,
        source: str,
        kind: str = DELIVERY_KIND_FYI,
        title: Optional[str] = None,
        metadata: Optional[dict] = None,
        notify_channels: Optional[list[tuple[str, object]]] = None,
    ) -> None:
        """Deliver a card into a session's history without starting a turn.

        Sync: call it from a worker thread (asyncio.to_thread) when the caller is
        on the event loop.
        """
        session_id = self.live_id(session_id)
        target = self._store.get_session(session_id)
        if target.status in FINISHED_STATUSES and not target.accepts_followup:
            logger.info("Not delivering to session '%s': already finished", session_id)
            return
        if kind not in DELIVERY_KINDS:
            raise ValueError(f"Invalid delivery kind '{kind}' (expected one of: {', '.join(sorted(DELIVERY_KINDS))})")

        event = {
            "type": "delivery",
            "delivery_id": f"dlv-{uuid4().hex[:8]}",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "message": message,
            "source": source,
            "kind": kind,
            "title": title,
            # Names only: the card outlives this call.
            "notify_channels": [name for name, _config in notify_channels or []],
            **(metadata or {}),
        }
        if self._store.hold_delivery(session_id, event):
            return
        self._flush_delivery(session_id, event)

    def live_id(self, session_id: str) -> str:
        live = self._store.resolve_live(session_id)
        return live.id if live else session_id

    def _on_turn_end(self, session_id: str) -> None:
        self._flush_deferred_deliveries(session_id)
        self._close_job_batch(session_id)

    @staticmethod
    def _close_job_batch(session_id: str) -> None:
        from tsugite.tools.jobs import get_jobs_orchestrator

        orchestrator = get_jobs_orchestrator()
        if orchestrator:
            orchestrator.close_batch_barrier(session_id)

    def _flush_deferred_deliveries(self, session_id: str) -> None:
        for event in self._store.take_deferred_deliveries(session_id):
            try:
                self._flush_delivery(session_id, event)
            except Exception:
                logger.exception("Delivery '%s' for session '%s' was dropped", event.get("delivery_id"), session_id)

    def _flush_delivery(self, session_id: str, event: dict) -> None:
        is_needs_ack = event["kind"] == DELIVERY_KIND_NEEDS_ACK
        session = self._store.record_delivery(session_id, event, needs_ack=is_needs_ack)
        self._emit_attention(session_id)
        if self._event_bus:
            payload = {k: v for k, v in event.items() if k not in ("type", "notify_channels")}
            self._event_bus.emit("session_event", {"session_id": session_id, "event_type": "delivery", **payload})
            self._event_bus.emit(
                "session_update",
                {
                    "action": "delivered",
                    "id": session_id,
                    "pending_deliveries": session.pending_delivery_ids,
                },
            )
        if is_needs_ack:
            self._notify_delivery(session_id, event)

    def _notify_delivery(self, session_id: str, event: dict) -> None:
        channels = [
            (name, config)
            for name in event.get("notify_channels") or []
            if (config := self._notification_channels.get(name))
        ]
        if not channels:
            return
        title = event.get("title")
        text = f"**{title}**\n\n{event['message']}" if title else event["message"]
        try:
            send_notification_nowait(text, channels, url=f"#chats?sessionId={session_id}")
        except Exception as e:
            logger.error("Delivery notification for session '%s' failed: %s", session_id, e)
            self._report_send_failure(
                session_id,
                ref_id=f"{session_id}:delivery-notify:{event['delivery_id']}",
                error=f"Delivery notification failed: {e}",
            )

    def clear_attention(self, session_id: str, delivery_id: Optional[str] = None) -> Session:
        session_id = self.live_id(session_id)
        session = self._store.clear_attention(session_id, delivery_id)
        self._emit_attention(session_id)
        if self._event_bus:
            self._event_bus.emit(
                "session_update",
                {
                    "action": "attention_cleared",
                    "id": session_id,
                    "pending_deliveries": session.pending_delivery_ids,
                },
            )
        return session

    def open_attention(self, session_id: str, *, source: str, ref_id: str, kind: str) -> None:
        """Open an attention record; a re-report of something already open announces nothing."""
        opened = self._store.attention.open(
            owner_kind=OWNER_SESSION,
            owner_id=session_id,
            source=source,
            ref_id=ref_id,
            kind=kind,
        )
        if opened:
            self._emit_attention(session_id)

    def clear_attention_ref(self, source: str, ref_id: str) -> None:
        for record in self._store.attention.clear_ref(source, ref_id):
            self._emit_attention(record.owner_id)

    def _emit_attention(self, session_id: str) -> None:
        if not self._event_bus:
            return
        records = self._store.attention.open_records(session_id)
        self._event_bus.emit("session_update", {"action": "attention", "id": session_id, **attention_fields(records)})

    def _report_send_failure(self, session_id: str, *, ref_id: str, error: str) -> None:
        report_send_failure(self._store, self._event_bus, session_id, ref_id=ref_id, error=error)

    def mark_viewed(self, session_id: str, ts: Optional[str] = None) -> Session:
        session = self._store.mark_viewed(session_id, ts=ts)
        if self._event_bus:
            self._event_bus.emit("session_update", {"action": "viewed", "id": session_id})
        return session

    def update_session_metadata(self, session_id: str, updates: dict) -> Session:
        session = self._store.set_metadata_bulk(session_id, updates)
        self._emit_metadata_event(session_id, session.metadata)
        return session

    def delete_session_metadata(self, session_id: str, key: str) -> Session:
        session = self._store.delete_metadata(session_id, key)
        self._emit_metadata_event(session_id, session.metadata)
        return session

    def _emit_metadata_event(self, session_id: str, metadata: dict) -> None:
        if self._event_bus:
            self._event_bus.emit(
                "session_update", {"action": "metadata_updated", "id": session_id, "metadata": metadata}
            )

    async def _auto_title_background_session(self, session: Session, result_str: str, adapter) -> None:
        try:
            from tsugite_daemon.memory import compute_session_title

            title = await compute_session_title(
                session.prompt or "", result_str, adapter.resolve_model(), adapter.runtime.compaction_model
            )
            if title:
                self.rename_session(session.id, title)
        except Exception as e:
            logger.debug("Auto-title failed for session '%s': %s", session.id, e)

    def cancel_session(self, session_id: str) -> None:
        task = self._active_tasks.get(session_id)
        if task and not task.done():
            task.cancel()
        self._store.update_session(session_id, status=SessionStatus.CANCELLED.value)

    async def reply_to_session(
        self,
        session_id: str,
        message: str,
        source: str = "session",
        metadata: dict | None = None,
        revive: bool = False,
    ) -> str | None:
        """Send a follow-up message to an existing session, running a turn in it.

        Returns the turn's reply, or None when no turn ran: a finished session
        takes one only when `revive` is set.
        """
        session_id = self.live_id(session_id)
        target = self._store.get_session(session_id)  # raises if the session is unknown
        if not revive and target.status in FINISHED_STATUSES and not target.accepts_followup:
            logger.info("Session '%s' takes no reply: already finished", session_id)
            return None

        adapter = self._adapter
        if not adapter:
            raise ValueError(f"No adapter available to run session '{session_id}'")

        meta = {"conv_id_override": session_id, "session_id": session_id}
        if metadata:
            meta.update(metadata)

        channel_context = ChannelContext(
            source=source,
            channel_id=None,
            user_id=f"session:{session_id}",
            reply_to=f"session:{session_id}",
            metadata=meta,
        )

        # Set the current session ContextVar so tools that fall back to
        # get_current_session_id() (e.g. session_metadata, return_value) resolve
        # correctly during the reply turn. Restored on exit so we don't bleed into
        # the caller's context (this runs in the caller's context, not a fresh task).
        token = _current_session_id.set(session_id)
        try:
            result = await adapter.handle_message(
                user_id=f"session:{session_id}",
                message=message,
                channel_context=channel_context,
            )
        finally:
            _current_session_id.reset(token)

        self._store.update_session(session_id)
        if self._event_bus:
            self._event_bus.emit("history_update", {"session_id": session_id})
        return result
