"""Session runner — executes agent sessions in the background."""

import asyncio
import contextvars
import logging
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, Callable, Coroutine, Optional

from tsugite.daemon.adapters.base import ChannelContext
from tsugite.daemon.session_store import (
    Session,
    SessionSource,
    SessionStatus,
    SessionStore,
)

logger = logging.getLogger(__name__)

# Context var for per-session state
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


# Transient events that should reach live subscribers but not the JSONL event log.
# Persisting them would bloat history with high-frequency heartbeats whose only
# value is real-time UI feedback.
_TRANSIENT_EVENT_TYPES = frozenset({"llm_wait_progress"})


class LoggingProgressHandler:
    """Wraps SSE event emission to also append events to the session JSONL log and broadcast via SSE."""

    def __init__(self, store: SessionStore, session_id: str, broadcaster=None):
        self._store = store
        self._session_id = session_id
        self._broadcaster = broadcaster

    def handle_event(self, event) -> None:
        """Handle BaseEvent from EventBus — delegate to JSONLUIHandler's logic."""
        from tsugite.ui.jsonl import JSONLUIHandler

        handler_name = JSONLUIHandler._DISPATCH.get(type(event))
        if handler_name:
            getattr(JSONLUIHandler, handler_name)(self, event)

    def _emit(self, event_type: str, data: dict[str, Any]) -> None:
        event = {"type": event_type, "timestamp": datetime.now(timezone.utc).isoformat(), **data}
        if event_type not in _TRANSIENT_EVENT_TYPES:
            self._store.append_event(self._session_id, event)
        if self._broadcaster:
            self._broadcaster.emit(
                "session_event",
                {"session_id": self._session_id, "event_type": event_type, **data},
            )


NotifyCallback = Callable[[Session, str], Coroutine[Any, Any, None]]


class SessionRunner:
    """Manages async agent session execution."""

    def __init__(
        self,
        store: SessionStore,
        adapters: dict,
        notify_callback: Optional[NotifyCallback] = None,
        event_bus=None,
    ):
        self._store = store
        self._adapters = adapters
        self._notify_callback = notify_callback
        self._event_bus = event_bus
        self._active_tasks: dict[str, asyncio.Task] = {}

    @property
    def store(self) -> SessionStore:
        return self._store

    def is_session_running(self, session_id: str) -> bool:
        task = self._active_tasks.get(session_id)
        return task is not None and not task.done()

    def start_session(self, session: Session) -> Session:
        session.status = SessionStatus.RUNNING.value
        if not session.source:
            session.source = SessionSource.BACKGROUND.value
        session = self._store.create_session(session)

        if self._event_bus:
            self._event_bus.emit("session_update", {"action": "created", "id": session.id})

        progress = LoggingProgressHandler(self._store, session.id, broadcaster=self._event_bus)
        progress._emit("session_start", {"agent": session.agent, "prompt": session.prompt[:200]})

        loop = asyncio.get_running_loop()
        task = loop.create_task(self._run_session(session, progress))
        self._active_tasks[session.id] = task
        task.add_done_callback(lambda t: self._active_tasks.pop(session.id, None))

        return session

    async def _run_session(self, session: Session, progress: LoggingProgressHandler) -> None:
        adapter = self._adapters.get(session.agent)
        if not adapter:
            self._store.update_session(
                session.id, status=SessionStatus.FAILED.value, error=f"No adapter for agent '{session.agent}'"
            )
            return

        from tsugite.interaction import NonInteractiveBackend, set_interaction_backend

        custom_logger = SimpleNamespace(ui_handler=progress)

        metadata = {
            "session_id": session.id,
            "conv_id_override": session.id,
            "model_override": session.model,
        }
        if session.agent_file:
            metadata["agent_file_override"] = str(adapter._resolve_agent_path(session.agent_file) or session.agent_file)

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
                message=session.prompt,
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
                self._event_bus.emit("agent_status", {"agent": session.agent})
            logger.info("Session '%s' completed", session.id)

            if self._notify_callback:
                try:
                    await self._notify_callback(updated, result_str)
                except Exception as e:
                    logger.error("Session '%s' notify callback failed: %s", session.id, e)

            if session.parent_id:
                try:
                    summary = f"Session '{session.title or session.id}' completed: {result_str[:500]}"
                    await self.reply_to_session(session.parent_id, summary, source="session_completion")
                except Exception as e:
                    logger.warning("Failed to notify parent session '%s': %s", session.parent_id, e)

        except asyncio.CancelledError:
            self._store.update_session(session.id, status=SessionStatus.CANCELLED.value)
            progress._emit("session_cancelled", {})
            if self._event_bus:
                self._event_bus.emit("session_update", {"action": "cancelled", "id": session.id})
            logger.info("Session '%s' cancelled", session.id)
            if session.parent_id:
                try:
                    await self.reply_to_session(
                        session.parent_id,
                        f"Session '{session.title or session.id}' was cancelled",
                        source="session_cancelled",
                    )
                except Exception as notify_err:
                    logger.warning(
                        "Failed to notify parent session '%s' of cancellation: %s", session.parent_id, notify_err
                    )
        except Exception as e:
            updated = self._store.update_session(session.id, status=SessionStatus.FAILED.value, error=str(e))
            progress._emit("session_error", {"error": str(e)})
            if self._event_bus:
                self._event_bus.emit("session_update", {"action": "failed", "id": session.id})
            logger.error("Session '%s' failed: %s", session.id, e)
            if self._notify_callback:
                try:
                    await self._notify_callback(updated, f"FAILED: {str(e)[:500]}")
                except Exception as notify_err:
                    logger.error("Session '%s' failure notify callback failed: %s", session.id, notify_err)
            if session.parent_id:
                try:
                    error_summary = f"Session '{session.title or session.id}' failed: {str(e)[:500]}"
                    await self.reply_to_session(session.parent_id, error_summary, source="session_failed")
                except Exception as notify_err:
                    logger.warning("Failed to notify parent session '%s' of failure: %s", session.parent_id, notify_err)
        finally:
            self._store.flush()

    def rename_session(self, session_id: str, title: str) -> Session:
        session = self._store.update_session(session_id, title=title)
        if self._event_bus:
            self._event_bus.emit("session_update", {"action": "titled", "id": session_id, "title": title})
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
            from tsugite.daemon.memory import compute_session_title

            title = await compute_session_title(session.prompt or "", result_str, adapter.resolve_model())
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
    ) -> str:
        """Send a follow-up message to an existing session."""
        session = self._store.get_session(session_id)

        adapter = self._adapters.get(session.agent)
        if not adapter:
            raise ValueError(f"No adapter for agent '{session.agent}' (session '{session_id}')")

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

        result = await adapter.handle_message(
            user_id=f"session:{session_id}",
            message=message,
            channel_context=channel_context,
        )

        self._store.update_session(session_id, last_active=datetime.now(timezone.utc).isoformat())
        if self._event_bus:
            self._event_bus.emit("history_update", {"agent": session.agent})
        return result

    def get_active_sessions(self) -> list[Session]:
        return [s for s in self._store.list_sessions() if s.status == SessionStatus.RUNNING.value]
