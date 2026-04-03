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


class LoggingProgressHandler:
    """Wraps SSE event emission to also append events to the session JSONL log."""

    def __init__(self, store: SessionStore, session_id: str):
        self._store = store
        self._session_id = session_id

    def handle_event(self, event) -> None:
        """Handle BaseEvent from EventBus — delegate to JSONLUIHandler's logic."""
        from tsugite.ui.jsonl import JSONLUIHandler

        # Reuse JSONLUIHandler's event->_emit dispatch
        JSONLUIHandler.handle_event(self, event)

    def _emit(self, event_type: str, data: dict[str, Any]) -> None:
        event = {"type": event_type, "timestamp": datetime.now(timezone.utc).isoformat(), **data}
        self._store.append_event(self._session_id, event)


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

        progress = LoggingProgressHandler(self._store, session.id)
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
            self._store.update_session(
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
                    await self._notify_callback(self._store.get_session(session.id), result_str)
                except Exception as e:
                    logger.error("Session '%s' notify callback failed: %s", session.id, e)

        except asyncio.CancelledError:
            self._store.update_session(session.id, status=SessionStatus.CANCELLED.value)
            progress._emit("session_cancelled", {})
            if self._event_bus:
                self._event_bus.emit("session_update", {"action": "cancelled", "id": session.id})
            logger.info("Session '%s' cancelled", session.id)
        except Exception as e:
            self._store.update_session(session.id, status=SessionStatus.FAILED.value, error=str(e))
            progress._emit("session_error", {"error": str(e)})
            if self._event_bus:
                self._event_bus.emit("session_update", {"action": "failed", "id": session.id})
            logger.error("Session '%s' failed: %s", session.id, e)

    def rename_session(self, session_id: str, title: str) -> Session:
        session = self._store.update_session(session_id, title=title)
        if self._event_bus:
            self._event_bus.emit("session_update", {"action": "titled", "id": session_id, "title": title})
        return session

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
        return result

    def get_active_sessions(self) -> list[Session]:
        return [s for s in self._store.list_sessions() if s.status == SessionStatus.RUNNING.value]
