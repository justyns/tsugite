"""Session runner — executes agent sessions with review gate support."""

import asyncio
import contextvars
import logging
import threading
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, Callable, Coroutine, Optional

from tsugite.daemon.session_store import (
    ReviewDecision,
    ReviewGate,
    Session,
    SessionSource,
    SessionStatus,
    SessionStore,
)

logger = logging.getLogger(__name__)

# Context var for per-session state
_current_session_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("current_session_id", default=None)


def get_current_session_id() -> Optional[str]:
    return _current_session_id.get()


class SessionReviewBackend:
    """InteractionBackend that creates review gates and blocks until resolved.

    Used inside session execution to intercept ask_user calls.
    """

    TIMEOUT = 3600  # 1 hour

    def __init__(self, store: SessionStore, session_id: str, on_review_created=None, event_bus=None):
        self._store = store
        self._session_id = session_id
        self._review_events: dict[str, threading.Event] = {}
        self._on_review_created = on_review_created
        self._event_bus = event_bus

    def ask_user(self, question: str, question_type: str = "text", options: Optional[list[str]] = None) -> str:
        context = {}
        if question_type:
            context["question_type"] = question_type
        if options:
            context["options"] = options
        review = self.create_and_wait(question, context=context)
        return review.decision

    def create_and_wait(self, title: str, description: str = "", context: Optional[dict] = None) -> ReviewGate:
        """Create a review gate and block until resolved. Returns the resolved ReviewGate."""
        review = ReviewGate(
            id="", session_id=self._session_id, title=title, description=description, context=context or {}
        )
        review = self._store.create_review(review)

        if self._event_bus:
            self._event_bus.emit(
                "review_update", {"action": "created", "id": review.id, "session_id": self._session_id}
            )

        if self._on_review_created:
            self._on_review_created(review)

        event = threading.Event()
        self._review_events[review.id] = event

        logger.info("Session '%s' waiting for review '%s': %s", self._session_id, review.id, title)

        if not event.wait(timeout=self.TIMEOUT):
            logger.warning("Review '%s' timed out after %ds", review.id, self.TIMEOUT)
            self._store.resolve_review(review.id, ReviewDecision.DECLINED.value, "Timed out")
            return self._store.get_review(review.id)

        resolved = self._store.get_review(review.id)
        self._review_events.pop(review.id, None)
        return resolved


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
ReviewNotifyCallback = Callable[[Session, ReviewGate], Coroutine[Any, Any, None]]


class SessionRunner:
    """Manages async agent session execution with review gates."""

    def __init__(
        self,
        store: SessionStore,
        adapters: dict,
        notify_callback: Optional[NotifyCallback] = None,
        review_notify_callback: Optional[ReviewNotifyCallback] = None,
        event_bus=None,
    ):
        self._store = store
        self._adapters = adapters
        self._notify_callback = notify_callback
        self._review_notify_callback = review_notify_callback
        self._event_bus = event_bus
        self._active_tasks: dict[str, asyncio.Task] = {}
        self._review_backends: dict[str, SessionReviewBackend] = {}

    @property
    def store(self) -> SessionStore:
        return self._store

    def start_session(self, session: Session) -> Session:
        session.status = SessionStatus.RUNNING.value
        if not session.source:
            session.source = SessionSource.BACKGROUND.value
        session = self._store.create_session(session)

        progress = LoggingProgressHandler(self._store, session.id)
        progress._emit("session_start", {"agent": session.agent, "prompt": session.prompt[:200]})

        loop = asyncio.get_event_loop()
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

        from tsugite.daemon.adapters.base import ChannelContext
        from tsugite.interaction import set_interaction_backend

        def _on_review_created(review: ReviewGate):
            if self._review_notify_callback:
                loop = asyncio.get_event_loop()
                loop.create_task(self._review_notify_callback(session, review))

        review_backend = SessionReviewBackend(
            self._store, session.id, on_review_created=_on_review_created, event_bus=self._event_bus
        )
        self._review_backends[session.id] = review_backend

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

        _current_session_id.set(session.id)
        set_interaction_backend(review_backend)

        try:
            result = await adapter.handle_message(
                user_id=f"session:{session.id}",
                message=session.prompt,
                channel_context=channel_context,
                custom_logger=custom_logger,
            )
            self._store.update_session(
                session.id,
                status=SessionStatus.COMPLETED.value,
                result=str(result),
                current_review_id=None,
            )
            progress._emit("session_complete", {"result_preview": str(result)[:500]})
            if self._event_bus:
                self._event_bus.emit("session_update", {"action": "completed", "id": session.id})
                self._event_bus.emit("agent_status", {"agent": session.agent})
            logger.info("Session '%s' completed", session.id)

            if self._notify_callback:
                try:
                    await self._notify_callback(self._store.get_session(session.id), str(result))
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
        finally:
            self._review_backends.pop(session.id, None)

    def cancel_session(self, session_id: str) -> None:
        task = self._active_tasks.get(session_id)
        if task and not task.done():
            task.cancel()
        self._store.update_session(session_id, status=SessionStatus.CANCELLED.value)

    def resolve_review(self, review_id: str, decision: str, comment: str = "") -> ReviewGate:
        review = self._store.resolve_review(review_id, decision, comment)

        # Set session back to running
        session = self._store.get_session(review.session_id)
        if session.status == SessionStatus.WAITING_FOR_REVIEW.value:
            self._store.update_session(review.session_id, status=SessionStatus.RUNNING.value, current_review_id=None)

        # Unblock the review backend
        backend = self._review_backends.get(review.session_id)
        if backend:
            event = backend._review_events.get(review_id)
            if event:
                event.set()

        return review

    def create_review_for_session(
        self, session_id: str, title: str, description: str = "", context: Optional[dict] = None
    ) -> ReviewGate:
        """Create a review gate within a session and block until resolved."""
        backend = self._review_backends.get(session_id)
        if not backend:
            raise RuntimeError(f"No review backend for session '{session_id}'")
        return backend.create_and_wait(title, description, context)

    def get_active_sessions(self) -> list[Session]:
        active_states = {SessionStatus.RUNNING.value, SessionStatus.WAITING_FOR_REVIEW.value}
        return [s for s in self._store.list_sessions() if s.status in active_states]
