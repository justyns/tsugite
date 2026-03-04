"""Session runner — executes agent sessions with review gate support."""

import asyncio
import contextvars
import logging
import threading
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, Callable, Coroutine, Optional

from tsugite.daemon.agent_session import (
    AgentSession,
    AgentSessionStore,
    ReviewDecision,
    ReviewGate,
    SessionState,
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

    def __init__(self, store: AgentSessionStore, session_id: str):
        self._store = store
        self._session_id = session_id
        self._review_events: dict[str, threading.Event] = {}

    def ask_user(self, question: str, question_type: str = "text", options: Optional[list[str]] = None) -> str:
        return self._create_and_wait_for_review(question, question_type, options)

    def _create_and_wait_for_review(
        self, title: str, question_type: str = "text", options: Optional[list[str]] = None
    ) -> str:
        context = {}
        if question_type:
            context["question_type"] = question_type
        if options:
            context["options"] = options

        review = ReviewGate(
            id="",  # auto-generated
            session_id=self._session_id,
            title=title,
            context=context,
        )
        review = self._store.create_review(review)

        event = threading.Event()
        self._review_events[review.id] = event

        logger.info("Session '%s' waiting for review '%s': %s", self._session_id, review.id, title)

        if not event.wait(timeout=self.TIMEOUT):
            logger.warning("Review '%s' timed out after %ds", review.id, self.TIMEOUT)
            self._store.resolve_review(review.id, ReviewDecision.DECLINED.value, "Timed out")
            return "declined"

        resolved = self._store.get_review(review.id)
        self._review_events.pop(review.id, None)
        return resolved.decision


class LoggingProgressHandler:
    """Wraps SSE event emission to also append events to the session JSONL log."""

    def __init__(self, store: AgentSessionStore, session_id: str):
        self._store = store
        self._session_id = session_id

    def _emit(self, event_type: str, data: dict[str, Any]) -> None:
        event = {"type": event_type, "timestamp": datetime.now(timezone.utc).isoformat(), **data}
        self._store.append_event(self._session_id, event)


NotifyCallback = Callable[[AgentSession, str], Coroutine[Any, Any, None]]


class SessionRunner:
    """Manages async agent session execution with review gates."""

    def __init__(
        self,
        store: AgentSessionStore,
        adapters: dict,
        notify_callback: Optional[NotifyCallback] = None,
    ):
        self._store = store
        self._adapters = adapters
        self._notify_callback = notify_callback
        self._active_tasks: dict[str, asyncio.Task] = {}
        self._review_backends: dict[str, SessionReviewBackend] = {}

    @property
    def store(self) -> AgentSessionStore:
        return self._store

    def start_session(self, session: AgentSession) -> AgentSession:
        session = self._store.create_session(session)
        self._store.update_session(session.id, state=SessionState.RUNNING.value)
        self._store.append_event(
            session.id,
            {
                "type": "session_start",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "agent": session.agent,
                "prompt": session.prompt[:200],
            },
        )

        loop = asyncio.get_event_loop()
        task = loop.create_task(self._run_session(session))
        self._active_tasks[session.id] = task
        task.add_done_callback(lambda t: self._active_tasks.pop(session.id, None))

        return self._store.get_session(session.id)

    async def _run_session(self, session: AgentSession) -> None:
        adapter = self._adapters.get(session.agent)
        if not adapter:
            self._store.update_session(
                session.id, state=SessionState.FAILED.value, error=f"No adapter for agent '{session.agent}'"
            )
            return

        from tsugite.daemon.adapters.base import ChannelContext
        from tsugite.interaction import set_interaction_backend

        review_backend = SessionReviewBackend(self._store, session.id)
        self._review_backends[session.id] = review_backend

        progress = LoggingProgressHandler(self._store, session.id)
        custom_logger = SimpleNamespace(ui_handler=progress)

        channel_context = ChannelContext(
            source="session",
            channel_id=None,
            user_id=f"session:{session.id}",
            reply_to=f"session:{session.id}",
            metadata={
                "session_id": session.id,
                "model_override": session.model,
            },
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
                state=SessionState.COMPLETED.value,
                result=str(result),
                completed_at=datetime.now(timezone.utc).isoformat(),
                current_review_id=None,
            )
            self._store.append_event(
                session.id,
                {
                    "type": "session_complete",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "result_preview": str(result)[:500],
                },
            )
            logger.info("Session '%s' completed", session.id)

            if self._notify_callback:
                try:
                    await self._notify_callback(self._store.get_session(session.id), str(result))
                except Exception as e:
                    logger.error("Session '%s' notify callback failed: %s", session.id, e)

        except asyncio.CancelledError:
            self._store.update_session(session.id, state=SessionState.CANCELLED.value)
            self._store.append_event(
                session.id,
                {
                    "type": "session_cancelled",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )
            logger.info("Session '%s' cancelled", session.id)
        except Exception as e:
            self._store.update_session(session.id, state=SessionState.FAILED.value, error=str(e))
            self._store.append_event(
                session.id,
                {
                    "type": "session_error",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "error": str(e),
                },
            )
            logger.error("Session '%s' failed: %s", session.id, e)
        finally:
            self._review_backends.pop(session.id, None)

    def cancel_session(self, session_id: str) -> None:
        task = self._active_tasks.get(session_id)
        if task and not task.done():
            task.cancel()
        self._store.update_session(session_id, state=SessionState.CANCELLED.value)

    def resolve_review(self, review_id: str, decision: str, comment: str = "") -> ReviewGate:
        review = self._store.resolve_review(review_id, decision, comment)

        # Set session back to running
        session = self._store.get_session(review.session_id)
        if session.state == SessionState.WAITING_FOR_REVIEW.value:
            self._store.update_session(review.session_id, state=SessionState.RUNNING.value, current_review_id=None)

        # Unblock the review backend
        backend = self._review_backends.get(review.session_id)
        if backend:
            event = backend._review_events.get(review_id)
            if event:
                event.set()

        return review

    def get_active_sessions(self) -> list[AgentSession]:
        active_states = {SessionState.RUNNING.value, SessionState.WAITING_FOR_REVIEW.value}
        return [s for s in self._store.list_sessions() if s.state in active_states]
