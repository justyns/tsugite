"""SSE broadcasting, streaming, and the per-chat progress/interaction backends."""

import asyncio
import json
import threading
from collections import deque
from typing import Any, Callable, Optional
from uuid import uuid4

from tsugite.events.base import BaseEvent
from tsugite.ui.jsonl import JSONLUIHandler
from tsugite_daemon.adapters.base import _PERSIST_EVENT_TYPES


class SSEBroadcaster:
    """Pub/sub for pushing real-time events to SSE subscribers.

    Every event gets a monotonic ``seq`` and lands in a bounded ring buffer, so
    a reconnecting client can replay what it missed (sleep/wake, network blip)
    instead of silently going stale. ``epoch`` identifies this daemon process:
    a client holding a different epoch reconnected across a restart and must
    fully resync rather than trust a delta.
    """

    REPLAY_BUFFER_SIZE = 512

    def __init__(self):
        self._subscribers: list[asyncio.Queue] = []
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._loop_thread_id: Optional[int] = None
        self._seq = 0
        self._buffer: deque[dict] = deque(maxlen=self.REPLAY_BUFFER_SIZE)
        self.epoch = uuid4().hex[:12]

    @property
    def seq(self) -> int:
        return self._seq

    def subscribe(self) -> asyncio.Queue:
        if self._loop is None:
            self._loop = asyncio.get_running_loop()
            self._loop_thread_id = threading.current_thread().ident
        q: asyncio.Queue = asyncio.Queue(maxsize=64)
        q.lagged = False
        self._subscribers.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue):
        try:
            self._subscribers.remove(q)
        except ValueError:
            pass

    def emit(self, event_type: str, data: dict | None = None):
        self._seq += 1
        msg = {"seq": self._seq, "type": event_type, "data": data or {}}
        self._buffer.append(msg)
        if not self._subscribers:
            return
        on_loop = threading.current_thread().ident == self._loop_thread_id
        for q in list(self._subscribers):
            try:
                if on_loop:
                    q.put_nowait(msg)
                elif self._loop:
                    self._loop.call_soon_threadsafe(self._put_or_lag, q, msg)
            except asyncio.QueueFull:
                # Don't silently drop: flag the slow subscriber so its stream
                # tells it to fully resync instead of losing events.
                q.lagged = True
            except RuntimeError:
                pass

    @staticmethod
    def _put_or_lag(q: asyncio.Queue, msg: dict) -> None:
        try:
            q.put_nowait(msg)
        except asyncio.QueueFull:
            q.lagged = True

    def replay_since(self, last_seq: int) -> Optional[list[dict]]:
        """Events after ``last_seq``, or None when the gap is unreplayable
        (older than the buffer) and the client must fully resync."""
        if last_seq >= self._seq:
            return []
        if not self._buffer or self._buffer[0]["seq"] > last_seq + 1:
            return None
        return [m for m in self._buffer if m["seq"] > last_seq]


async def sse_stream(queue: asyncio.Queue, keepalive_interval: float = 15.0):
    """Shared async generator for SSE streams with keepalive.

    A subscriber whose queue overflowed (flagged by the broadcaster) is told to
    fully resync instead of continuing with a silent gap in its event stream.
    """
    while True:
        if getattr(queue, "lagged", False):
            queue.lagged = False
            yield f"data: {json.dumps({'type': 'resync_required', 'data': {}})}\n\n"
        try:
            data = await asyncio.wait_for(queue.get(), timeout=keepalive_interval)
        except asyncio.TimeoutError:
            yield ": keepalive\n\n"
            continue
        if data is None:
            break
        yield f"data: {json.dumps(data)}\n\n"


# Events the per-chat SSE already delivers to the active client. Skipping them
# on the cross-session broadcaster prevents the active tab from rendering the
# same progress twice, and keeps the broadcast to what other surfaces (sidebar
# progress cache, non-active session detail view) actually read.
#
# Turn-end events (final_result, error, cancelled) are deliberately included
# here even though other tabs won't see them live: the same _emit() flow also
# fires history_update on the global event bus after the turn settles, which
# triggers loadHistory() in those tabs and rebuilds the message list from JSONL.
# Broadcasting turn-end events too would race the active tab's per-chat reader
# (the session's `sending` flag in sessionsState is cleared in streaming.js's
# finally block, leaving a window where late-arriving session_event(final_result)
# bypasses the dedup guard and pushes a duplicate bubble until the next reload).
_BROADCAST_SKIP_EVENTS = frozenset(
    {
        "stream_chunk",
        "stream_complete",
        "prompt_snapshot",
        "final_result",
        "error",
        "cancelled",
    }
)


class SSEProgressHandler(JSONLUIHandler):
    """Converts agent events to SSE messages via an async queue."""

    def __init__(self):
        self.queue: asyncio.Queue = asyncio.Queue()
        self.done = False
        self.has_final = False
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._persist_event: Optional[Callable] = None
        self._broadcaster: Optional["SSEBroadcaster"] = None
        self._session_id: Optional[str] = None

    def set_loop(self, loop: asyncio.AbstractEventLoop):
        self._loop = loop

    def set_event_persister(self, fn: Callable):
        """Set a callback to persist select events to the session event log."""
        self._persist_event = fn

    def set_broadcaster(self, broadcaster: "SSEBroadcaster") -> None:
        self._broadcaster = broadcaster

    def set_session_id(self, session_id: str) -> None:
        self._session_id = session_id

    latest_prompt_messages: Optional[list] = None

    def handle_event(self, event: BaseEvent) -> None:
        """Handle event from agent thread -- schedule onto the event loop."""
        from tsugite.events import PromptSnapshotEvent

        if isinstance(event, PromptSnapshotEvent):
            if event.messages:
                self.latest_prompt_messages = event.messages
            if not event.token_breakdown:
                return  # Messages-only update, don't emit SSE or persist
        super().handle_event(event)

    def _emit(self, event_type: str, data: dict[str, Any]) -> None:
        if event_type == "final_result":
            self.has_final = True
        payload = {"type": event_type, **data}
        if self._loop and self._loop.is_running():
            try:
                running_loop = asyncio.get_running_loop()
            except RuntimeError:
                running_loop = None
            if running_loop is self._loop:
                self.queue.put_nowait(payload)
            else:
                self._loop.call_soon_threadsafe(self.queue.put_nowait, payload)
        else:
            self.queue.put_nowait(payload)

        if event_type in _PERSIST_EVENT_TYPES and self._persist_event:
            self._persist_event(payload)

        if self._broadcaster and self._session_id and event_type not in _BROADCAST_SKIP_EVENTS:
            self._broadcaster.emit(
                "session_event",
                {"session_id": self._session_id, "event_type": event_type, **data},
            )

    def signal_done(self):
        """Set done and wake up the generator."""
        self.done = True
        self.queue.put_nowait(None)

    async def event_generator(self):
        while True:
            try:
                data = await asyncio.wait_for(self.queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                if self.done:
                    break
                yield ": keepalive\n\n"
                continue
            if data is None:
                break
            yield f"data: {json.dumps(data)}\n\n"
        # Yield once so any pending call_soon_threadsafe(put_nowait) callbacks
        # scheduled from worker threads get a chance to land in the queue before
        # we drain. Without this, events emitted just before signal_done can
        # race past the drain and be silently dropped.
        await asyncio.sleep(0)
        while not self.queue.empty():
            data = self.queue.get_nowait()
            if data is not None:
                yield f"data: {json.dumps(data)}\n\n"
        yield 'data: {"type": "done"}\n\n'


class HTTPInteractionBackend:
    """Interaction backend for HTTP -- emits SSE events, blocks until response."""

    TIMEOUT = 300  # 5 minutes

    def __init__(self, progress: SSEProgressHandler):
        self._progress = progress
        self._event = threading.Event()
        self._response: Optional[str] = None
        self.pending_message: Optional[str] = None

    def ask_user(self, question: str, question_type: str = "text", options: Optional[list[str]] = None) -> str:
        self._event.clear()
        self._response = None
        payload = {"question": question, "question_type": question_type}
        if options:
            payload["options"] = options
        self._progress._emit("ask_user", payload)

        if not self._event.wait(timeout=self.TIMEOUT):
            raise RuntimeError("Timed out waiting for user response (HTTP)")
        return self._response or ""

    def submit_response(self, response: str) -> None:
        self._response = response
        self._event.set()
