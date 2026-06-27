"""Cooperative cancellation for the agent loop.

The daemon runs the synchronous agent loop in a worker thread via
``asyncio.to_thread``. A running Python thread cannot be preempted from the
outside, so cancelling ``task.cancel()`` on the awaiting coroutine stops the
SSE stream but leaves the worker running to completion. Stopping the agent has
to be *cooperative*: the adapter binds a ``threading.Event`` to the run context
(mirroring ``set_interaction_backend``) and the agent loop checks it at safe
checkpoints - between turns and before each code execution - bailing cleanly so
partial work is still persisted.

The Event is shared with the daemon's ``ActiveChat`` so the cancel endpoint
(running on the event loop) can signal the worker thread, which observes the
same object across the ``copy_context`` + ``to_thread`` boundary.
"""

import contextvars
import threading
from typing import Optional

_cancel_event_var: contextvars.ContextVar[Optional[threading.Event]] = contextvars.ContextVar(
    "cancel_event", default=None
)


def set_cancel_event(event: Optional[threading.Event]) -> None:
    """Bind a cancel Event to the current context (once per chat request by the adapter)."""
    _cancel_event_var.set(event)


def is_cancelled() -> bool:
    """True if a cancel Event is bound to this context and has been signalled."""
    event = _cancel_event_var.get()
    return event is not None and event.is_set()
