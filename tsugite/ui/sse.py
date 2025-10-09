"""SSE UI handler for web interface."""

import asyncio
import os
import queue
from typing import Any, Dict

from rich.console import Console

from tsugite.ui.base import CustomUIHandler, UIEvent


class SSEUIHandler(CustomUIHandler):
    """UI handler that queues events for SSE streaming."""

    def __init__(self):
        super().__init__(
            console=Console(file=open(os.devnull, "w")),
            show_code=True,
            show_observations=True,
            show_llm_messages=False,
            show_execution_results=True,
            show_execution_logs=True,
            show_panels=False,
        )
        self.event_queue: queue.Queue = queue.Queue()
        self._final_result = None
        self._done = False

    def handle_event(self, event: UIEvent, data: Dict[str, Any]) -> None:
        """Handle a UI event by queuing it for SSE streaming."""
        # Store final answer for retrieval
        if event == UIEvent.FINAL_ANSWER:
            self._final_result = data.get("answer", "")
            self._done = True

        # Queue event for SSE (thread-safe)
        try:
            # Create SSE-compatible event
            sse_event = {"event": event.name.lower(), "data": data}
            self.event_queue.put(sse_event)
            print(f"[SSE] Queued event: {event.name}")
        except Exception as e:
            print(f"[SSE] Error queuing event: {e}")

    async def get_event(self) -> Dict[str, Any]:
        """Get the next event from the queue (async wrapper)."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.event_queue.get)

    def has_events(self) -> bool:
        """Check if there are pending events."""
        return not self.event_queue.empty()

    @property
    def is_done(self) -> bool:
        """Check if execution is complete."""
        return self._done

    @property
    def final_result(self) -> str:
        """Get the final result if available."""
        return self._final_result or ""
