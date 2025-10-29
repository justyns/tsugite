"""EventBus for broadcasting events to multiple handlers."""

import sys
from typing import Callable, List

from .base import BaseEvent


class EventBus:
    """Broadcast events to multiple handlers with error isolation."""

    def __init__(self):
        self._handlers: List[Callable[[BaseEvent], None]] = []

    def subscribe(self, handler: Callable[[BaseEvent], None]) -> None:
        """Subscribe a handler to receive events.

        Args:
            handler: Callable that accepts a BaseEvent
        """
        if handler not in self._handlers:
            self._handlers.append(handler)

    def unsubscribe(self, handler: Callable[[BaseEvent], None]) -> None:
        """Unsubscribe a handler.

        Args:
            handler: Handler to remove
        """
        if handler in self._handlers:
            self._handlers.remove(handler)

    def emit(self, event: BaseEvent) -> None:
        """Emit event to all subscribers with error isolation.

        Args:
            event: Event to broadcast
        """
        for handler in self._handlers:
            try:
                handler(event)
            except Exception as e:
                import traceback

                print(f"Handler error: {e}", file=sys.stderr)
                traceback.print_exc(file=sys.stderr)

    def has_handlers(self) -> bool:
        """Check if any handlers are subscribed.

        Returns:
            True if handlers exist
        """
        return len(self._handlers) > 0

    def handler_count(self) -> int:
        """Get number of subscribed handlers.

        Returns:
            Handler count
        """
        return len(self._handlers)
