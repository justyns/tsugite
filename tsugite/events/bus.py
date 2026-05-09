"""EventBus for broadcasting events to multiple handlers."""

import sys
import traceback
from dataclasses import dataclass
from typing import Any, Callable, List, Optional

from .base import BaseEvent


@dataclass
class Subscription:
    """A filtered event subscription used by plugins.

    event_name: match against BaseEvent.event_name; None receives all events.
    predicate: optional gate evaluated after event_name matches.
    """

    handler: Callable[[BaseEvent], None]
    event_name: Optional[str] = None
    predicate: Optional[Callable[[BaseEvent], bool]] = None


def _safe_invoke(fn: Callable, event: BaseEvent, label: str, default: Any = None) -> Any:
    try:
        return fn(event)
    except Exception as e:
        print(f"{label} error: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return default


def subscribe(
    event_name: Optional[str] = None,
    *,
    predicate: Optional[Callable[[BaseEvent], bool]] = None,
):
    """Register a function as a plugin event subscriber.

    The decorator wraps the function in a Subscription and appends it to the
    plugin subscription registry at import time. Combined with a module-only
    entry point in tsugite.plugins, no register_event_subscribers() function
    is needed.
    """

    def decorator(fn: Callable[[BaseEvent], None]) -> Callable[[BaseEvent], None]:
        from tsugite.plugins import _plugin_subscriptions

        _plugin_subscriptions.append(Subscription(handler=fn, event_name=event_name, predicate=predicate))
        return fn

    return decorator


class EventBus:
    """Broadcast events to multiple handlers with error isolation."""

    def __init__(self):
        self._handlers: List[Callable[[BaseEvent], None]] = []
        self._filtered: List[Subscription] = []
        self._attach_plugin_subscriptions()

    def _attach_plugin_subscriptions(self) -> None:
        try:
            from tsugite.plugins import get_plugin_subscriptions
        except ImportError:
            return
        self._filtered.extend(get_plugin_subscriptions())

    def subscribe(self, handler: Callable[[BaseEvent], None]) -> None:
        """Subscribe a handler to receive every event (no filtering)."""
        if handler not in self._handlers:
            self._handlers.append(handler)

    def unsubscribe(self, handler: Callable[[BaseEvent], None]) -> None:
        if handler in self._handlers:
            self._handlers.remove(handler)

    def subscribe_filtered(
        self,
        handler: Callable[[BaseEvent], None],
        event_name: Optional[str] = None,
        predicate: Optional[Callable[[BaseEvent], bool]] = None,
    ) -> None:
        """Subscribe a handler to events matching event_name and predicate.

        event_name=None receives all events. predicate is evaluated only if
        the event_name filter matches (or is None).
        """
        self._filtered.append(Subscription(handler=handler, event_name=event_name, predicate=predicate))

    def unsubscribe_filtered(self, handler: Callable[[BaseEvent], None]) -> None:
        self._filtered = [s for s in self._filtered if s.handler != handler]

    def emit(self, event: BaseEvent) -> None:
        """Emit event to all subscribers with error isolation."""
        for handler in self._handlers:
            _safe_invoke(handler, event, "Handler")

        if not self._filtered:
            return
        name = event.event_name
        for sub in self._filtered:
            if sub.event_name is not None and sub.event_name != name:
                continue
            if sub.predicate is not None and not _safe_invoke(
                sub.predicate, event, "Subscription predicate", default=False
            ):
                continue
            _safe_invoke(sub.handler, event, "Subscription handler")

    def has_handlers(self) -> bool:
        return len(self._handlers) > 0 or len(self._filtered) > 0

    def handler_count(self) -> int:
        return len(self._handlers) + len(self._filtered)
