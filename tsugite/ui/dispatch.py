"""Event-to-handler dispatch shared by the UI handlers.

Handlers declare the events they take with `@handles(...)`; the class-level
`_DISPATCH` map is derived from those markers when the class is created. This
keeps the declaration next to the method that implements it, so a handler cannot
be registered for an event it does not define, or defined and never registered.
"""

from tsugite.events.base import BaseEvent

_MARKER = "__handles_events__"


def handles(*event_types: type[BaseEvent]):
    """Mark a method as the handler for one or more event types.

    Several events may share a handler when their payloads are the same shape:

        @handles(FileReadEvent, FileWriteEvent)
        def _handle_file_io(self, event): ...
    """
    if not event_types:
        raise ValueError("@handles requires at least one event type")

    def decorator(fn):
        setattr(fn, _MARKER, event_types)
        return fn

    return decorator


def _collect(cls: type) -> dict[type[BaseEvent], str]:
    """Build the dispatch map, walking the MRO base-first.

    Later bases overwrite earlier ones, so a subclass re-pointing an inherited
    event at a new method wins. Within a single class, two methods claiming the
    same event is always a mistake - one would silently shadow the other, which
    is the failure this whole mechanism exists to prevent.
    """
    dispatch: dict[type[BaseEvent], str] = {}
    for base in reversed(cls.__mro__):
        claimed: dict[type[BaseEvent], str] = {}
        for name, attr in vars(base).items():
            for event_type in getattr(attr, _MARKER, ()):
                if event_type in claimed:
                    raise TypeError(
                        f"{base.__qualname__} registers {event_type.__name__} twice: "
                        f"{claimed[event_type]}() and {name}()"
                    )
                claimed[event_type] = name
        dispatch.update(claimed)
    return dispatch


class EventDispatchMixin:
    """Routes events to the methods that declared them with `@handles`."""

    _DISPATCH: dict[type[BaseEvent], str] = {}

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        cls._DISPATCH = _collect(cls)

    def handle_event(self, event: BaseEvent) -> None:
        """Route `event` to its handler; ignore events this handler doesn't take.

        An unhandled event is not an error: each handler deliberately covers a
        subset of the event types (the JSONL protocol carries audit events the
        console has no way to render, and vice versa).

        Subclasses that need to do work around the dispatch - take a lock,
        refresh a live display - override this and call `super().handle_event()`.
        """
        method_name = self._DISPATCH.get(type(event))
        if method_name is not None:
            getattr(self, method_name)(event)
