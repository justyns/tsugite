"""Event handlers declare what they handle; the dispatch map is derived.

Each UI handler used to carry a hand-written `dict[type, str]` mapping event
classes to method-name strings. Three copies, no type checking, and a typo'd or
forgotten entry failed silently at runtime.

These tests deliberately assert *invariants* rather than a literal expected map.
A copy of the mapping here would be the same hand-maintained table in a new
place: adding a handler would fail an assert, you would paste the entry in, and
nothing would have been checked.
"""

import pytest

from tsugite.events import ContentBlockEvent, TaskStartEvent, WarningEvent
from tsugite.ui.base import CustomUIHandler
from tsugite.ui.dispatch import EventDispatchMixin, handles
from tsugite.ui.jsonl import JSONLUIHandler
from tsugite.ui.plain import PlainUIHandler
from tsugite.ui.repl_handler import ReplUIHandler

HANDLERS = [CustomUIHandler, JSONLUIHandler, ReplUIHandler]


@pytest.mark.parametrize("handler_cls", HANDLERS)
def test_every_handler_method_is_registered(handler_cls):
    """A `_handle_*` method that nothing routes to is dead code or a missing decorator.

    This is the invariant the old string tables could not express: they listed
    what was registered, but nothing tied that back to what was implemented.
    """
    defined = {name for name in dir(handler_cls) if name.startswith("_handle_")}
    registered = set(handler_cls._DISPATCH.values())

    assert not (defined - registered), (
        f"{handler_cls.__name__} defines handlers nothing dispatches to: {sorted(defined - registered)}"
    )


@pytest.mark.parametrize("handler_cls", HANDLERS)
def test_every_dispatch_target_is_callable(handler_cls):
    """Guards a subclass shadowing an inherited handler with a non-callable."""
    for event, method_name in handler_cls._DISPATCH.items():
        assert callable(getattr(handler_cls, method_name, None)), (
            f"{handler_cls.__name__} routes {event.__name__} to non-callable {method_name!r}"
        )


@pytest.mark.parametrize("handler_cls", HANDLERS)
def test_handlers_route_something(handler_cls):
    assert handler_cls._DISPATCH, f"{handler_cls.__name__} routes no events at all"


def test_repl_does_not_claim_content_blocks():
    """Regression: ReplUIHandler's table claimed ContentBlockEvent but the class
    never defined `_handle_content_block`, so every content block raised
    AttributeError into EventBus's error isolation and printed a traceback."""
    assert ContentBlockEvent not in ReplUIHandler._DISPATCH


def test_plain_inherits_the_console_dispatch_exactly():
    """PlainUIHandler overrides a dozen handlers without repeating `@handles`.

    That works because the map holds method *names* and `getattr` resolves the
    override. If the map ever held functions instead, Plain would silently run
    CustomUIHandler's implementations.
    """
    assert PlainUIHandler._DISPATCH == CustomUIHandler._DISPATCH


def test_a_new_handler_needs_no_table_edit():
    """Declaring `@handles` is the whole registration step."""

    class Probe(EventDispatchMixin):
        def __init__(self):
            self.seen = []

        @handles(TaskStartEvent)
        def _handle_task_start(self, event):
            self.seen.append(event)

    probe = Probe()
    event = TaskStartEvent(task="t", model="m")
    probe.handle_event(event)

    assert probe.seen == [event]


def test_unhandled_events_are_ignored():
    """Each handler covers a subset of event types on purpose."""

    class Probe(EventDispatchMixin):
        @handles(TaskStartEvent)
        def _handle_task_start(self, event): ...

    Probe().handle_event(WarningEvent(message="nobody handles this"))


def test_subclass_overrides_win_without_redeclaring():
    class Base(EventDispatchMixin):
        @handles(TaskStartEvent)
        def _handle_task_start(self, event):
            self.seen = "base"

    class Child(Base):
        def _handle_task_start(self, event):
            self.seen = "child"

    child = Child()
    child.handle_event(TaskStartEvent(task="t", model="m"))

    assert child.seen == "child"
    assert TaskStartEvent in Child._DISPATCH


def test_subclass_may_repoint_an_event_at_a_new_method():
    class Base(EventDispatchMixin):
        @handles(TaskStartEvent)
        def _handle_task_start(self, event):
            self.seen = "base"

    class Child(Base):
        @handles(TaskStartEvent)
        def _handle_differently(self, event):
            self.seen = "child"

    child = Child()
    child.handle_event(TaskStartEvent(task="t", model="m"))

    assert child.seen == "child"
    assert Child._DISPATCH[TaskStartEvent] == "_handle_differently"


def test_two_methods_claiming_one_event_is_rejected():
    """The silent-shadowing failure this mechanism exists to prevent.

    Caught at class-creation time rather than becoming a coin flip over dict
    insertion order.
    """
    with pytest.raises(TypeError, match="registers TaskStartEvent twice"):

        class Broken(EventDispatchMixin):
            @handles(TaskStartEvent)
            def _handle_one(self, event): ...

            @handles(TaskStartEvent)
            def _handle_two(self, event): ...


def test_handles_requires_an_event_type():
    with pytest.raises(ValueError):
        handles()
