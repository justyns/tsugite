"""Tests for send_message tool."""

from tsugite.events import EventBus, InfoEvent
from tsugite.tools.interactive import send_message
from tsugite.ui_context import set_ui_context


def test_send_message_emits_info_event():
    """send_message() should emit InfoEvent."""
    events = []
    bus = EventBus()
    bus.subscribe(lambda e: events.append(e))
    set_ui_context(event_bus=bus)

    result = send_message("Test message")

    assert result == "Message sent: Test message"
    assert len(events) == 1
    assert isinstance(events[0], InfoEvent)
    assert events[0].message == "Test message"


def test_send_message_without_event_bus():
    """send_message() should work without event bus."""
    set_ui_context(event_bus=None)
    result = send_message("Test")
    assert result == "Message sent: Test"


def test_send_message_with_formatted_string():
    """send_message() should handle formatted strings."""
    events = []
    bus = EventBus()
    bus.subscribe(lambda e: events.append(e))
    set_ui_context(event_bus=bus)

    count = 42
    result = send_message(f"Found {count} items")

    assert result == "Message sent: Found 42 items"
    assert events[0].message == "Found 42 items"


def test_send_message_multiple_calls():
    """send_message() can be called multiple times."""
    events = []
    bus = EventBus()
    bus.subscribe(lambda e: events.append(e))
    set_ui_context(event_bus=bus)

    send_message("First update")
    send_message("Second update")
    send_message("Third update")

    assert len(events) == 3
    assert events[0].message == "First update"
    assert events[1].message == "Second update"
    assert events[2].message == "Third update"
