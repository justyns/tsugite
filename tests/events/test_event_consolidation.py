"""Tests for consolidated event system."""

import json

from tsugite.events import (
    BaseEvent,
    ErrorEvent,
    EventBus,
    FileWriteEvent,
    ObservationEvent,
    TaskStartEvent,
)
from tsugite.ui.jsonl import JSONLUIHandler


class TestEventBus:
    """Test EventBus dispatch to multiple handlers."""

    def test_event_bus_single_handler(self):
        """Test EventBus with single handler."""
        received_events = []

        def handler(event: BaseEvent):
            received_events.append(event)

        bus = EventBus()
        bus.subscribe(handler)

        event = TaskStartEvent(task="Test task", model="test-model")
        bus.emit(event)

        assert len(received_events) == 1
        assert received_events[0] == event

    def test_event_bus_multiple_handlers(self):
        """Test EventBus with multiple handlers."""
        handler1_events = []
        handler2_events = []

        def handler1(event: BaseEvent):
            handler1_events.append(event)

        def handler2(event: BaseEvent):
            handler2_events.append(event)

        bus = EventBus()
        bus.subscribe(handler1)
        bus.subscribe(handler2)

        event = TaskStartEvent(task="Test task", model="test-model")
        bus.emit(event)

        assert len(handler1_events) == 1
        assert len(handler2_events) == 1
        assert handler1_events[0] == event
        assert handler2_events[0] == event

    def test_event_bus_error_handling(self):
        """Test EventBus continues despite handler errors."""
        handler2_events = []

        def failing_handler(event: BaseEvent):
            raise RuntimeError("Handler error")

        def handler2(event: BaseEvent):
            handler2_events.append(event)

        bus = EventBus()
        bus.subscribe(failing_handler)
        bus.subscribe(handler2)

        event = TaskStartEvent(task="Test task", model="test-model")
        bus.emit(event)

        # handler2 should still receive the event
        assert len(handler2_events) == 1


class TestJSONLSerialization:
    """Test JSONL handler with new event structures."""

    def test_jsonl_observation_success(self, capsys):
        """Test JSONL serialization of successful ObservationEvent."""
        handler = JSONLUIHandler()

        event = ObservationEvent(success=True, observation="Tool result", tool="read_file")

        handler.handle_event(event)

        captured = capsys.readouterr()
        data = json.loads(captured.out.strip())

        assert data["type"] == "tool_result"
        assert data["tool"] == "read_file"
        assert data["success"] is True
        assert data["output"] == "Tool result"

    def test_jsonl_observation_failure(self, capsys):
        """Test JSONL serialization of failed ObservationEvent."""
        handler = JSONLUIHandler()

        event = ObservationEvent(success=False, error="Tool error", tool="read_file")

        handler.handle_event(event)

        captured = capsys.readouterr()
        data = json.loads(captured.out.strip())

        assert data["type"] == "tool_result"
        assert data["tool"] == "read_file"
        assert data["success"] is False
        assert data["error"] == "Tool error"

    def test_jsonl_error_event(self, capsys):
        """Test JSONL serialization of ErrorEvent."""
        handler = JSONLUIHandler()

        event = ErrorEvent(error="Critical error", error_type="Runtime Error", step=3)

        handler.handle_event(event)

        captured = capsys.readouterr()
        data = json.loads(captured.out.strip())

        assert data["type"] == "error"
        assert data["error"] == "Critical error"
        assert data["step"] == 3

    def test_jsonl_file_write_event(self, capsys):
        """Test JSONL serialization of FileWriteEvent."""
        handler = JSONLUIHandler()

        event = FileWriteEvent(path="/tmp/out.txt", line_count=10, byte_count=256, operation="tool_call")
        handler.handle_event(event)

        captured = capsys.readouterr()
        data = json.loads(captured.out.strip())

        assert data["type"] == "file_write"
        assert data["path"] == "/tmp/out.txt"
        assert data["line_count"] == 10
        assert data["byte_count"] == 256
        assert data["operation"] == "tool_call"
