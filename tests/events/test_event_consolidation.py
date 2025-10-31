"""Tests for consolidated event system."""

import json

from tsugite.events import (
    BaseEvent,
    ErrorEvent,
    EventBus,
    EventType,
    ExecutionResultEvent,
    ObservationEvent,
    TaskStartEvent,
)
from tsugite.ui.jsonl import JSONLUIHandler


class TestEventStructure:
    """Test event creation with structured data."""

    def test_execution_result_with_logs_and_output(self):
        """Test ExecutionResultEvent with structured logs and output."""
        event = ExecutionResultEvent(
            logs=["print statement 1", "print statement 2"], output="Final result", success=True
        )

        assert event.logs == ["print statement 1", "print statement 2"]
        assert event.output == "Final result"
        assert event.success is True
        assert event.error is None

    def test_execution_result_with_error(self):
        """Test ExecutionResultEvent with error."""
        event = ExecutionResultEvent(success=False, error="Division by zero")

        assert event.success is False
        assert event.error == "Division by zero"
        assert event.logs == []
        assert event.output is None

    def test_observation_success(self):
        """Test ObservationEvent for successful tool execution."""
        event = ObservationEvent(success=True, observation="Tool executed successfully", tool="read_file")

        assert event.success is True
        assert event.observation == "Tool executed successfully"
        assert event.tool == "read_file"
        assert event.error is None

    def test_observation_failure(self):
        """Test ObservationEvent for failed tool execution."""
        event = ObservationEvent(success=False, error="File not found", tool="read_file")

        assert event.success is False
        assert event.error == "File not found"
        assert event.tool == "read_file"
        assert event.observation == ""

    def test_observation_code_execution(self):
        """Test ObservationEvent for code execution (no tool)."""
        event = ObservationEvent(observation="Code output", tool=None)

        assert event.observation == "Code output"
        assert event.tool is None
        assert event.success is True


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

    def test_jsonl_execution_result_success(self, capsys):
        """Test JSONL serialization of successful ExecutionResultEvent."""
        handler = JSONLUIHandler()

        event = ExecutionResultEvent(logs=["Log line 1", "Log line 2"], output="Final output", success=True)

        handler.handle_event(event)

        captured = capsys.readouterr()
        data = json.loads(captured.out.strip())

        assert data["type"] == "tool_result"
        assert data["tool"] == "code_execution"
        assert data["success"] is True
        assert "Log line 1" in data["output"]
        assert "Final output" in data["output"]

    def test_jsonl_execution_result_failure(self, capsys):
        """Test JSONL serialization of failed ExecutionResultEvent."""
        handler = JSONLUIHandler()

        event = ExecutionResultEvent(success=False, error="Execution error")

        handler.handle_event(event)

        captured = capsys.readouterr()
        data = json.loads(captured.out.strip())

        assert data["type"] == "tool_result"
        assert data["tool"] == "code_execution"
        assert data["success"] is False
        assert data["error"] == "Execution error"

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


class TestErrorConsolidation:
    """Test consistent error representation patterns."""

    def test_tool_error_uses_observation_event(self):
        """Tool errors should use ObservationEvent with success=False."""
        event = ObservationEvent(success=False, error="File not found: test.txt", tool="read_file")

        assert event.success is False
        assert event.error == "File not found: test.txt"
        assert event.tool == "read_file"
        assert event.event_type == EventType.OBSERVATION

    def test_execution_error_uses_execution_result_event(self):
        """Execution errors should use ExecutionResultEvent with success=False."""
        event = ExecutionResultEvent(success=False, error="ZeroDivisionError: division by zero")

        assert event.success is False
        assert event.error == "ZeroDivisionError: division by zero"
        assert event.event_type == EventType.EXECUTION_RESULT

    def test_general_error_uses_error_event(self):
        """General/fatal errors should use ErrorEvent."""
        event = ErrorEvent(error="Max turns exceeded", error_type="Execution Error", step=10)

        assert event.error == "Max turns exceeded"
        assert event.error_type == "Execution Error"
        assert event.step == 10
        assert event.event_type == EventType.ERROR
