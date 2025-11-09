"""Tests for REPL UI handler."""

from unittest.mock import MagicMock, patch

from rich.console import Console

from tsugite.events import (
    CostSummaryEvent,
    ErrorEvent,
    FinalAnswerEvent,
    StepStartEvent,
    StreamChunkEvent,
    StreamCompleteEvent,
    TaskStartEvent,
    ToolCallEvent,
)
from tsugite.ui.repl_handler import ReplUIHandler


def test_handler_initialization():
    """Test handler initialization."""
    console = Console()
    handler = ReplUIHandler(console)

    assert handler.console == console
    assert handler.compact is True
    assert handler.current_status is None


def test_handle_task_start():
    """Test task start event."""
    console = Console()
    handler = ReplUIHandler(console, show_debug_messages=True)

    event = TaskStartEvent(task="test task", model="test-model")

    with patch.object(console, "print") as mock_print:
        handler.handle_event(event)
        # Should print model in debug mode
        assert mock_print.called


def test_handle_step_start():
    """Test step start event."""
    console = Console()
    handler = ReplUIHandler(console)

    event = StepStartEvent(step=1, recovering_from_error=False)

    with patch.object(console, "status") as mock_status:
        mock_status_obj = MagicMock()
        mock_status.return_value = mock_status_obj

        handler.handle_event(event)

        # Should create status spinner
        mock_status.assert_called_once()
        mock_status_obj.start.assert_called_once()


def test_handle_tool_call():
    """Test tool call event."""
    console = Console()
    handler = ReplUIHandler(console, compact=True)

    event = ToolCallEvent(tool="Tool: test_tool")

    with patch.object(console, "print") as mock_print:
        handler.handle_event(event)
        # Should print tool name in compact mode
        assert mock_print.called


def test_handle_final_answer():
    """Test final answer event."""
    console = Console()
    handler = ReplUIHandler(console)

    # Set up a mock status
    mock_status = MagicMock()
    handler.current_status = mock_status

    event = FinalAnswerEvent(answer="Test answer")

    with patch.object(console, "print") as mock_print:
        handler.handle_event(event)
        # Should stop status and print answer
        mock_status.stop.assert_called_once()
        assert mock_print.called
        assert handler.current_status is None


def test_handle_error():
    """Test error event."""
    console = Console()
    handler = ReplUIHandler(console)

    event = ErrorEvent(error="Test error", error_type="TestError", suppress_from_ui=False)

    with patch.object(console, "print") as mock_print:
        handler.handle_event(event)
        # Should print error panel
        assert mock_print.called


def test_handle_error_suppressed():
    """Test suppressed error event."""
    console = Console()
    handler = ReplUIHandler(console, show_debug_messages=False)

    event = ErrorEvent(error="Test error", error_type="TestError", suppress_from_ui=True)

    with patch.object(console, "print") as mock_print:
        handler.handle_event(event)
        # Should not print suppressed error
        assert not mock_print.called


def test_handle_cost_summary():
    """Test cost summary event."""
    console = Console()
    handler = ReplUIHandler(console)

    event = CostSummaryEvent(
        cost=0.001234,
        tokens=1000,
        duration_seconds=2.5,
        cached_tokens=100,
        cache_creation_input_tokens=50,
        cache_read_input_tokens=50,
    )

    with patch.object(console, "print") as mock_print:
        handler.handle_event(event)
        # Should print cost summary
        assert mock_print.called


def test_contains_error():
    """Test error detection."""
    handler = ReplUIHandler(Console())

    assert handler._contains_error("Error: something went wrong")
    assert handler._contains_error("FAILED to process")
    assert handler._contains_error("Exception occurred")
    assert not handler._contains_error("Success")
    assert not handler._contains_error("Everything is fine")


def test_stop():
    """Test stopping handler."""
    console = Console()
    handler = ReplUIHandler(console)

    # Set up mock status
    mock_status = MagicMock()
    handler.current_status = mock_status

    handler.stop()

    # Should stop status
    mock_status.stop.assert_called_once()
    assert handler.current_status is None


def test_handle_stream_chunk():
    """Test stream chunk event - prints chunks in real-time."""
    console = Console()
    handler = ReplUIHandler(console)

    event = StreamChunkEvent(chunk="Hello ")

    with patch.object(console, "print") as mock_print:
        handler.handle_event(event)
        # Should print chunks in real-time for better UX
        assert mock_print.called
        # First call should be newline (after stopping spinner), second should be the chunk
        assert mock_print.call_count == 2
        # Should accumulate chunks
        assert handler.streaming_content == "Hello "
        assert handler.is_streaming is True


def test_handle_stream_complete():
    """Test stream complete event."""
    console = Console()
    handler = ReplUIHandler(console)

    # Set up streaming state
    handler.streaming_content = "Hello world"
    handler.is_streaming = True

    event = StreamCompleteEvent()

    with patch.object(console, "print") as mock_print:
        handler.handle_event(event)
        # Should print newline
        assert mock_print.called
        # Should reset streaming state
        assert handler.is_streaming is False
        assert handler.streaming_content == ""
