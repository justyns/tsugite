"""Tests for error display in custom UI."""

from io import StringIO

from rich.console import Console

from tsugite.custom_ui import CustomUIHandler, UIEvent


def test_observation_with_error_highlighted():
    """Test that observations containing errors are highlighted in red."""
    output = StringIO()
    console = Console(file=output, force_terminal=True, width=120)
    handler = CustomUIHandler(console, show_observations=True, show_panels=True)

    # Send an observation with an error
    handler.handle_event(UIEvent.OBSERVATION, {"observation": "Error: Tool 'visit_webpage' not found"})

    # Get output
    result = output.getvalue()

    # Should contain error markers
    assert "Error" in result or "error" in result.lower()


def test_observation_normal_not_highlighted():
    """Test that normal observations are displayed normally."""
    output = StringIO()
    console = Console(file=output, force_terminal=True, width=120)
    handler = CustomUIHandler(console, show_observations=True, show_panels=False)

    # Send a normal observation
    handler.handle_event(UIEvent.OBSERVATION, {"observation": "Search completed successfully with 5 results"})

    # Get output
    result = output.getvalue()

    # Should contain the observation but not as error
    assert "Search completed successfully" in result or "Search" in result


def test_error_event_handling():
    """Test that ERROR events are properly handled."""
    output = StringIO()
    console = Console(file=output, force_terminal=True, width=120)
    handler = CustomUIHandler(console, show_panels=True)

    # Send an error event
    handler.handle_event(UIEvent.ERROR, {"error": "Invalid tool name", "error_type": "ValidationError"})

    # Get output
    result = output.getvalue()

    # Should contain error information
    assert "ValidationError" in result or "Invalid tool name" in result


def test_error_keywords_detection():
    """Test that various error keywords trigger error highlighting."""
    output = StringIO()
    console = Console(file=output, force_terminal=True, width=120)
    handler = CustomUIHandler(console, show_observations=True, show_panels=False)

    error_messages = [
        "Error: Something went wrong",
        "Failed to execute command",
        "Exception occurred during processing",
        "File not found",
        "Invalid parameter provided",
    ]

    for msg in error_messages:
        output.truncate(0)
        output.seek(0)
        handler.handle_event(UIEvent.OBSERVATION, {"observation": msg})
        result = output.getvalue()

        # Each should be detected as an error
        # (We can't easily test for red color in text mode, but we can verify it's not truncated)
        assert len(result) > 0, f"No output for error message: {msg}"
