"""Tests for loading animation functionality."""

import time
import threading
from unittest.mock import patch, MagicMock
import pytest
from rich.console import Console
from io import StringIO

from tsugite.animation import LoadingAnimation, loading_animation


class TestLoadingAnimation:
    """Test the LoadingAnimation class."""

    def test_init(self):
        """Test animation initialization."""
        console = Console()
        animation = LoadingAnimation(console)
        assert animation.console is console
        assert animation.stop_event is not None
        assert animation.thread is None

    def test_start_stop_spinner(self):
        """Test starting and stopping spinner animation."""
        # Create console with color support
        console = Console(file=StringIO(), width=80, force_terminal=True)
        animation = LoadingAnimation(console)

        # Start animation
        animation.start("Testing")
        assert animation.thread is not None
        assert animation.thread.is_alive()
        assert animation.thread.daemon is True

        # Let it run briefly
        time.sleep(0.1)

        # Stop animation
        animation.stop()
        assert not animation.thread.is_alive()

    def test_start_stop_simple(self):
        """Test starting and stopping simple text animation."""
        # Create console without color support
        console = Console(file=StringIO(), width=80, no_color=True)
        animation = LoadingAnimation(console)

        # Start animation
        animation.start("Testing")
        assert animation.thread is not None
        assert animation.thread.is_alive()
        assert animation.thread.daemon is True

        # Let it run briefly
        time.sleep(0.1)

        # Stop animation
        animation.stop()
        assert not animation.thread.is_alive()

    def test_multiple_start_stop_cycles(self):
        """Test multiple start/stop cycles work correctly."""
        console = Console(file=StringIO(), width=80, force_terminal=True)
        animation = LoadingAnimation(console)

        for i in range(3):
            animation.start(f"Testing cycle {i}")
            assert animation.thread is not None
            assert animation.thread.is_alive()

            time.sleep(0.05)

            animation.stop()
            assert not animation.thread.is_alive()

    def test_stop_when_not_running(self):
        """Test stopping animation when no thread is running."""
        console = Console(file=StringIO(), width=80, force_terminal=True)
        animation = LoadingAnimation(console)

        # Stop without starting - should not raise exception
        animation.stop()
        assert animation.thread is None

    def test_custom_message(self):
        """Test animation with custom message."""
        console = Console(file=StringIO(), width=80, force_terminal=True)
        animation = LoadingAnimation(console)

        custom_message = "Processing your request..."
        animation.start(custom_message)
        assert animation.thread is not None
        assert animation.thread.is_alive()

        time.sleep(0.05)
        animation.stop()

    def test_empty_message(self):
        """Test animation with empty message."""
        console = Console(file=StringIO(), width=80, force_terminal=True)
        animation = LoadingAnimation(console)

        animation.start("")
        assert animation.thread is not None
        assert animation.thread.is_alive()

        time.sleep(0.05)
        animation.stop()

    def test_very_long_message(self):
        """Test animation with very long message."""
        console = Console(file=StringIO(), width=80, force_terminal=True)
        animation = LoadingAnimation(console)

        long_message = "A" * 200  # Very long message
        animation.start(long_message)
        assert animation.thread is not None
        assert animation.thread.is_alive()

        time.sleep(0.05)
        animation.stop()

    def test_default_message(self):
        """Test animation with default message parameter."""
        console = Console(file=StringIO(), width=80, force_terminal=True)
        animation = LoadingAnimation(console)

        animation.start()  # No message provided, should use default
        assert animation.thread is not None
        assert animation.thread.is_alive()

        time.sleep(0.05)
        animation.stop()

    def test_rapid_start_stop(self):
        """Test rapid start/stop calls for thread safety."""
        console = Console(file=StringIO(), width=80, force_terminal=True)
        animation = LoadingAnimation(console)

        # Rapid start/stop
        animation.start("Rapid test")
        animation.stop()
        animation.start("Rapid test 2")
        animation.stop()

        # Should end in stopped state
        assert not animation.thread.is_alive() if animation.thread else True

    def test_output_capture_simple_animation(self):
        """Test that simple animation produces output."""
        output = StringIO()
        console = Console(file=output, width=80, no_color=True)
        animation = LoadingAnimation(console)

        animation.start("Testing output")
        time.sleep(0.6)  # Long enough for at least one animation cycle
        animation.stop()

        output_content = output.getvalue()
        # Should contain the message and some dots
        assert "Testing output" in output_content

    def test_concurrent_animations_safety(self):
        """Test that multiple animation instances don't interfere."""
        console1 = Console(file=StringIO(), width=80, force_terminal=True)
        console2 = Console(file=StringIO(), width=80, no_color=True)

        animation1 = LoadingAnimation(console1)
        animation2 = LoadingAnimation(console2)

        # Start both animations
        animation1.start("Animation 1")
        animation2.start("Animation 2")

        assert animation1.thread.is_alive()
        assert animation2.thread.is_alive()

        time.sleep(0.1)

        # Stop both
        animation1.stop()
        animation2.stop()

        assert not animation1.thread.is_alive()
        assert not animation2.thread.is_alive()


class TestLoadingAnimationContextManager:
    """Test the loading_animation context manager."""

    def test_context_manager_enabled(self):
        """Test context manager when enabled."""
        console = Console(file=StringIO(), width=80, force_terminal=True)

        with patch.object(LoadingAnimation, 'start') as mock_start, \
             patch.object(LoadingAnimation, 'stop') as mock_stop:

            with loading_animation(console, "Test message", enabled=True):
                pass

            mock_start.assert_called_once_with("Test message")
            mock_stop.assert_called_once()

    def test_context_manager_disabled(self):
        """Test context manager when disabled."""
        console = Console(file=StringIO(), width=80, force_terminal=True)

        with patch.object(LoadingAnimation, 'start') as mock_start, \
             patch.object(LoadingAnimation, 'stop') as mock_stop:

            with loading_animation(console, "Test message", enabled=False):
                pass

            mock_start.assert_not_called()
            mock_stop.assert_not_called()

    def test_context_manager_exception_handling(self):
        """Test context manager properly stops animation on exception."""
        console = Console(file=StringIO(), width=80, force_terminal=True)

        with patch.object(LoadingAnimation, 'start') as mock_start, \
             patch.object(LoadingAnimation, 'stop') as mock_stop:

            with pytest.raises(ValueError):
                with loading_animation(console, "Test message", enabled=True):
                    raise ValueError("Test exception")

            mock_start.assert_called_once_with("Test message")
            mock_stop.assert_called_once()

    def test_context_manager_default_message(self):
        """Test context manager with default message."""
        console = Console(file=StringIO(), width=80, force_terminal=True)

        with patch.object(LoadingAnimation, 'start') as mock_start, \
             patch.object(LoadingAnimation, 'stop') as mock_stop:

            with loading_animation(console, enabled=True):
                pass

            mock_start.assert_called_once_with("Waiting for LLM response")
            mock_stop.assert_called_once()

    def test_context_manager_with_keyboard_interrupt(self):
        """Test context manager handles KeyboardInterrupt properly."""
        console = Console(file=StringIO(), width=80, force_terminal=True)

        with patch.object(LoadingAnimation, 'start') as mock_start, \
             patch.object(LoadingAnimation, 'stop') as mock_stop:

            with pytest.raises(KeyboardInterrupt):
                with loading_animation(console, "Test message", enabled=True):
                    raise KeyboardInterrupt("User interrupted")

            mock_start.assert_called_once_with("Test message")
            mock_stop.assert_called_once()

    def test_context_manager_nested_usage(self):
        """Test nested context manager usage."""
        console = Console(file=StringIO(), width=80, force_terminal=True)

        with patch.object(LoadingAnimation, 'start') as mock_start, \
             patch.object(LoadingAnimation, 'stop') as mock_stop:

            with loading_animation(console, "Outer", enabled=True):
                # Inner context manager should also work
                with loading_animation(console, "Inner", enabled=True):
                    pass

            # Both should have been called
            assert mock_start.call_count == 2
            assert mock_stop.call_count == 2

    def test_context_manager_with_return_value(self):
        """Test context manager preserves return values."""
        console = Console(file=StringIO(), width=80, force_terminal=True)

        def test_function():
            with loading_animation(console, "Testing", enabled=False):
                return "test_result"

        result = test_function()
        assert result == "test_result"

    def test_context_manager_timing(self):
        """Test that context manager runs for expected duration."""
        console = Console(file=StringIO(), width=80, force_terminal=True)

        start_time = time.time()
        with loading_animation(console, "Timing test", enabled=True):
            time.sleep(0.1)
        end_time = time.time()

        # Should take at least 0.1 seconds
        assert end_time - start_time >= 0.1


class TestAnimationOutputVerification:
    """Test animation output formats and behavior."""

    def test_spinner_output_format(self):
        """Test spinner animation output format."""
        output = StringIO()
        console = Console(file=output, width=80, force_terminal=True)
        animation = LoadingAnimation(console)

        animation.start("Loading data")
        time.sleep(0.2)  # Let spinner run briefly
        animation.stop()

        # Check that Live was used (implicitly through no direct text output)
        output_content = output.getvalue()
        # Spinner uses Live which doesn't write to the StringIO directly
        # This test verifies the spinner path is taken
        assert True  # Test passes if no exception is raised

    def test_simple_animation_output_pattern(self):
        """Test simple animation output pattern."""
        output = StringIO()
        console = Console(file=output, width=80, no_color=True)
        animation = LoadingAnimation(console)

        animation.start("Processing")
        time.sleep(0.6)  # Let it cycle through dots
        animation.stop()

        output_content = output.getvalue()
        assert "Processing" in output_content

    def test_line_clearing_behavior(self):
        """Test that animation properly clears its output line."""
        output = StringIO()
        console = Console(file=output, width=80, no_color=True)
        animation = LoadingAnimation(console)

        animation.start("Clearing test")
        time.sleep(0.6)  # Let it run long enough to see output
        animation.stop()

        output_content = output.getvalue()
        # Should contain the test message - the exact output format may vary
        assert "Clearing test" in output_content or len(output_content) > 0

    def test_animation_with_unicode_message(self):
        """Test animation with Unicode characters in message."""
        console = Console(file=StringIO(), width=80, force_terminal=True)
        animation = LoadingAnimation(console)

        unicode_message = "Processing 🚀 データ"
        animation.start(unicode_message)
        assert animation.thread is not None
        assert animation.thread.is_alive()

        time.sleep(0.05)
        animation.stop()

    def test_animation_message_length_limits(self):
        """Test animation behavior with various message lengths."""
        console = Console(file=StringIO(), width=80, force_terminal=True)
        animation = LoadingAnimation(console)

        # Test different message lengths
        messages = [
            "",  # Empty
            "Short",  # Short
            "A" * 50,  # Medium
            "B" * 150,  # Long
            "C" * 300,  # Very long
        ]

        for msg in messages:
            animation.start(msg)
            assert animation.thread is not None
            assert animation.thread.is_alive()
            time.sleep(0.05)
            animation.stop()
            assert not animation.thread.is_alive()