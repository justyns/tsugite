"""Comprehensive integration tests for CLI animation feature."""

import threading
import time
from io import StringIO
from unittest.mock import MagicMock, patch

import pytest
from rich.console import Console
from typer.testing import CliRunner

from tsugite.cli import app


@pytest.fixture
def cli_runner():
    """Create a CLI test runner."""
    return CliRunner()


@pytest.fixture
def sample_agent_file(temp_dir):
    """Create a sample agent file for testing."""
    agent_content = """---
name: test-agent
model: gpt-4o-mini
tools:
  - shell
max_steps: 5
---
# Test Agent

This is a test agent for integration testing.

Task: {{ user_prompt }}
"""
    agent_file = temp_dir / "test_agent.md"
    agent_file.write_text(agent_content)
    return agent_file


@pytest.fixture
def temp_dir(tmp_path):
    """Create a temporary directory for test files."""
    return tmp_path


class TestAnimationIntegrationScenarios:
    """Test real-world animation integration scenarios."""

    def test_end_to_end_animation_flow(self, cli_runner, sample_agent_file):
        """Test complete animation flow from CLI invocation to completion."""
        with (
            patch("tsugite.agent_runner.run_agent") as mock_run_agent,
            patch("tsugite.agent_runner.validate_agent_execution") as mock_validate,
        ):
            mock_validate.return_value = (True, "Agent is valid")

            # Simulate a slow agent execution
            def slow_agent_execution(*args, **kwargs):
                time.sleep(0.2)  # Simulate LLM response time
                return "Agent completed successfully"

            mock_run_agent.side_effect = slow_agent_execution

            start_time = time.time()
            result = cli_runner.invoke(app, ["run", str(sample_agent_file), "test prompt"])
            end_time = time.time()

            assert result.exit_code == 0
            assert "Agent completed successfully" in result.stdout
            # Should take at least the simulated time
            assert end_time - start_time >= 0.2

    def test_animation_behavior_with_actual_console(self, cli_runner, sample_agent_file):
        """Test animation behavior with real console instances."""

        # Create test console instances
        color_console = Console(file=StringIO(), force_terminal=True, width=80)

        with (
            patch("tsugite.cli.console", color_console),
            patch("tsugite.agent_runner.run_agent") as mock_run_agent,
            patch("tsugite.agent_runner.validate_agent_execution") as mock_validate,
        ):
            mock_validate.return_value = (True, "Agent is valid")
            mock_run_agent.return_value = "Test completion"

            result = cli_runner.invoke(app, ["run", str(sample_agent_file), "test prompt"])

            assert result.exit_code == 0

    def test_animation_interruption_handling(self, cli_runner, sample_agent_file):
        """Test animation behavior when agent execution is interrupted."""

        with (
            patch("tsugite.agent_runner.run_agent") as mock_run_agent,
            patch("tsugite.agent_runner.validate_agent_execution") as mock_validate,
            patch("tsugite.ui.custom_agent_ui") as mock_custom_ui,
        ):
            mock_validate.return_value = (True, "Agent is valid")
            mock_run_agent.side_effect = KeyboardInterrupt("User interrupted")

            # Setup context manager mock
            mock_context = MagicMock()
            mock_custom_ui.return_value = mock_context

            result = cli_runner.invoke(app, ["run", str(sample_agent_file), "test prompt", "--native-ui"])

            assert result.exit_code == 130  # Standard exit code for KeyboardInterrupt
            # Context manager __exit__ should have been called during cleanup
            mock_context.__exit__.assert_called_once()

    def test_animation_with_different_terminal_sizes(self, cli_runner, sample_agent_file):
        """Test animation behavior with different terminal widths."""

        terminal_widths = [40, 80, 120, 200]

        for width in terminal_widths:
            with (
                patch("tsugite.agent_runner.run_agent") as mock_run_agent,
                patch("tsugite.agent_runner.validate_agent_execution") as mock_validate,
            ):
                mock_validate.return_value = (True, "Agent is valid")
                mock_run_agent.return_value = "Test completion"

                # Mock console with specific width
                test_console = Console(file=StringIO(), width=width, force_terminal=True)

                with patch("tsugite.cli.console", test_console):
                    result = cli_runner.invoke(app, ["run", str(sample_agent_file), "test prompt"])

                assert result.exit_code == 0

    def test_animation_with_long_running_agent(self, cli_runner, sample_agent_file):
        """Test animation during longer agent execution."""

        with (
            patch("tsugite.agent_runner.run_agent") as mock_run_agent,
            patch("tsugite.agent_runner.validate_agent_execution") as mock_validate,
        ):
            mock_validate.return_value = (True, "Agent is valid")

            # Simulate longer execution
            def long_agent_execution(*args, **kwargs):
                time.sleep(1.0)  # Longer simulated execution
                return "Long running agent completed"

            mock_run_agent.side_effect = long_agent_execution

            start_time = time.time()
            result = cli_runner.invoke(app, ["run", str(sample_agent_file), "test prompt"])
            end_time = time.time()

            assert result.exit_code == 0
            assert end_time - start_time >= 1.0


class TestAnimationModeDetection:
    """Test animation mode detection logic."""

    def test_interactive_mode_detection(self, cli_runner, sample_agent_file):
        """Test detection of interactive vs non-interactive mode with native UI."""

        test_cases = [
            # (args, expected_progress)
            (["--native-ui"], True),  # Native UI interactive mode
            (["--native-ui", "--non-interactive"], True),  # Non-interactive doesn't affect progress
            (["--native-ui", "--no-color"], False),  # No color mode disables progress
            (["--native-ui", "--non-interactive", "--no-color"], False),  # Both flags - no-color disables progress
        ]

        for args, expected_progress in test_cases:
            with (
                patch("tsugite.ui.custom_agent_ui") as mock_custom_ui,
                patch("tsugite.agent_runner.run_agent") as mock_run_agent,
                patch("tsugite.agent_runner.validate_agent_execution") as mock_validate,
            ):
                mock_validate.return_value = (True, "Agent is valid")
                mock_run_agent.return_value = "Test completion"
                mock_custom_ui.return_value.__enter__ = MagicMock(return_value=MagicMock())
                mock_custom_ui.return_value.__exit__ = MagicMock(return_value=None)

                command_args = ["run", str(sample_agent_file), "test prompt"] + args
                result = cli_runner.invoke(app, command_args)

                assert result.exit_code == 0
                mock_custom_ui.assert_called_once()
                call_args = mock_custom_ui.call_args
                assert call_args.kwargs["show_progress"] == expected_progress

    def test_color_detection_logic(self, cli_runner, sample_agent_file):
        """Test color detection affects animation mode."""

        with (
            patch("tsugite.agent_runner.run_agent") as mock_run_agent,
            patch("tsugite.agent_runner.validate_agent_execution") as mock_validate,
        ):
            mock_validate.return_value = (True, "Agent is valid")
            mock_run_agent.return_value = "Test completion"

            # Test with no-color flag
            with patch("tsugite.ui.custom_agent_ui") as mock_custom_ui:
                mock_custom_ui.return_value.__enter__ = MagicMock(return_value=MagicMock())
                mock_custom_ui.return_value.__exit__ = MagicMock(return_value=None)

                result = cli_runner.invoke(
                    app, ["run", str(sample_agent_file), "test prompt", "--native-ui", "--no-color"]
                )

                assert result.exit_code == 0
                call_args = mock_custom_ui.call_args
                assert call_args.kwargs["show_progress"] is False


class TestAnimationErrorHandling:
    """Test animation error handling scenarios."""

    def test_animation_failure_doesnt_break_cli(self, cli_runner, sample_agent_file):
        """Test that animation failures don't break CLI functionality."""

        with (
            patch("tsugite.animation.LoadingAnimation.start") as mock_start,
            patch("tsugite.agent_runner.run_agent") as mock_run_agent,
            patch("tsugite.agent_runner.validate_agent_execution") as mock_validate,
        ):
            mock_validate.return_value = (True, "Agent is valid")
            mock_run_agent.return_value = "Test completion"
            # Make animation start fail
            mock_start.side_effect = Exception("Animation failed")

            result = cli_runner.invoke(app, ["run", str(sample_agent_file), "test prompt"])

            # CLI should still work even if animation fails
            assert result.exit_code == 0

    def test_animation_stop_failure_handling(self, cli_runner, sample_agent_file):
        """Test handling of animation stop failures."""

        with (
            patch("tsugite.animation.LoadingAnimation.stop") as mock_stop,
            patch("tsugite.agent_runner.run_agent") as mock_run_agent,
            patch("tsugite.agent_runner.validate_agent_execution") as mock_validate,
        ):
            mock_validate.return_value = (True, "Agent is valid")
            mock_run_agent.return_value = "Test completion"
            # Make animation stop fail
            mock_stop.side_effect = Exception("Stop failed")

            result = cli_runner.invoke(app, ["run", str(sample_agent_file), "test prompt"])

            # CLI should still complete successfully
            assert result.exit_code == 0


class TestAnimationPerformance:
    """Test animation performance characteristics."""

    def test_animation_startup_overhead(self, cli_runner, sample_agent_file):
        """Test that animation doesn't add significant startup overhead."""

        with (
            patch("tsugite.agent_runner.run_agent") as mock_run_agent,
            patch("tsugite.agent_runner.validate_agent_execution") as mock_validate,
        ):
            mock_validate.return_value = (True, "Agent is valid")
            mock_run_agent.return_value = "Quick completion"

            # Measure time with animation enabled
            start_time = time.time()
            result1 = cli_runner.invoke(app, ["run", str(sample_agent_file), "test prompt"])
            enabled_time = time.time() - start_time

            # Measure time with animation disabled
            start_time = time.time()
            result2 = cli_runner.invoke(app, ["run", str(sample_agent_file), "test prompt", "--non-interactive"])
            disabled_time = time.time() - start_time

            assert result1.exit_code == 0
            assert result2.exit_code == 0

            # Animation should add minimal overhead (less than 1 second)
            overhead = enabled_time - disabled_time
            assert overhead < 1.0

    def test_animation_thread_cleanup(self, cli_runner, sample_agent_file):
        """Test that animation threads are properly cleaned up."""

        initial_thread_count = threading.active_count()

        with (
            patch("tsugite.agent_runner.run_agent") as mock_run_agent,
            patch("tsugite.agent_runner.validate_agent_execution") as mock_validate,
        ):
            mock_validate.return_value = (True, "Agent is valid")
            mock_run_agent.return_value = "Test completion"

            result = cli_runner.invoke(app, ["run", str(sample_agent_file), "test prompt"])

            assert result.exit_code == 0

        # Wait briefly for thread cleanup
        time.sleep(0.1)

        # Thread count should return to initial level
        final_thread_count = threading.active_count()
        assert final_thread_count <= initial_thread_count
