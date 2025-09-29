"""Tests for the CLI interface."""

import pytest
from typer.testing import CliRunner
from pathlib import Path
from unittest.mock import patch, MagicMock
from tsugite.tsugite import app


@pytest.fixture
def cli_runner():
    """Create a CLI test runner."""
    return CliRunner()


@pytest.fixture
def mock_agent_execution():
    """Mock both run_agent and validate_agent_execution for CLI tests."""
    with (
        patch("tsugite.tsugite.run_agent") as mock_run_agent,
        patch("tsugite.tsugite.validate_agent_execution") as mock_validate,
    ):
        mock_run_agent.return_value = "Test agent execution completed"
        mock_validate.return_value = (True, "Agent is valid")
        yield mock_run_agent, mock_validate


def test_cli_help(cli_runner):
    """Test that help is displayed correctly."""
    result = cli_runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "Micro-agent runner for task automation" in result.stdout
    assert "run" in result.stdout
    assert "history" in result.stdout
    assert "version" in result.stdout


def test_version_command(cli_runner):
    """Test the version command."""
    result = cli_runner.invoke(app, ["version"])

    assert result.exit_code == 0
    assert "Tsugite version 0.1.0" in result.stdout


def test_run_command_nonexistent_file(cli_runner, temp_dir):
    """Test run command with nonexistent agent file."""
    nonexistent = temp_dir / "nonexistent.md"

    result = cli_runner.invoke(app, ["run", str(nonexistent), "test prompt"])

    assert result.exit_code == 1
    assert "Agent file not found" in result.stdout


def test_run_command_non_markdown_file(cli_runner, temp_dir):
    """Test run command with non-markdown file."""
    text_file = temp_dir / "test.txt"
    text_file.write_text("Not a markdown file")

    result = cli_runner.invoke(app, ["run", str(text_file), "test prompt"])

    assert result.exit_code == 1
    assert "must be a .md file" in result.stdout


def test_run_command_valid_file(cli_runner, sample_agent_file, mock_agent_execution):
    """Test run command with valid agent file."""
    result = cli_runner.invoke(app, ["run", str(sample_agent_file), "test prompt"])

    assert result.exit_code == 0
    assert "Agent: test_agent.md" in result.stdout
    assert "Task: test prompt" in result.stdout
    assert "Starting agent execution..." in result.stdout


def test_run_command_with_options(cli_runner, sample_agent_file, temp_dir, mock_agent_execution):
    """Test run command with various options."""
    # Test with --root option
    result = cli_runner.invoke(app, ["run", str(sample_agent_file), "test prompt", "--root", str(temp_dir)])

    assert result.exit_code == 0
    assert str(temp_dir) in result.stdout


def test_run_command_with_model_override(cli_runner, sample_agent_file, mock_agent_execution):
    """Test run command with model override."""
    result = cli_runner.invoke(
        app,
        [
            "run",
            str(sample_agent_file),
            "test prompt",
            "--model",
            "ollama:custom-model",
        ],
    )

    assert result.exit_code == 0


def test_run_command_non_interactive(cli_runner, sample_agent_file, mock_agent_execution):
    """Test run command in non-interactive mode."""
    result = cli_runner.invoke(app, ["run", str(sample_agent_file), "test prompt", "--non-interactive"])

    assert result.exit_code == 0


def test_run_command_no_color(cli_runner, sample_agent_file, mock_agent_execution):
    """Test run command with no color output."""
    result = cli_runner.invoke(app, ["run", str(sample_agent_file), "test prompt", "--no-color"])
    assert result.exit_code == 0
    # Check that ANSI escape codes are not present (basic check)
    assert "\033[" not in result.stdout


def test_run_command_json_logging(cli_runner, sample_agent_file, mock_agent_execution):
    """Test run command with JSON logging."""
    result = cli_runner.invoke(app, ["run", str(sample_agent_file), "test prompt", "--log-json"])
    assert result.exit_code == 0


def test_run_command_custom_history_dir(cli_runner, sample_agent_file, temp_dir, mock_agent_execution):
    """Test run command with custom history directory."""
    history_dir = temp_dir / "custom_history"
    result = cli_runner.invoke(
        app,
        [
            "run",
            str(sample_agent_file),
            "test prompt",
            "--history-dir",
            str(history_dir),
        ],
    )
    assert result.exit_code == 0


def test_run_command_all_options(cli_runner, sample_agent_file, temp_dir, mock_agent_execution):
    """Test run command with all options combined."""
    history_dir = temp_dir / "history"
    result = cli_runner.invoke(
        app,
        [
            "run",
            str(sample_agent_file),
            "test prompt",
            "--root",
            str(temp_dir),
            "--model",
            "custom-model",
            "--non-interactive",
            "--no-color",
            "--log-json",
            "--history-dir",
            str(history_dir),
        ],
    )
    assert result.exit_code == 0


def test_history_command_show(cli_runner):
    """Test history show command."""
    result = cli_runner.invoke(app, ["history", "show"])

    assert result.exit_code == 0
    assert "History show not yet implemented" in result.stdout


def test_history_command_show_with_since(cli_runner):
    """Test history show command with --since option."""
    result = cli_runner.invoke(app, ["history", "show", "--since", "2023-12-01"])

    assert result.exit_code == 0
    assert "History show not yet implemented" in result.stdout


def test_history_command_clear(cli_runner):
    """Test history clear command."""
    result = cli_runner.invoke(app, ["history", "clear"])

    assert result.exit_code == 0
    assert "History clear not yet implemented" in result.stdout


def test_run_command_directory_change(cli_runner, sample_agent_file, temp_dir, mock_agent_execution):
    """Test that --root option changes working directory."""
    marker_file = temp_dir / "marker.txt"
    marker_file.write_text("marker")
    # Mock os.chdir to verify it's called
    with patch("os.chdir") as mock_chdir:
        result = cli_runner.invoke(app, ["run", str(sample_agent_file), "test prompt", "--root", str(temp_dir)])
        assert result.exit_code == 0
        # Should be called twice: once to change to temp_dir, once to restore
        assert mock_chdir.call_count == 2
        mock_chdir.assert_any_call(str(temp_dir))


def test_no_args_shows_help(cli_runner):
    """Test that running without arguments shows help."""
    result = cli_runner.invoke(app, [])

    # Should show help due to no_args_is_help=True
    assert "Usage:" in result.stdout or "Commands:" in result.stdout


def test_run_command_relative_path(cli_runner, temp_dir, mock_agent_execution):
    """Test run command with relative path."""
    # Create agent file in temp directory
    agent_content = """---
name: relative_test
model: openai:gpt-4o-mini
tools: []
---
# Test Agent
"""
    agent_file = temp_dir / "agent.md"
    agent_file.write_text(agent_content)
    # Change working directory and use relative path
    import os

    original_cwd = os.getcwd()
    try:
        os.chdir(temp_dir)
        result = cli_runner.invoke(app, ["run", "agent.md", "test prompt"])
    finally:
        os.chdir(original_cwd)
    assert result.exit_code == 0


def test_invalid_command(cli_runner):
    """Test invalid command shows error."""
    result = cli_runner.invoke(app, ["invalid-command"])

    assert result.exit_code != 0


def test_cli_output_formatting(cli_runner, sample_agent_file, mock_agent_execution):
    """Test that CLI output is properly formatted."""
    result = cli_runner.invoke(app, ["run", str(sample_agent_file), "test prompt"])
    assert result.exit_code == 0
    # Check for panel formatting (Rich library)
    output = result.stdout
    assert "Agent:" in output
    assert "Task:" in output
    assert "Directory:" in output
    assert "test_agent.md" in output
    assert "test prompt" in output


def test_empty_prompt(cli_runner, sample_agent_file, mock_agent_execution):
    """Test run command with empty prompt."""
    result = cli_runner.invoke(app, ["run", str(sample_agent_file), ""])
    assert result.exit_code == 0
    assert "Task:" in result.stdout


def test_long_prompt(cli_runner, sample_agent_file, mock_agent_execution):
    """Test run command with very long prompt."""
    long_prompt = "A" * 1000
    result = cli_runner.invoke(app, ["run", str(sample_agent_file), long_prompt])
    assert result.exit_code == 0
    # The prompt should be in the output (possibly truncated in display)


def test_special_characters_in_prompt(cli_runner, sample_agent_file, mock_agent_execution):
    """Test run command with special characters in prompt."""
    special_prompt = "Test with émojis 🚀 and spécial characters! @#$%^&*()"
    result = cli_runner.invoke(app, ["run", str(sample_agent_file), special_prompt])
    assert result.exit_code == 0


def test_agent_file_with_spaces_in_path(cli_runner, temp_dir, mock_agent_execution):
    """Test run command with agent file path containing spaces."""
    agent_content = """---
name: space_test
model: openai:gpt-4o-mini
tools: []
---
# Test Agent
"""
    agent_dir = temp_dir / "my agents"
    agent_dir.mkdir()
    agent_file = agent_dir / "test agent.md"
    agent_file.write_text(agent_content)
    result = cli_runner.invoke(app, ["run", str(agent_file), "test prompt"])
    assert result.exit_code == 0


class TestAnimationCLIIntegration:
    """Test animation integration with CLI commands."""

    @patch("tsugite.tsugite.loading_animation")
    def test_animation_enabled_in_interactive_color_mode(
        self, mock_loading_animation, cli_runner, sample_agent_file, mock_agent_execution
    ):
        """Test that animation is enabled in default interactive color mode."""
        mock_loading_animation.return_value.__enter__ = MagicMock()
        mock_loading_animation.return_value.__exit__ = MagicMock(return_value=None)

        result = cli_runner.invoke(app, ["run", str(sample_agent_file), "test prompt"])

        assert result.exit_code == 0
        # Animation should be called with enabled=True
        mock_loading_animation.assert_called_once()
        call_args = mock_loading_animation.call_args
        assert call_args.kwargs["enabled"] is True
        assert call_args.kwargs["message"] == "Waiting for LLM response"

    @patch("tsugite.tsugite.loading_animation")
    def test_animation_disabled_with_non_interactive_flag(
        self, mock_loading_animation, cli_runner, sample_agent_file, mock_agent_execution
    ):
        """Test that animation is disabled with --non-interactive flag."""
        mock_loading_animation.return_value.__enter__ = MagicMock()
        mock_loading_animation.return_value.__exit__ = MagicMock(return_value=None)

        result = cli_runner.invoke(app, ["run", str(sample_agent_file), "test prompt", "--non-interactive"])

        assert result.exit_code == 0
        # Animation should be called with enabled=False
        mock_loading_animation.assert_called_once()
        call_args = mock_loading_animation.call_args
        assert call_args.kwargs["enabled"] is False

    @patch("tsugite.tsugite.loading_animation")
    def test_animation_disabled_with_no_color_flag(
        self, mock_loading_animation, cli_runner, sample_agent_file, mock_agent_execution
    ):
        """Test that animation is disabled with --no-color flag."""
        mock_loading_animation.return_value.__enter__ = MagicMock()
        mock_loading_animation.return_value.__exit__ = MagicMock(return_value=None)

        result = cli_runner.invoke(app, ["run", str(sample_agent_file), "test prompt", "--no-color"])

        assert result.exit_code == 0
        # Animation should be called with enabled=False
        mock_loading_animation.assert_called_once()
        call_args = mock_loading_animation.call_args
        assert call_args.kwargs["enabled"] is False

    @patch("tsugite.tsugite.loading_animation")
    def test_animation_disabled_with_both_flags(
        self, mock_loading_animation, cli_runner, sample_agent_file, mock_agent_execution
    ):
        """Test that animation is disabled with both --non-interactive and --no-color flags."""
        mock_loading_animation.return_value.__enter__ = MagicMock()
        mock_loading_animation.return_value.__exit__ = MagicMock(return_value=None)

        result = cli_runner.invoke(
            app, ["run", str(sample_agent_file), "test prompt", "--non-interactive", "--no-color"]
        )

        assert result.exit_code == 0
        # Animation should be called with enabled=False
        mock_loading_animation.assert_called_once()
        call_args = mock_loading_animation.call_args
        assert call_args.kwargs["enabled"] is False

    @patch("tsugite.tsugite.loading_animation")
    @patch("tsugite.tsugite.run_agent")
    def test_animation_context_manager_usage(
        self, mock_run_agent, mock_loading_animation, cli_runner, sample_agent_file
    ):
        """Test that animation context manager is properly used around run_agent."""
        mock_run_agent.return_value = "Test completion"
        mock_context = MagicMock()
        mock_loading_animation.return_value = mock_context

        with patch("tsugite.tsugite.validate_agent_execution") as mock_validate:
            mock_validate.return_value = (True, "Agent is valid")

            result = cli_runner.invoke(app, ["run", str(sample_agent_file), "test prompt"])

        assert result.exit_code == 0
        # Verify context manager was used
        mock_context.__enter__.assert_called_once()
        mock_context.__exit__.assert_called_once()
        # Verify run_agent was called
        mock_run_agent.assert_called_once()

    @patch("tsugite.tsugite.loading_animation")
    def test_animation_with_agent_execution_error(self, mock_loading_animation, cli_runner, sample_agent_file):
        """Test animation cleanup when agent execution fails."""
        mock_context = MagicMock()
        mock_loading_animation.return_value = mock_context

        with (
            patch("tsugite.tsugite.run_agent") as mock_run_agent,
            patch("tsugite.tsugite.validate_agent_execution") as mock_validate,
        ):

            mock_validate.return_value = (True, "Agent is valid")
            mock_run_agent.side_effect = RuntimeError("Agent execution failed")

            result = cli_runner.invoke(app, ["run", str(sample_agent_file), "test prompt"])

        assert result.exit_code == 1
        # Verify context manager was still properly used despite error
        mock_context.__enter__.assert_called_once()
        mock_context.__exit__.assert_called_once()

    @patch("tsugite.tsugite.loading_animation")
    def test_animation_console_parameter(
        self, mock_loading_animation, cli_runner, sample_agent_file, mock_agent_execution
    ):
        """Test that correct console instance is passed to animation."""
        mock_loading_animation.return_value.__enter__ = MagicMock()
        mock_loading_animation.return_value.__exit__ = MagicMock(return_value=None)

        result = cli_runner.invoke(app, ["run", str(sample_agent_file), "test prompt"])

        assert result.exit_code == 0
        # Verify console parameter was passed
        mock_loading_animation.assert_called_once()
        call_args = mock_loading_animation.call_args
        # Console should be the first positional argument
        if call_args.args:
            console_arg = call_args.args[0]
            assert console_arg is not None
            # Check it's a Console-like object (has the methods we expect)
            assert hasattr(console_arg, "print")
        else:
            # Console might be passed as keyword argument
            assert "console" in call_args.kwargs
            console_arg = call_args.kwargs["console"]
            assert console_arg is not None
            assert hasattr(console_arg, "print")
