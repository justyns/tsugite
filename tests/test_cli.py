"""Tests for the CLI interface."""

from unittest.mock import MagicMock, patch

import pytest

from tsugite.cli import app


@pytest.fixture
def mock_agent_execution():
    """Mock both run_agent and validate_agent_execution for CLI tests."""
    with (
        patch("tsugite.agent_runner.run_agent") as mock_run_agent,
        patch("tsugite.md_agents.validate_agent_execution") as mock_validate,
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
    # Progress messages go to stderr
    assert "Agent: test_agent.md" in result.stderr
    assert "Task: test prompt" in result.stderr
    # Final result goes to stdout
    assert "Test agent execution completed" in result.stdout


def test_run_command_with_options(cli_runner, sample_agent_file, temp_dir, mock_agent_execution):
    """Test run command with various options."""
    # Test with --root option
    result = cli_runner.invoke(app, ["run", str(sample_agent_file), "test prompt", "--root", str(temp_dir)])

    assert result.exit_code == 0
    # Directory info goes to stderr
    assert str(temp_dir) in result.stderr


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


@patch("tsugite.utils.should_use_plain_output", return_value=False)
@patch("tsugite.ui.custom_agent_ui")
def test_show_reasoning_flag(mock_custom_ui, mock_plain_output, cli_runner, sample_agent_file, mock_agent_execution):
    """Test --show-reasoning and --no-show-reasoning flags."""
    mock_custom_ui.return_value.__enter__ = MagicMock(return_value=MagicMock())
    mock_custom_ui.return_value.__exit__ = MagicMock(return_value=None)

    # Test default (should enable show_llm_messages)
    result = cli_runner.invoke(app, ["run", str(sample_agent_file), "test prompt"])
    assert result.exit_code == 0
    mock_custom_ui.assert_called()
    call_kwargs = mock_custom_ui.call_args.kwargs
    assert call_kwargs["show_llm_messages"] is True

    # Reset mock
    mock_custom_ui.reset_mock()

    # Test --show-reasoning (explicit enable)
    result = cli_runner.invoke(app, ["run", str(sample_agent_file), "test prompt", "--show-reasoning"])
    assert result.exit_code == 0
    mock_custom_ui.assert_called()
    call_kwargs = mock_custom_ui.call_args.kwargs
    assert call_kwargs["show_llm_messages"] is True

    # Reset mock
    mock_custom_ui.reset_mock()

    # Test --no-show-reasoning (disable)
    result = cli_runner.invoke(app, ["run", str(sample_agent_file), "test prompt", "--no-show-reasoning"])
    assert result.exit_code == 0
    mock_custom_ui.assert_called()
    call_kwargs = mock_custom_ui.call_args.kwargs
    assert call_kwargs["show_llm_messages"] is False


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
        result = cli_runner.invoke(app, ["run", "--no-workspace", "agent.md", "test prompt"])
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
    # Check for panel formatting (Rich library) - progress info goes to stderr
    output = result.stderr
    assert "Agent:" in output
    assert "Task:" in output
    assert "Directory:" in output
    assert "test_agent.md" in output
    assert "test prompt" in output


def test_empty_prompt(cli_runner, sample_agent_file, mock_agent_execution):
    """Test run command with empty prompt."""
    result = cli_runner.invoke(app, ["run", str(sample_agent_file), ""])
    assert result.exit_code == 0
    # Task info goes to stderr
    assert "Task:" in result.stderr


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
    """Test animation integration with CLI commands via custom_agent_ui."""

    @pytest.mark.parametrize(
        "extra_flags,expected_progress",
        [
            ([], True),  # No flags - progress enabled
            (["--non-interactive"], True),  # Non-interactive doesn't affect progress
            (["--no-color"], False),  # No color disables progress
            (["--non-interactive", "--no-color"], False),  # Both flags - no-color disables progress
        ],
    )
    @patch("tsugite.utils.should_use_plain_output", return_value=False)
    @patch("tsugite.ui.custom_agent_ui")
    def test_animation_flags(
        self,
        mock_custom_ui,
        mock_plain_output,
        cli_runner,
        sample_agent_file,
        mock_agent_execution,
        extra_flags,
        expected_progress,
    ):
        """Test that animation is enabled/disabled via show_progress based on CLI flags."""
        mock_custom_ui.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_custom_ui.return_value.__exit__ = MagicMock(return_value=None)

        cmd = ["run", str(sample_agent_file), "test prompt"] + extra_flags
        result = cli_runner.invoke(app, cmd)

        assert result.exit_code == 0
        mock_custom_ui.assert_called_once()
        call_args = mock_custom_ui.call_args
        assert call_args.kwargs["show_progress"] is expected_progress
        # Verify default UI flags are properly set
        assert call_args.kwargs["show_panels"] is False

    @patch("tsugite.utils.should_use_plain_output", return_value=False)
    @patch("tsugite.ui.custom_agent_ui")
    @patch("tsugite.agent_runner.run_agent")
    def test_animation_context_manager_usage(
        self, mock_run_agent, mock_custom_ui, mock_plain_output, cli_runner, sample_agent_file
    ):
        """Test that custom_agent_ui context manager is properly used around run_agent in default UI."""
        mock_run_agent.return_value = "Test completion"
        mock_context = MagicMock()
        mock_custom_ui.return_value = mock_context

        with patch("tsugite.md_agents.validate_agent_execution") as mock_validate:
            mock_validate.return_value = (True, "Agent is valid")

            result = cli_runner.invoke(app, ["run", str(sample_agent_file), "test prompt"])

        assert result.exit_code == 0
        # Verify context manager was used
        mock_context.__enter__.assert_called_once()
        mock_context.__exit__.assert_called_once()
        # Verify run_agent was called
        mock_run_agent.assert_called_once()

    @patch("tsugite.utils.should_use_plain_output", return_value=False)
    @patch("tsugite.ui.custom_agent_ui")
    def test_animation_with_agent_execution_error(
        self, mock_custom_ui, mock_plain_output, cli_runner, sample_agent_file
    ):
        """Test animation cleanup when agent execution fails in default UI."""
        mock_context = MagicMock()
        mock_custom_ui.return_value = mock_context

        with (
            patch("tsugite.agent_runner.run_agent") as mock_run_agent,
            patch("tsugite.md_agents.validate_agent_execution") as mock_validate,
        ):
            mock_validate.return_value = (True, "Agent is valid")
            mock_run_agent.side_effect = RuntimeError("Agent execution failed")

            result = cli_runner.invoke(app, ["run", str(sample_agent_file), "test prompt"])

        assert result.exit_code == 1
        # Verify context manager was still properly used despite error
        mock_context.__enter__.assert_called_once()
        mock_context.__exit__.assert_called_once()

    @patch("tsugite.utils.should_use_plain_output", return_value=False)
    @patch("tsugite.ui.custom_agent_ui")
    def test_animation_console_parameter(
        self, mock_custom_ui, mock_plain_output, cli_runner, sample_agent_file, mock_agent_execution
    ):
        """Test that correct console instance is passed to custom_agent_ui in default UI."""
        mock_custom_ui.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_custom_ui.return_value.__exit__ = MagicMock(return_value=None)

        result = cli_runner.invoke(app, ["run", str(sample_agent_file), "test prompt"])

        assert result.exit_code == 0
        # Verify console parameter was passed
        mock_custom_ui.assert_called_once()
        call_args = mock_custom_ui.call_args
        # Console should be passed as keyword argument
        assert "console" in call_args.kwargs
        console_arg = call_args.kwargs["console"]
        assert console_arg is not None
        assert hasattr(console_arg, "print")


class TestHeadlessMode:
    """Test headless mode for CI/scripts."""

    def test_headless_mode_basic(self, cli_runner, sample_agent_file, mock_agent_execution):
        """Test that headless mode produces clean output without UI decorations."""
        result = cli_runner.invoke(app, ["run", str(sample_agent_file), "test prompt", "--headless"])

        assert result.exit_code == 0
        # Should not contain Panel decorations
        assert "Tsugite Agent Runner" not in result.stdout
        assert "Agent:" not in result.stdout or "[cyan]Agent:[/cyan]" not in result.stdout
        assert "Starting agent execution" not in result.stdout
        # Should not contain "=" decorations
        assert "=" * 50 not in result.stdout
        assert "Agent Execution Complete" not in result.stdout
        # Should have the final result
        assert "Test agent execution completed" in result.stdout

    def test_headless_mode_no_ansi(self, cli_runner, sample_agent_file, mock_agent_execution):
        """Test that headless mode produces no ANSI color codes."""
        result = cli_runner.invoke(app, ["run", str(sample_agent_file), "test prompt", "--headless"])

        assert result.exit_code == 0
        # Check that ANSI escape codes are not present
        assert "\033[" not in result.stdout

    def test_headless_mode_no_panel_borders(self, cli_runner, sample_agent_file, mock_agent_execution):
        """Test that headless mode produces no Rich panel box drawing characters."""
        result = cli_runner.invoke(app, ["run", str(sample_agent_file), "test prompt", "--headless"])

        assert result.exit_code == 0
        # Check for Rich panel box drawing characters (should not be present)
        panel_chars = ["╭", "╮", "╰", "╯", "│", "─", "┌", "┐", "└", "┘", "├", "┤"]
        for char in panel_chars:
            assert char not in result.stdout, f"Found panel border character '{char}' in headless output"

    def test_headless_mode_with_verbose(self, cli_runner, sample_agent_file, mock_agent_execution):
        """Test that headless verbose mode still works."""
        result = cli_runner.invoke(app, ["run", str(sample_agent_file), "test prompt", "--headless", "--verbose"])

        assert result.exit_code == 0
        # Should still have clean output
        assert "Test agent execution completed" in result.stdout

    @patch("tsugite.ui.custom_agent_ui")
    def test_headless_uses_custom_ui_with_correct_flags(
        self, mock_custom_ui, cli_runner, sample_agent_file, mock_agent_execution
    ):
        """Test that headless mode uses custom_agent_ui with correct flags."""
        mock_custom_ui.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_custom_ui.return_value.__exit__ = MagicMock(return_value=None)

        result = cli_runner.invoke(app, ["run", str(sample_agent_file), "test prompt", "--headless"])

        assert result.exit_code == 0
        # Verify custom_agent_ui was called with correct flags
        mock_custom_ui.assert_called_once()
        call_kwargs = mock_custom_ui.call_args.kwargs
        assert call_kwargs["show_code"] is False
        assert call_kwargs["show_observations"] is False
        assert call_kwargs["show_progress"] is False
        assert call_kwargs["show_llm_messages"] is False
        assert call_kwargs["show_panels"] is False

    @patch("tsugite.ui.custom_agent_ui")
    def test_headless_verbose_enables_output(self, mock_custom_ui, cli_runner, sample_agent_file, mock_agent_execution):
        """Test that headless --verbose enables detailed output."""
        mock_custom_ui.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_custom_ui.return_value.__exit__ = MagicMock(return_value=None)

        result = cli_runner.invoke(app, ["run", str(sample_agent_file), "test prompt", "--headless", "--verbose"])

        assert result.exit_code == 0
        # Verify custom_agent_ui was called with verbose flags
        mock_custom_ui.assert_called_once()
        call_kwargs = mock_custom_ui.call_args.kwargs
        assert call_kwargs["show_code"] is True
        assert call_kwargs["show_observations"] is True
        assert call_kwargs["show_llm_messages"] is True
        # Progress and panels should still be disabled in headless
        assert call_kwargs["show_progress"] is False
        assert call_kwargs["show_panels"] is False

    def test_headless_validation_error(self, cli_runner, temp_dir):
        """Test that validation errors in headless mode are handled correctly."""
        # Create invalid agent (missing required fields)
        invalid_agent = temp_dir / "invalid.md"
        invalid_agent.write_text("---\n---\nInvalid agent")

        result = cli_runner.invoke(app, ["run", str(invalid_agent), "test prompt", "--headless"])

        assert result.exit_code == 1
        # In headless mode, errors go to stderr but test output captures them differently
        # Just verify the exit code is correct
        assert result.exit_code == 1


@pytest.mark.parametrize(
    "terminal_width,expected_logo",
    [
        (40, "NARROW"),  # Very narrow terminal
        (79, "NARROW"),  # Just below threshold
        (80, "WIDE"),  # At threshold
        (200, "WIDE"),  # Very wide terminal
    ],
)
def test_logo_selection(terminal_width, expected_logo):
    """Test that correct logo is selected based on terminal width."""
    from rich.console import Console

    from tsugite.cli.helpers import get_logo
    from tsugite.constants import TSUGITE_LOGO_NARROW, TSUGITE_LOGO_WIDE

    mock_console = MagicMock(spec=Console)
    mock_console.width = terminal_width

    result = get_logo(mock_console)
    expected = TSUGITE_LOGO_NARROW if expected_logo == "NARROW" else TSUGITE_LOGO_WIDE
    assert result == expected


class TestAutoDiscovery:
    """Test auto-discovery feature where CLI defaults to default."""

    def test_run_without_agent_defaults_to_builtin(self, cli_runner, temp_dir):
        """Test that running without an agent defaults to default."""
        from tsugite.cli.helpers import parse_cli_arguments

        # Test that parse_cli_arguments defaults to default
        agents, prompt, stdin_attachment = parse_cli_arguments(["test", "task"], check_stdin=False)

        assert agents == ["+default"]
        assert prompt == "test task"
        assert stdin_attachment is None

    def test_run_explicit_agent_overrides_default(self, cli_runner, sample_agent_file, mock_agent_execution):
        """Test that explicitly specifying an agent still works."""
        result = cli_runner.invoke(app, ["run", str(sample_agent_file), "test prompt"])

        assert result.exit_code == 0
        # Should use the specified agent, not default (agent name goes to stderr)
        assert "test_agent.md" in result.stderr

    @patch("tsugite.agent_runner.run_agent")
    @patch("tsugite.md_agents.validate_agent_execution")
    def test_builtin_default_agent_execution(self, mock_validate, mock_run, cli_runner):
        """Test that default agent can be executed."""
        mock_validate.return_value = (True, "Agent is valid")
        mock_run.return_value = "Task completed"

        # Run without specifying an agent
        result = cli_runner.invoke(app, ["run", "What is 2+2?"])

        # Should not fail due to missing agent file
        assert "Agent file not found" not in result.stdout
        # May fail on validation or execution, but not on path issues
        # The actual execution depends on whether the API key is set

    @patch("tsugite.agent_runner.run_agent")
    @patch("tsugite.md_agents.validate_agent_execution")
    def test_auto_discovery_with_available_agents(self, mock_validate, mock_run, cli_runner, tmp_path, monkeypatch):
        """Test auto-discovery when agents are available."""
        monkeypatch.chdir(tmp_path)

        # Create a test agent
        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()
        test_agent = agents_dir / "helper.md"
        test_agent.write_text("""---
name: helper
description: Helps with tasks
---
Content
""")

        mock_validate.return_value = (True, "Agent is valid")
        mock_run.return_value = "Task completed"

        # The default should be able to discover the helper agent
        # via its prefetch mechanism
        cli_runner.invoke(app, ["run", "help me with something"])

        # Should execute successfully (verified by no exception)
        # The actual delegation to helper depends on LLM decision


class TestRunCommandHistory:
    """Tests for history integration with run command."""

    def test_run_command_saves_history_by_default(self, cli_runner, sample_agent_file, tmp_path, monkeypatch):
        """Test that history is saved by default when running an agent."""
        # Mock history functions
        with (
            patch("tsugite.agent_runner.run_agent") as mock_run_agent,
            patch("tsugite.md_agents.validate_agent_execution") as mock_validate,
            patch("tsugite.agent_runner.history_integration.save_run_to_history") as mock_save_history,
        ):
            # Mock run_agent to return AgentExecutionResult model
            from tests.conftest import mock_agent_execution_result

            mock_run_agent.return_value = mock_agent_execution_result(
                response="Test result",
                token_count=1000,
                cost=0.05,
                step_count=3,
            )
            mock_validate.return_value = (True, "Valid")
            mock_save_history.return_value = "test_conv_id"

            result = cli_runner.invoke(app, ["run", str(sample_agent_file), "test prompt"])

            assert result.exit_code == 0

            # Verify save_run_to_history was called
            mock_save_history.assert_called_once()
            call_kwargs = mock_save_history.call_args[1]
            assert call_kwargs["prompt"] == "test prompt"
            assert call_kwargs["result"] == "Test result"
            assert call_kwargs["token_count"] == 1000
            assert call_kwargs["cost"] == 0.05

    def test_run_command_no_history_flag(self, cli_runner, sample_agent_file, tmp_path):
        """Test that --no-history flag prevents saving to history."""
        with (
            patch("tsugite.agent_runner.run_agent") as mock_run_agent,
            patch("tsugite.md_agents.validate_agent_execution") as mock_validate,
            patch("tsugite.agent_runner.history_integration.save_run_to_history") as mock_save_history,
        ):
            mock_run_agent.return_value = "Test result"
            mock_validate.return_value = (True, "Valid")

            result = cli_runner.invoke(app, ["run", str(sample_agent_file), "test prompt", "--no-history"])

            assert result.exit_code == 0

            # Verify save_run_to_history was NOT called
            mock_save_history.assert_not_called()

    def test_run_command_history_with_metadata(self, cli_runner, sample_agent_file):
        """Test that token count and cost are passed to history."""
        from tests.conftest import mock_agent_execution_result

        with (
            patch("tsugite.agent_runner.run_agent") as mock_run_agent,
            patch("tsugite.md_agents.validate_agent_execution") as mock_validate,
            patch("tsugite.agent_runner.history_integration.save_run_to_history") as mock_save_history,
        ):
            mock_run_agent.return_value = mock_agent_execution_result(
                response="Result",
                token_count=2500,
                cost=0.12,
                execution_steps=[],
                system_message="System prompt",
                attachments=[],
            )
            mock_validate.return_value = (True, "Valid")
            mock_save_history.return_value = "conv_123"

            result = cli_runner.invoke(app, ["run", str(sample_agent_file), "complex task"])

            assert result.exit_code == 0

            # Verify metadata was passed
            call_kwargs = mock_save_history.call_args[1]
            assert call_kwargs["token_count"] == 2500
            assert call_kwargs["cost"] == 0.12
            assert call_kwargs["execution_steps"] == []

    def test_run_command_history_conversation_created(self, cli_runner, sample_agent_file, tmp_path):
        """Test that conversation file is created after run."""
        from tests.conftest import mock_agent_execution_result

        with (
            patch("tsugite.history.storage.get_history_dir", return_value=tmp_path),
            patch("tsugite.history.storage.get_machine_name", return_value="test_machine"),
            patch("tsugite.agent_runner.history_integration.get_history_dir", return_value=tmp_path),
            patch("tsugite.agent_runner.run_agent") as mock_run_agent,
            patch("tsugite.md_agents.validate_agent_execution") as mock_validate,
            patch("tsugite.config.load_config") as mock_config,
            patch("tsugite.md_agents.parse_agent_file") as mock_parse,
        ):
            from tsugite.history import SessionStorage, Turn

            mock_run_agent.return_value = mock_agent_execution_result(
                response="Result",
                token_count=100,
                cost=0.01,
                execution_steps=[],
                system_message="System prompt",
                attachments=[],
            )
            mock_validate.return_value = (True, "Valid")
            mock_config.return_value = MagicMock(history_enabled=True)

            mock_agent = MagicMock()
            mock_agent.config = MagicMock(disable_history=False, name="test_agent")
            mock_parse.return_value = mock_agent

            result = cli_runner.invoke(app, ["run", str(sample_agent_file), "task"])

            assert result.exit_code == 0

            # Verify conversation file was created
            conv_files = list(tmp_path.glob("*.jsonl"))
            assert len(conv_files) > 0

            # Verify it can be loaded using V2 API
            storage = SessionStorage.load(conv_files[0])
            records = storage.load_records()
            turns = [r for r in records if isinstance(r, Turn)]
            assert len(turns) == 1  # One turn

    def test_run_command_history_error_handling(self, cli_runner, sample_agent_file, capsys):
        """Test that history errors don't crash the run."""
        from tests.conftest import mock_agent_execution_result

        with (
            patch("tsugite.agent_runner.run_agent") as mock_run_agent,
            patch("tsugite.md_agents.validate_agent_execution") as mock_validate,
            patch("tsugite.agent_runner.history_integration.save_run_to_history") as mock_save_history,
        ):
            mock_run_agent.return_value = mock_agent_execution_result(
                response="Result",
                token_count=100,
                cost=0.01,
                execution_steps=[],
                system_message="System prompt",
                attachments=[],
            )
            mock_validate.return_value = (True, "Valid")

            # save_run_to_history raises exception
            mock_save_history.side_effect = Exception("Database error")

            result = cli_runner.invoke(app, ["run", str(sample_agent_file), "task"])

            # Run should still succeed
            assert result.exit_code == 0
            assert "Result" in result.stdout

    def test_run_command_multistep_history_saves(self, cli_runner, tmp_path):
        """Test that multi-step agents save to history (without metadata)."""
        # Create a multi-step agent
        multistep_agent = tmp_path / "multistep.md"
        multistep_agent.write_text("""---
name: multistep_test
model: openai:gpt-4o-mini
tools: []
---
# Multi-step agent

<!-- tsu:step name="step1" -->
Step 1

<!-- tsu:step name="step2" -->
Step 2
""")

        with (
            patch("tsugite.agent_runner.run_multistep_agent") as mock_run_multistep,
            patch("tsugite.md_agents.validate_agent_execution") as mock_validate,
            patch("tsugite.agent_runner.history_integration.save_run_to_history") as mock_save_history,
        ):
            # Multi-step returns just a string, not a tuple
            mock_run_multistep.return_value = "Multi-step complete"
            mock_validate.return_value = (True, "Valid")
            mock_save_history.return_value = "conv_456"

            result = cli_runner.invoke(app, ["run", str(multistep_agent), "run steps"])

            assert result.exit_code == 0

            # Verify save_run_to_history was called (even without metadata)
            mock_save_history.assert_called_once()
            call_kwargs = mock_save_history.call_args[1]
            assert call_kwargs["result"] == "Multi-step complete"
            # Token count and cost should be None for multi-step
            assert call_kwargs["token_count"] is None
            assert call_kwargs["cost"] is None


class TestUnpackExecutionResult:
    """Tests for _unpack_execution_result function."""

    def test_unpack_agent_execution_result(self):
        """Test unpacking AgentExecutionResult object."""
        from tests.conftest import mock_agent_execution_result
        from tsugite.agent_runner import run_agent
        from tsugite.cli import _unpack_execution_result

        # Create an AgentExecutionResult
        result = mock_agent_execution_result(
            response="Test response",
            token_count=100,
            cost=0.005,
            step_count=3,
            execution_steps=[{"step": 1}],
            system_message="System prompt",
            attachments=[("file.txt", "content")],
        )

        # Unpack it
        unpacked = _unpack_execution_result(result, should_save_history=True, executor=run_agent)

        # Verify all fields are extracted correctly
        assert unpacked[0] == "Test response"
        assert unpacked[1] == 100
        assert unpacked[2] == 0.005
        assert unpacked[3] == [{"step": 1}]
        assert unpacked[4] == "System prompt"
        assert unpacked[5] == [("file.txt", "content")]

    def test_unpack_plain_string(self):
        """Test unpacking plain string result."""
        from tsugite.agent_runner import run_agent
        from tsugite.cli import _unpack_execution_result

        result = "Simple string response"
        unpacked = _unpack_execution_result(result, should_save_history=False, executor=run_agent)

        # Should return string + None for all other fields
        assert unpacked[0] == "Simple string response"
        assert unpacked[1] is None
        assert unpacked[2] is None
        assert unpacked[3] is None
        assert unpacked[4] is None
        assert unpacked[5] is None
