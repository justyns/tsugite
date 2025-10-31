"""Tests for stdin support in tsugite."""

import io
from unittest.mock import patch

import pytest

from tsugite.cli.helpers import STDIN_ATTACHMENT_NAME, parse_cli_arguments
from tsugite.utils import has_stdin_data, read_stdin


class TestStdinDetection:
    """Tests for stdin detection utilities."""

    @patch("sys.stdin.isatty")
    def test_has_stdin_data_interactive_terminal(self, mock_isatty):
        """Test that interactive terminals return False."""
        mock_isatty.return_value = True
        assert has_stdin_data() is False

    @patch("sys.stdin.isatty")
    @patch("select.select")
    def test_has_stdin_data_with_data(self, mock_select, mock_isatty):
        """Test detection when stdin has data."""
        mock_isatty.return_value = False
        mock_select.return_value = ([True], [], [])
        assert has_stdin_data() is True

    @patch("sys.stdin.isatty")
    @patch("select.select")
    def test_has_stdin_data_no_data(self, mock_select, mock_isatty):
        """Test detection when stdin has no data."""
        mock_isatty.return_value = False
        mock_select.return_value = ([], [], [])
        assert has_stdin_data() is False

    @patch("sys.stdin", new_callable=io.StringIO)
    def test_read_stdin(self, mock_stdin):
        """Test reading from stdin."""
        test_data = "test stdin content\nline 2\n"
        mock_stdin.write(test_data)
        mock_stdin.seek(0)

        result = read_stdin()
        assert result == test_data


class TestParseCliArgumentsWithStdin:
    """Tests for parse_cli_arguments with stdin support."""

    @patch("sys.stdin.isatty")
    @patch("select.select")
    @patch("sys.stdin.read")
    def test_stdin_with_prompt(self, mock_read, mock_select, mock_isatty):
        """Test stdin is detected and read when prompt is provided."""
        mock_isatty.return_value = False
        mock_select.return_value = ([True], [], [])
        mock_read.return_value = "stdin content"

        agents, prompt, stdin_attachment = parse_cli_arguments(["analyze this"])

        assert agents == ["+default"]
        assert prompt == "analyze this"
        assert stdin_attachment == (STDIN_ATTACHMENT_NAME, "stdin content")

    @patch("sys.stdin.isatty")
    def test_no_stdin_interactive(self, mock_isatty):
        """Test no stdin attachment in interactive terminal."""
        mock_isatty.return_value = True

        agents, prompt, stdin_attachment = parse_cli_arguments(["test prompt"])

        assert agents == ["+default"]
        assert prompt == "test prompt"
        assert stdin_attachment is None

    @patch("sys.stdin.isatty")
    @patch("select.select")
    @patch("sys.stdin.read")
    def test_stdin_empty_content(self, mock_read, mock_select, mock_isatty):
        """Test stdin with only whitespace is ignored."""
        mock_isatty.return_value = False
        mock_select.return_value = ([True], [], [])
        mock_read.return_value = "   \n  \t  \n"

        agents, prompt, stdin_attachment = parse_cli_arguments(["test"])

        assert stdin_attachment is None

    @patch("sys.stdin.isatty")
    @patch("select.select")
    @patch("sys.stdin.read")
    def test_stdin_with_agent_ref(self, mock_read, mock_select, mock_isatty):
        """Test stdin works with explicit agent reference."""
        mock_isatty.return_value = False
        mock_select.return_value = ([True], [], [])
        mock_read.return_value = "error log data"

        agents, prompt, stdin_attachment = parse_cli_arguments(["+debugger", "analyze error"])

        assert agents == ["+debugger"]
        assert prompt == "analyze error"
        assert stdin_attachment == (STDIN_ATTACHMENT_NAME, "error log data")

    def test_stdin_disabled_when_check_stdin_false(self):
        """Test stdin is not checked when check_stdin=False."""
        agents, prompt, stdin_attachment = parse_cli_arguments(["test"], check_stdin=False)

        assert stdin_attachment is None

    @patch("sys.stdin.isatty")
    @patch("select.select")
    @patch("sys.stdin.read")
    def test_stdin_multiline_content(self, mock_read, mock_select, mock_isatty):
        """Test stdin with multiline content."""
        mock_isatty.return_value = False
        mock_select.return_value = ([True], [], [])
        multiline = "line 1\nline 2\nline 3\n"
        mock_read.return_value = multiline

        agents, prompt, stdin_attachment = parse_cli_arguments(["process this"])

        assert stdin_attachment == (STDIN_ATTACHMENT_NAME, multiline)


class TestStdinIntegration:
    """Integration tests for stdin with attachment system."""

    @pytest.fixture
    def temp_dir(self, tmp_path):
        """Create temporary directory for tests."""
        return tmp_path

    @patch("sys.stdin.isatty")
    @patch("select.select")
    @patch("sys.stdin.read")
    def test_stdin_in_assemble_prompt(self, mock_read, mock_select, mock_isatty, temp_dir):
        """Test stdin is properly included in assembled prompt."""
        from rich.console import Console

        from tsugite.cli.helpers import assemble_prompt_with_attachments

        mock_isatty.return_value = False
        mock_select.return_value = ([True], [], [])
        mock_read.return_value = "stdin data"

        stdin_attachment = (STDIN_ATTACHMENT_NAME, "stdin data")
        console = Console()

        prompt, attachments = assemble_prompt_with_attachments(
            prompt="analyze this",
            agent_attachments=None,
            cli_attachments=None,
            base_dir=temp_dir,
            refresh_cache=False,
            console=console,
            stdin_attachment=stdin_attachment,
        )

        assert len(attachments) == 1
        assert attachments[0] == (STDIN_ATTACHMENT_NAME, "stdin data")

    @patch("sys.stdin.isatty")
    @patch("select.select")
    @patch("sys.stdin.read")
    def test_stdin_with_other_attachments(self, mock_read, mock_select, mock_isatty, temp_dir):
        """Test stdin is added after other attachments."""
        from rich.console import Console

        from tsugite.cli.helpers import assemble_prompt_with_attachments

        mock_isatty.return_value = False
        mock_select.return_value = ([True], [], [])
        mock_read.return_value = "stdin content"

        # Create test file
        test_file = temp_dir / "test.txt"
        test_file.write_text("file content")

        stdin_attachment = (STDIN_ATTACHMENT_NAME, "stdin content")
        console = Console()

        prompt, attachments = assemble_prompt_with_attachments(
            prompt=f"analyze @{test_file}",
            agent_attachments=None,
            cli_attachments=None,
            base_dir=temp_dir,
            refresh_cache=False,
            console=console,
            stdin_attachment=stdin_attachment,
        )

        assert len(attachments) == 2
        assert attachments[0][0] == str(test_file)
        assert attachments[1] == (STDIN_ATTACHMENT_NAME, "stdin content")


class TestStdinUseCases:
    """Test real-world stdin use cases."""

    @patch("sys.stdin.isatty")
    @patch("select.select")
    @patch("sys.stdin.read")
    def test_pipe_error_log(self, mock_read, mock_select, mock_isatty):
        """Test: cat error.log | tsugite run +debugger 'analyze'"""
        mock_isatty.return_value = False
        mock_select.return_value = ([True], [], [])
        mock_read.return_value = "[ERROR] Connection refused\n[ERROR] Timeout"

        agents, prompt, stdin_attachment = parse_cli_arguments(["+debugger", "analyze error"])

        assert agents == ["+debugger"]
        assert prompt == "analyze error"
        assert stdin_attachment[0] == STDIN_ATTACHMENT_NAME
        assert "Connection refused" in stdin_attachment[1]

    @patch("sys.stdin.isatty")
    @patch("select.select")
    @patch("sys.stdin.read")
    def test_pipe_command_output(self, mock_read, mock_select, mock_isatty):
        """Test: docker ps | tsugite run 'explain what's running'"""
        mock_isatty.return_value = False
        mock_select.return_value = ([True], [], [])
        mock_read.return_value = "CONTAINER ID   IMAGE     COMMAND\nabc123         nginx     '/nginx'"

        agents, prompt, stdin_attachment = parse_cli_arguments(["explain what's running"])

        assert stdin_attachment[0] == STDIN_ATTACHMENT_NAME
        assert "nginx" in stdin_attachment[1]

    @patch("sys.stdin.isatty")
    @patch("select.select")
    @patch("sys.stdin.read")
    def test_pipe_git_diff(self, mock_read, mock_select, mock_isatty):
        """Test: git diff | tsugite run +reviewer 'review changes'"""
        mock_isatty.return_value = False
        mock_select.return_value = ([True], [], [])
        mock_read.return_value = "diff --git a/file.py\n+added line\n-removed line"

        agents, prompt, stdin_attachment = parse_cli_arguments(["+reviewer", "review changes"])

        assert stdin_attachment[0] == STDIN_ATTACHMENT_NAME
        assert "diff --git" in stdin_attachment[1]
