"""Tests for stdin support in tsugite."""

import io
from unittest.mock import patch

import pytest

from tsugite.attachments.base import Attachment, AttachmentContentType
from tsugite.cli.helpers import STDIN_ATTACHMENT_NAME, parse_cli_arguments
from tsugite.utils import has_stdin_data, read_stdin


@pytest.fixture
def mock_stdin_pipe():
    """Mock stdin as a pipe with data available."""

    def _make(content="stdin content"):
        return {
            "isatty": patch("sys.stdin.isatty", return_value=False),
            "select": patch("select.select", return_value=([True], [], [])),
            "read": patch("sys.stdin.read", return_value=content),
        }

    return _make


class TestStdinDetection:
    """Tests for stdin detection utilities."""

    def test_has_stdin_data_interactive_returns_false(self):
        """Test that interactive terminals return False."""
        with patch("sys.stdin.isatty", return_value=True):
            assert has_stdin_data() is False

    def test_has_stdin_data_pipe_with_data(self):
        """Test detection when stdin has data."""
        with patch("sys.stdin.isatty", return_value=False), patch("select.select", return_value=([True], [], [])):
            assert has_stdin_data() is True

    def test_has_stdin_data_pipe_without_data(self):
        """Test detection when stdin has no data."""
        with patch("sys.stdin.isatty", return_value=False), patch("select.select", return_value=([], [], [])):
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

    def test_stdin_with_prompt(self, mock_stdin_pipe):
        """Test stdin is detected and read when prompt is provided."""
        mocks = mock_stdin_pipe("stdin content")
        with mocks["isatty"], mocks["select"], mocks["read"]:
            agents, prompt, stdin_attachment = parse_cli_arguments(["analyze this"])

        assert agents == ["+default"]
        assert prompt == "analyze this"
        assert isinstance(stdin_attachment, Attachment)
        assert stdin_attachment.name == STDIN_ATTACHMENT_NAME
        assert stdin_attachment.content == "stdin content"

    def test_no_stdin_interactive(self):
        """Test no stdin attachment in interactive terminal."""
        with patch("sys.stdin.isatty", return_value=True):
            agents, prompt, stdin_attachment = parse_cli_arguments(["test prompt"])

        assert agents == ["+default"]
        assert prompt == "test prompt"
        assert stdin_attachment is None

    def test_stdin_empty_content(self, mock_stdin_pipe):
        """Test stdin with only whitespace is ignored."""
        mocks = mock_stdin_pipe("   \n  \t  \n")
        with mocks["isatty"], mocks["select"], mocks["read"]:
            agents, prompt, stdin_attachment = parse_cli_arguments(["test"])

        assert stdin_attachment is None

    def test_stdin_with_agent_ref(self, mock_stdin_pipe):
        """Test stdin works with explicit agent reference."""
        mocks = mock_stdin_pipe("error log data")
        with mocks["isatty"], mocks["select"], mocks["read"]:
            agents, prompt, stdin_attachment = parse_cli_arguments(["+debugger", "analyze error"])

        assert agents == ["+debugger"]
        assert prompt == "analyze error"
        assert isinstance(stdin_attachment, Attachment)
        assert stdin_attachment.content == "error log data"

    def test_stdin_disabled_when_check_stdin_false(self):
        """Test stdin is not checked when check_stdin=False."""
        agents, prompt, stdin_attachment = parse_cli_arguments(["test"], check_stdin=False)
        assert stdin_attachment is None

    def test_stdin_multiline_content(self, mock_stdin_pipe):
        """Test stdin with multiline content."""
        multiline = "line 1\nline 2\nline 3\n"
        mocks = mock_stdin_pipe(multiline)
        with mocks["isatty"], mocks["select"], mocks["read"]:
            agents, prompt, stdin_attachment = parse_cli_arguments(["process this"])

        assert isinstance(stdin_attachment, Attachment)
        assert stdin_attachment.content == multiline


class TestStdinIntegration:
    """Integration tests for stdin with attachment system."""

    def test_stdin_in_assemble_prompt(self, tmp_path):
        """Test stdin is properly included in assembled prompt."""
        from rich.console import Console

        from tsugite.cli.helpers import assemble_prompt_with_attachments

        stdin_attachment = Attachment(
            name=STDIN_ATTACHMENT_NAME,
            content="stdin data",
            content_type=AttachmentContentType.TEXT,
            mime_type="text/plain",
        )

        prompt, attachments = assemble_prompt_with_attachments(
            prompt="analyze this",
            agent_attachments=None,
            cli_attachments=None,
            base_dir=tmp_path,
            refresh_cache=False,
            console=Console(),
            stdin_attachment=stdin_attachment,
        )

        assert len(attachments) == 1
        assert attachments[0].name == STDIN_ATTACHMENT_NAME
        assert attachments[0].content == "stdin data"

    def test_stdin_with_other_attachments(self, tmp_path):
        """Test stdin is added after other attachments."""
        from rich.console import Console

        from tsugite.cli.helpers import assemble_prompt_with_attachments

        test_file = tmp_path / "test.txt"
        test_file.write_text("file content")

        stdin_attachment = Attachment(
            name=STDIN_ATTACHMENT_NAME,
            content="stdin content",
            content_type=AttachmentContentType.TEXT,
            mime_type="text/plain",
        )

        prompt, attachments = assemble_prompt_with_attachments(
            prompt=f"analyze @{test_file}",
            agent_attachments=None,
            cli_attachments=None,
            base_dir=tmp_path,
            refresh_cache=False,
            console=Console(),
            stdin_attachment=stdin_attachment,
        )

        assert len(attachments) == 2
        assert attachments[0].name == str(test_file)
        assert attachments[1].name == STDIN_ATTACHMENT_NAME
        assert attachments[1].content == "stdin content"


class TestStdinUseCases:
    """Test real-world stdin use cases."""

    def test_pipe_error_log(self, mock_stdin_pipe):
        """Test: cat error.log | tsugite run +debugger 'analyze'"""
        mocks = mock_stdin_pipe("[ERROR] Connection refused\n[ERROR] Timeout")
        with mocks["isatty"], mocks["select"], mocks["read"]:
            agents, prompt, stdin_attachment = parse_cli_arguments(["+debugger", "analyze error"])

        assert agents == ["+debugger"]
        assert prompt == "analyze error"
        assert "Connection refused" in stdin_attachment.content

    def test_pipe_command_output(self, mock_stdin_pipe):
        """Test: docker ps | tsugite run 'explain what's running'"""
        mocks = mock_stdin_pipe("CONTAINER ID   IMAGE     COMMAND\nabc123         nginx     '/nginx'")
        with mocks["isatty"], mocks["select"], mocks["read"]:
            agents, prompt, stdin_attachment = parse_cli_arguments(["explain what's running"])

        assert "nginx" in stdin_attachment.content

    def test_pipe_git_diff(self, mock_stdin_pipe):
        """Test: git diff | tsugite run +reviewer 'review changes'"""
        mocks = mock_stdin_pipe("diff --git a/file.py\n+added line\n-removed line")
        with mocks["isatty"], mocks["select"], mocks["read"]:
            agents, prompt, stdin_attachment = parse_cli_arguments(["+reviewer", "review changes"])

        assert "diff --git" in stdin_attachment.content
