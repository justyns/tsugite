"""Tests for REPL command handlers."""

from io import StringIO
from unittest.mock import MagicMock, patch

from rich.console import Console

from tsugite.ui.repl_commands import (
    handle_attach,
    handle_clear,
    handle_detach,
    handle_help,
    handle_history,
    handle_list_attachments,
    handle_save,
    handle_stats,
    handle_stream,
    handle_tools,
    handle_verbose,
    parse_command,
)


def _capture_console():
    return Console(file=StringIO(), width=120, color_system=None)


def test_parse_command_simple():
    """Test parsing simple command."""
    command, args, error = parse_command("/help")
    assert command == "/help"
    assert args == []
    assert error is None


def test_parse_command_with_args():
    """Test parsing command with arguments."""
    command, args, error = parse_command("/attach test.txt")
    assert command == "/attach"
    assert args == ["test.txt"]
    assert error is None


def test_handle_help():
    """Test help command handler."""
    console = _capture_console()
    handle_help(console)
    out = console.file.getvalue()
    assert "Available Commands" in out
    assert "/attach <path>" in out
    assert "Type your message to chat with the agent." in out


def test_handle_clear():
    """Test clear command handler."""
    console = Console()
    with patch.object(console, "clear") as mock_clear:
        handle_clear(console)
        mock_clear.assert_called_once()


def test_handle_stats():
    """Test stats command handler."""
    console = _capture_console()

    class StatsHolder:
        turn_count = 5
        total_tokens = 1000
        total_cost = 0.05

    handle_stats(console, StatsHolder())
    out = console.file.getvalue()
    assert "Session Statistics" in out
    assert "1,000" in out
    assert "$0.050000" in out


def test_handle_history():
    """Test history command handler."""
    console = Console()

    # Test with no results
    with patch("tsugite.ui.repl_commands.get_history_backend") as mock_backend:
        mock_backend.return_value.list_sessions.return_value = []
        with patch.object(console, "print") as mock_print:
            handle_history(console, limit=10)
            # Should indicate no history found
            assert mock_print.called
            call_args = str(mock_print.call_args)
            assert "no" in call_args.lower() or "history" in call_args.lower()


def test_handle_attach_file_not_found():
    """Test attach with non-existent file."""
    console = Console()
    manager = MagicMock()

    with patch.object(console, "print") as mock_print:
        handle_attach(console, "/nonexistent/file.txt", manager)
        # Should print error
        assert mock_print.called
        call_args = str(mock_print.call_args)
        assert "not found" in call_args.lower()


def test_handle_attach_success(tmp_path):
    """Test successful file attachment."""
    # Create test file
    test_file = tmp_path / "test.txt"
    test_file.write_text("test content")

    console = Console()
    manager = MagicMock()
    manager.attachments = {}

    with patch.object(console, "print") as mock_print:
        handle_attach(console, str(test_file), manager)
        # Should print success message
        assert mock_print.called
        call_args = str(mock_print.call_args)
        assert "attached" in call_args.lower()

    # Check attachment was added
    assert str(test_file) in manager.attachments


def test_handle_detach_no_attachments():
    """Test detach with no attachments."""
    console = _capture_console()
    manager = MagicMock()
    manager.attachments = {}

    handle_detach(console, "test.txt", manager)
    assert "No attachments to remove." in console.file.getvalue()


def test_handle_detach_success():
    """Test successful detachment."""
    console = Console()
    manager = MagicMock()
    manager.attachments = {"test.txt": "content"}

    with patch.object(console, "print") as mock_print:
        handle_detach(console, "test.txt", manager)
        # Should print success message
        assert mock_print.called

    # Check attachment was removed
    assert "test.txt" not in manager.attachments


def test_handle_list_attachments_empty():
    """Test listing attachments when none exist."""
    console = _capture_console()
    manager = MagicMock()
    manager.attachments = {}

    handle_list_attachments(console, manager)
    assert "No attachments." in console.file.getvalue()


def test_handle_list_attachments():
    """Test listing attachments."""
    console = _capture_console()
    manager = MagicMock()
    manager.attachments = {"file1.txt": "content1", "file2.txt": "x" * 2048}

    handle_list_attachments(console, manager)
    out = console.file.getvalue()
    assert "file1.txt" in out and "8B" in out
    assert "file2.txt" in out and "2.0KB" in out


def test_handle_save_no_conversation():
    """Test save with no active conversation."""
    console = _capture_console()
    manager = MagicMock()
    manager.conversation_id = None

    handle_save(console, "output.md", manager)
    assert "No active conversation to save." in console.file.getvalue()


def test_handle_tools_no_tools():
    """Test tools command with no tools."""
    console = _capture_console()
    manager = MagicMock()
    manager.available_tools = []

    handle_tools(console, manager)
    assert "No tools available for this agent." in console.file.getvalue()


def test_handle_stream_on():
    """Test enabling streaming."""
    console = Console()
    manager = MagicMock()
    manager.stream_enabled = False

    with patch.object(console, "print") as mock_print:
        handle_stream(console, "on", manager)
        assert manager.stream_enabled is True
        assert mock_print.called


def test_handle_stream_off():
    """Test disabling streaming."""
    console = Console()
    manager = MagicMock()
    manager.stream_enabled = True

    with patch.object(console, "print") as mock_print:
        handle_stream(console, "off", manager)
        assert manager.stream_enabled is False
        assert mock_print.called


def test_handle_stream_toggle():
    """Test toggling streaming."""
    console = Console()
    manager = MagicMock()
    manager.stream_enabled = False

    with patch.object(console, "print") as mock_print:
        handle_stream(console, None, manager)
        assert manager.stream_enabled is True
        assert mock_print.called


def test_handle_verbose_on():
    """Test enabling verbose mode."""
    console = Console()
    ui_handler = MagicMock()
    ui_handler.show_observations = False

    with patch.object(console, "print") as mock_print:
        handle_verbose(console, "on", ui_handler)
        assert ui_handler.show_observations is True
        assert mock_print.called


def test_handle_verbose_off():
    """Test disabling verbose mode."""
    console = Console()
    ui_handler = MagicMock()
    ui_handler.show_observations = True

    with patch.object(console, "print") as mock_print:
        handle_verbose(console, "off", ui_handler)
        assert ui_handler.show_observations is False
        assert mock_print.called


def test_handle_verbose_toggle():
    """Test toggling verbose mode."""
    console = Console()
    ui_handler = MagicMock()
    ui_handler.show_observations = False

    with patch.object(console, "print") as mock_print:
        handle_verbose(console, None, ui_handler)
        assert ui_handler.show_observations is True
        assert mock_print.called
