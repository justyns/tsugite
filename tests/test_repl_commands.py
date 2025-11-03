"""Tests for REPL command handlers."""

from unittest.mock import MagicMock, patch

from rich.console import Console

from tsugite.ui.repl_commands import (
    handle_attach,
    handle_clear,
    handle_detach,
    handle_help,
    handle_history,
    handle_list_attachments,
    handle_multiline,
    handle_save,
    handle_stats,
    handle_stream,
    handle_tools,
    handle_verbose,
    parse_command,
)


def test_parse_command_simple():
    """Test parsing simple command."""
    command, args = parse_command("/help")
    assert command == "/help"
    assert args == []


def test_parse_command_with_args():
    """Test parsing command with arguments."""
    command, args = parse_command("/attach test.txt")
    assert command == "/attach"
    assert args == ["test.txt"]


def test_parse_command_multiple_args():
    """Test parsing command with multiple arguments."""
    command, args = parse_command("/history 20")
    assert command == "/history"
    assert args == ["20"]


def test_handle_help():
    """Test help command handler."""
    console = Console()
    with patch.object(console, "print") as mock_print:
        handle_help(console)
        # Should print help table
        assert mock_print.called


def test_handle_clear():
    """Test clear command handler."""
    console = Console()
    with patch.object(console, "clear") as mock_clear:
        handle_clear(console)
        mock_clear.assert_called_once()


def test_handle_stats():
    """Test stats command handler."""
    console = Console()

    # Create a simple class to hold stats
    class StatsHolder:
        turn_count = 5
        total_tokens = 1000
        total_cost = 0.05

    manager = StatsHolder()

    with patch.object(console, "print") as mock_print:
        handle_stats(console, manager)
        # Should print stats table
        assert mock_print.called


def test_handle_history():
    """Test history command handler."""
    console = Console()

    # Test with no results
    with patch("tsugite.ui.repl_commands.query_index", return_value=[]):
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
    console = Console()
    manager = MagicMock()
    manager.attachments = {}

    with patch.object(console, "print") as mock_print:
        handle_detach(console, "test.txt", manager)
        # Should indicate no attachments
        assert mock_print.called


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
    console = Console()
    manager = MagicMock()
    manager.attachments = {}

    with patch.object(console, "print") as mock_print:
        handle_list_attachments(console, manager)
        # Should indicate no attachments
        assert mock_print.called


def test_handle_list_attachments():
    """Test listing attachments."""
    console = Console()
    manager = MagicMock()
    manager.attachments = {"file1.txt": "content1", "file2.txt": "content2"}

    with patch.object(console, "print") as mock_print:
        handle_list_attachments(console, manager)
        # Should print table
        assert mock_print.called


def test_handle_save_no_conversation():
    """Test save with no active conversation."""
    console = Console()
    manager = MagicMock()
    manager.conversation_id = None

    with patch.object(console, "print") as mock_print:
        handle_save(console, "output.md", manager)
        # Should indicate no conversation
        assert mock_print.called


def test_handle_tools_no_tools():
    """Test tools command with no tools."""
    console = Console()
    manager = MagicMock()
    manager.available_tools = []

    with patch.object(console, "print") as mock_print:
        handle_tools(console, manager)
        # Should indicate no tools
        assert mock_print.called


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


def test_handle_multiline():
    """Test multiline command."""
    console = Console()

    with patch.object(console, "print") as mock_print:
        result = handle_multiline(console, "on")
        # Currently returns True/False based on value
        assert isinstance(result, bool)
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
