"""Tests for REPL tab completion."""

from unittest.mock import MagicMock, patch

from prompt_toolkit.document import Document

from tsugite.ui.repl_completer import TsugiteCompleter


def test_completer_slash_commands():
    """Test command completion."""
    completer = TsugiteCompleter()
    document = Document("/h", cursor_position=2)

    completions = list(completer.get_completions(document, MagicMock()))
    completion_texts = [c.text for c in completions]

    assert "/help" in completion_texts
    assert "/history" in completion_texts


def test_completer_exact_command():
    """Test exact command match."""
    completer = TsugiteCompleter()
    document = Document("/help", cursor_position=5)

    completions = list(completer.get_completions(document, MagicMock()))
    completion_texts = [c.text for c in completions]

    assert "/help" in completion_texts


def test_completer_stream_arguments():
    """Test stream command argument completion."""
    completer = TsugiteCompleter()
    document = Document("/stream o", cursor_position=9)

    completions = list(completer.get_completions(document, MagicMock()))
    completion_texts = [c.text for c in completions]

    assert "on" in completion_texts
    assert "off" in completion_texts


def test_completer_agent_names():
    """Test agent name completion."""
    with patch.object(TsugiteCompleter, "_discover_agents", return_value=["chat-assistant", "code-helper"]):
        completer = TsugiteCompleter()
        document = Document("/agent c", cursor_position=8)

        completions = list(completer.get_completions(document, MagicMock()))
        completion_texts = [c.text for c in completions]

        assert "chat-assistant" in completion_texts
        assert "code-helper" in completion_texts


def test_completer_no_completion_for_text():
    """Test that free text doesn't trigger completion."""
    completer = TsugiteCompleter()
    document = Document("hello world", cursor_position=11)

    completions = list(completer.get_completions(document, MagicMock()))

    # No completions for free text
    assert len(completions) == 0


def test_discover_agents():
    """Test agent discovery."""
    completer = TsugiteCompleter()
    agents = completer._discover_agents()

    # Should at least find builtin agents
    assert isinstance(agents, list)
    # Builtin agents should be present
    assert len(agents) > 0


def test_complete_path(tmp_path):
    """Test file path completion."""
    # Create test files
    (tmp_path / "test1.txt").touch()
    (tmp_path / "test2.md").touch()
    (tmp_path / "subdir").mkdir()

    completer = TsugiteCompleter()

    # Test path completion
    paths = completer._complete_path(str(tmp_path / "test"))

    # Should find test1.txt and test2.md
    assert len(paths) >= 1
    # Check that at least one test file is found
    found_test_files = [p for p in paths if "test" in p and (p.endswith(".txt") or p.endswith(".md"))]
    assert len(found_test_files) >= 1


def test_update_agent_name():
    """Test updating current agent name."""
    completer = TsugiteCompleter(current_agent_name="old-agent")
    assert completer.current_agent_name == "old-agent"

    completer.update_agent_name("new-agent")
    assert completer.current_agent_name == "new-agent"
