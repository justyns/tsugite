"""Tests for interactive user input tools."""

import sys
from unittest.mock import patch

import pytest

from tsugite.tools import call_tool


@pytest.fixture
def interactive_tool(reset_tool_registry):
    """Register interactive tool for testing."""
    from tsugite.tools import tool
    from tsugite.tools.interactive import ask_user

    tool(ask_user)


def test_ask_user_text_interactive(interactive_tool, monkeypatch):
    """Test freeform text question in interactive mode."""
    # Mock TTY check to return True
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)

    # Mock user input
    with patch("tsugite.tools.interactive.Prompt.ask", return_value="This is my answer"):
        result = call_tool("ask_user", question="What is your name?", question_type="text")

    assert result == "This is my answer"


def test_ask_user_yes_no_yes(interactive_tool, monkeypatch):
    """Test yes/no question returning yes."""
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)

    with patch("tsugite.tools.interactive.Confirm.ask", return_value=True):
        result = call_tool("ask_user", question="Do you agree?", question_type="yes_no")

    assert result == "yes"


def test_ask_user_yes_no_no(interactive_tool, monkeypatch):
    """Test yes/no question returning no."""
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)

    with patch("tsugite.tools.interactive.Confirm.ask", return_value=False):
        result = call_tool("ask_user", question="Do you agree?", question_type="yes_no")

    assert result == "no"


def test_ask_user_choice(interactive_tool, monkeypatch):
    """Test multiple choice question."""
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)

    options = ["Option A", "Option B", "Option C"]

    # Mock questionary to return "Option B"
    mock_question = patch("tsugite.tools.interactive.questionary.select")
    with mock_question as mock_select:
        mock_select.return_value.ask.return_value = "Option B"
        result = call_tool(
            "ask_user",
            question="Choose an option:",
            question_type="choice",
            options=options,
        )

    assert result == "Option B"


def test_ask_user_choice_first_option(interactive_tool, monkeypatch):
    """Test selecting first option in choice question."""
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)

    options = ["First", "Second", "Third"]

    # Mock questionary to return "First"
    mock_question = patch("tsugite.tools.interactive.questionary.select")
    with mock_question as mock_select:
        mock_select.return_value.ask.return_value = "First"
        result = call_tool(
            "ask_user",
            question="Pick one:",
            question_type="choice",
            options=options,
        )

    assert result == "First"


def test_ask_user_choice_last_option(interactive_tool, monkeypatch):
    """Test selecting last option in choice question."""
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)

    options = ["Alpha", "Beta", "Gamma"]

    # Mock questionary to return "Gamma"
    mock_question = patch("tsugite.tools.interactive.questionary.select")
    with mock_question as mock_select:
        mock_select.return_value.ask.return_value = "Gamma"
        result = call_tool(
            "ask_user",
            question="Pick one:",
            question_type="choice",
            options=options,
        )

    assert result == "Gamma"


def test_ask_user_non_interactive_mode(interactive_tool, monkeypatch):
    """Test that tool fails in non-interactive mode."""
    # Mock TTY check to return False (non-interactive)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)

    with pytest.raises(RuntimeError, match="Cannot use ask_user tool in non-interactive mode"):
        call_tool("ask_user", question="What is your name?", question_type="text")


def test_ask_user_invalid_question_type(interactive_tool, monkeypatch):
    """Test validation of question_type parameter."""
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)

    with pytest.raises(RuntimeError, match="Tool 'ask_user' failed"):
        call_tool("ask_user", question="Test?", question_type="invalid_type")


def test_ask_user_choice_missing_options(interactive_tool, monkeypatch):
    """Test that choice type requires options parameter."""
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)

    with pytest.raises(RuntimeError, match="Tool 'ask_user' failed"):
        call_tool("ask_user", question="Choose:", question_type="choice")


def test_ask_user_choice_too_few_options(interactive_tool, monkeypatch):
    """Test that choice type requires at least 2 options."""
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)

    with pytest.raises(RuntimeError, match="Tool 'ask_user' failed"):
        call_tool("ask_user", question="Choose:", question_type="choice", options=["Only one"])


def test_ask_user_keyboard_interrupt(interactive_tool, monkeypatch):
    """Test handling of keyboard interrupt during user input."""
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)

    with patch("tsugite.tools.interactive.Prompt.ask", side_effect=KeyboardInterrupt()):
        with pytest.raises(RuntimeError, match="User input interrupted"):
            call_tool("ask_user", question="Test?", question_type="text")


def test_ask_user_default_question_type(interactive_tool, monkeypatch):
    """Test that default question type is 'text'."""
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)

    with patch("tsugite.tools.interactive.Prompt.ask", return_value="Answer"):
        result = call_tool("ask_user", question="Question?")

    assert result == "Answer"


def test_ask_user_empty_response(interactive_tool, monkeypatch):
    """Test handling of empty user response."""
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)

    with patch("tsugite.tools.interactive.Prompt.ask", return_value=""):
        result = call_tool("ask_user", question="Optional input?", question_type="text")

    assert result == ""


def test_ask_user_choice_with_many_options(interactive_tool, monkeypatch):
    """Test choice question with many options."""
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)

    options = [f"Option {i}" for i in range(1, 11)]  # 10 options

    # Mock questionary to return "Option 7"
    mock_question = patch("tsugite.tools.interactive.questionary.select")
    with mock_question as mock_select:
        mock_select.return_value.ask.return_value = "Option 7"
        result = call_tool(
            "ask_user",
            question="Select from many:",
            question_type="choice",
            options=options,
        )

    assert result == "Option 7"
