"""Tests for interactive user input tools."""

import sys
from unittest.mock import patch

import pytest

from tsugite.tools import call_tool


@pytest.fixture
def interactive_tool(reset_tool_registry):
    """Register interactive tools for testing."""
    from tsugite.tools import tool
    from tsugite.tools.interactive import ask_user, ask_user_batch

    tool(ask_user)
    tool(ask_user_batch)


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


# Tests for ask_user_batch


def test_ask_user_batch_mixed_types(interactive_tool, monkeypatch):
    """Test batch questions with mixed types (text, yes_no, choice)."""
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)

    questions = [
        {"id": "name", "question": "What is your name?", "type": "text"},
        {"id": "save", "question": "Save to file?", "type": "yes_no"},
        {"id": "format", "question": "Choose format:", "type": "choice", "options": ["json", "txt", "md"]},
    ]

    # Mock each interaction type
    with (
        patch("tsugite.tools.interactive.Prompt.ask", return_value="Alice"),
        patch("tsugite.tools.interactive.Confirm.ask", return_value=True),
        patch("tsugite.tools.interactive.questionary.select") as mock_select,
    ):
        mock_select.return_value.ask.return_value = "json"
        result = call_tool("ask_user_batch", questions=questions)

    assert result == {"name": "Alice", "save": "yes", "format": "json"}
    assert len(result) == 3


def test_ask_user_batch_all_text(interactive_tool, monkeypatch):
    """Test batch with all text questions."""
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)

    questions = [
        {"id": "first_name", "question": "First name?", "type": "text"},
        {"id": "last_name", "question": "Last name?", "type": "text"},
        {"id": "email", "question": "Email?", "type": "text"},
    ]

    # Mock Prompt.ask to return different values for each call
    responses = ["John", "Doe", "john@example.com"]
    with patch("tsugite.tools.interactive.Prompt.ask", side_effect=responses):
        result = call_tool("ask_user_batch", questions=questions)

    assert result == {
        "first_name": "John",
        "last_name": "Doe",
        "email": "john@example.com",
    }


def test_ask_user_batch_all_yes_no(interactive_tool, monkeypatch):
    """Test batch with all yes/no questions."""
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)

    questions = [
        {"id": "agree_terms", "question": "Agree to terms?", "type": "yes_no"},
        {"id": "subscribe", "question": "Subscribe to newsletter?", "type": "yes_no"},
        {"id": "public", "question": "Make profile public?", "type": "yes_no"},
    ]

    # Mock Confirm.ask to return different values
    with patch("tsugite.tools.interactive.Confirm.ask", side_effect=[True, False, True]):
        result = call_tool("ask_user_batch", questions=questions)

    assert result == {
        "agree_terms": "yes",
        "subscribe": "no",
        "public": "yes",
    }


def test_ask_user_batch_all_choice(interactive_tool, monkeypatch):
    """Test batch with all choice questions."""
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)

    questions = [
        {"id": "color", "question": "Favorite color?", "type": "choice", "options": ["red", "blue", "green"]},
        {"id": "size", "question": "Choose size:", "type": "choice", "options": ["small", "medium", "large"]},
    ]

    # Mock questionary.select to return different values
    mock_question = patch("tsugite.tools.interactive.questionary.select")
    with mock_question as mock_select:
        mock_select.return_value.ask.side_effect = ["blue", "large"]
        result = call_tool("ask_user_batch", questions=questions)

    assert result == {"color": "blue", "size": "large"}


def test_ask_user_batch_non_interactive(interactive_tool, monkeypatch):
    """Test that batch tool fails in non-interactive mode."""
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)

    questions = [{"id": "name", "question": "Name?", "type": "text"}]

    with pytest.raises(RuntimeError, match="Cannot use ask_user_batch tool in non-interactive mode"):
        call_tool("ask_user_batch", questions=questions)


def test_ask_user_batch_empty_list(interactive_tool, monkeypatch):
    """Test validation of empty questions list."""
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)

    with pytest.raises(RuntimeError, match="Tool 'ask_user_batch' failed"):
        call_tool("ask_user_batch", questions=[])


def test_ask_user_batch_auto_generate_id(interactive_tool, monkeypatch):
    """Test that ID is auto-generated from question text when not provided."""
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)

    questions = [{"question": "What is your name?", "type": "text"}]  # No 'id' provided

    with patch("tsugite.tools.interactive.Prompt.ask", return_value="Alice"):
        result = call_tool("ask_user_batch", questions=questions)

    # ID should be auto-generated from question
    assert "what_is_your_name" in result
    assert result["what_is_your_name"] == "Alice"


def test_ask_user_batch_missing_question(interactive_tool, monkeypatch):
    """Test validation when question is missing 'question' field."""
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)

    questions = [{"id": "name", "type": "text"}]  # Missing 'question'

    with pytest.raises(RuntimeError, match="Tool 'ask_user_batch' failed"):
        call_tool("ask_user_batch", questions=questions)


def test_ask_user_batch_missing_type(interactive_tool, monkeypatch):
    """Test validation when question is missing 'type' field."""
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)

    questions = [{"id": "name", "question": "What is your name?"}]  # Missing 'type'

    with pytest.raises(RuntimeError, match="Tool 'ask_user_batch' failed"):
        call_tool("ask_user_batch", questions=questions)


def test_ask_user_batch_invalid_type(interactive_tool, monkeypatch):
    """Test validation of invalid question type."""
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)

    questions = [{"id": "name", "question": "Name?", "type": "invalid_type"}]

    with pytest.raises(RuntimeError, match="Tool 'ask_user_batch' failed"):
        call_tool("ask_user_batch", questions=questions)


def test_ask_user_batch_duplicate_ids(interactive_tool, monkeypatch):
    """Test validation when questions have duplicate IDs."""
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)

    questions = [
        {"id": "name", "question": "First name?", "type": "text"},
        {"id": "name", "question": "Last name?", "type": "text"},  # Duplicate ID
    ]

    with pytest.raises(RuntimeError, match="Tool 'ask_user_batch' failed"):
        call_tool("ask_user_batch", questions=questions)


def test_ask_user_batch_choice_missing_options(interactive_tool, monkeypatch):
    """Test that choice questions require options."""
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)

    questions = [{"id": "color", "question": "Favorite color?", "type": "choice"}]  # Missing options

    with pytest.raises(RuntimeError, match="Tool 'ask_user_batch' failed"):
        call_tool("ask_user_batch", questions=questions)


def test_ask_user_batch_choice_too_few_options(interactive_tool, monkeypatch):
    """Test that choice questions need at least 2 options."""
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)

    questions = [{"id": "color", "question": "Favorite color?", "type": "choice", "options": ["red"]}]  # Only 1 option

    with pytest.raises(RuntimeError, match="Tool 'ask_user_batch' failed"):
        call_tool("ask_user_batch", questions=questions)


def test_ask_user_batch_keyboard_interrupt(interactive_tool, monkeypatch):
    """Test handling of keyboard interrupt during batch input."""
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)

    questions = [{"id": "name", "question": "Name?", "type": "text"}]

    with patch("tsugite.tools.interactive.Prompt.ask", side_effect=KeyboardInterrupt()):
        with pytest.raises(RuntimeError, match="User input interrupted"):
            call_tool("ask_user_batch", questions=questions)


def test_ask_user_batch_single_question(interactive_tool, monkeypatch):
    """Test batch with just one question works correctly."""
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)

    questions = [{"id": "name", "question": "What is your name?", "type": "text"}]

    with patch("tsugite.tools.interactive.Prompt.ask", return_value="Alice"):
        result = call_tool("ask_user_batch", questions=questions)

    assert result == {"name": "Alice"}
    assert len(result) == 1


def test_ask_user_batch_auto_id_multiple_questions(interactive_tool, monkeypatch):
    """Test auto-generated IDs for multiple questions."""
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)

    questions = [
        {"question": "What is your email?", "type": "text"},
        {"question": "Save to file?", "type": "yes_no"},
        {"question": "Choose format:", "type": "choice", "options": ["json", "txt"]},
    ]

    with (
        patch("tsugite.tools.interactive.Prompt.ask", return_value="test@example.com"),
        patch("tsugite.tools.interactive.Confirm.ask", return_value=True),
        patch("tsugite.tools.interactive.questionary.select") as mock_select,
    ):
        mock_select.return_value.ask.return_value = "json"
        result = call_tool("ask_user_batch", questions=questions)

    # Check auto-generated IDs
    assert "what_is_your_email" in result
    assert "save_to_file" in result
    assert "choose_format" in result
    assert result["what_is_your_email"] == "test@example.com"
    assert result["save_to_file"] == "yes"
    assert result["choose_format"] == "json"


def test_ask_user_batch_auto_id_duplicate_handling(interactive_tool, monkeypatch):
    """Test handling of duplicate auto-generated IDs."""
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)

    questions = [
        {"question": "Name?", "type": "text"},
        {"question": "Name?", "type": "text"},  # Same question text
        {"question": "Name?", "type": "text"},  # Same question text again
    ]

    with patch("tsugite.tools.interactive.Prompt.ask", side_effect=["Alice", "Bob", "Charlie"]):
        result = call_tool("ask_user_batch", questions=questions)

    # Should have unique IDs with numeric suffixes
    assert "name" in result
    assert "name_1" in result
    assert "name_2" in result
    assert result["name"] == "Alice"
    assert result["name_1"] == "Bob"
    assert result["name_2"] == "Charlie"


def test_ask_user_batch_question_type_field(interactive_tool, monkeypatch):
    """Test accepting 'question_type' field (compatibility with ask_user)."""
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)

    questions = [
        {"id": "name", "question": "What is your name?", "question_type": "text"},  # Using 'question_type'
        {"id": "save", "question": "Save?", "type": "yes_no"},  # Using 'type'
    ]

    with (
        patch("tsugite.tools.interactive.Prompt.ask", return_value="Alice"),
        patch("tsugite.tools.interactive.Confirm.ask", return_value=True),
    ):
        result = call_tool("ask_user_batch", questions=questions)

    assert result == {"name": "Alice", "save": "yes"}


def test_ask_user_batch_mixed_explicit_and_auto_ids(interactive_tool, monkeypatch):
    """Test mixing explicit IDs with auto-generated IDs."""
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)

    questions = [
        {"id": "user_name", "question": "What is your name?", "type": "text"},  # Explicit ID
        {"question": "What is your email?", "type": "text"},  # Auto-generated ID
    ]

    with patch("tsugite.tools.interactive.Prompt.ask", side_effect=["Alice", "alice@example.com"]):
        result = call_tool("ask_user_batch", questions=questions)

    assert "user_name" in result
    assert "what_is_your_email" in result
    assert result["user_name"] == "Alice"
    assert result["what_is_your_email"] == "alice@example.com"
