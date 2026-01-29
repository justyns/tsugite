"""Interactive user input tools for Tsugite agents."""

import re
import sys
import termios
import time
from contextlib import contextmanager
from typing import List, Optional

import nest_asyncio
import questionary
from questionary import Style
from rich.console import Console
from rich.prompt import Confirm, Prompt

from ..tools import tool
from ..ui_context import paused_progress
from ..utils import is_interactive

# Allow nested event loops (needed for questionary in async contexts)
nest_asyncio.apply()

# Custom style for questionary to match Rich theme
QUESTIONARY_STYLE = Style(
    [
        ("qmark", "fg:cyan bold"),  # Question mark
        ("question", "fg:cyan bold"),  # Question text
        ("answer", "fg:yellow bold"),  # Selected answer
        ("pointer", "fg:yellow bold"),  # Selection pointer
        ("highlighted", "fg:yellow bold"),  # Highlighted option
        ("selected", "fg:green"),  # Already selected (for checkbox)
        ("instruction", "fg:white"),  # Instructions - white/default for better readability
    ]
)


def _flush_input_buffer() -> None:
    """Flush any pending input from stdin to prevent accidental key presses.

    This prevents issues where a user accidentally hits Enter twice and
    unintentionally confirms a pre-selected option.
    """
    if not is_interactive():
        return

    try:
        # Save current terminal settings
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)

        # Flush input buffer
        termios.tcflush(fd, termios.TCIFLUSH)

        # Restore settings
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    except (termios.error, OSError):
        # If we can't flush (e.g., not a real TTY), just continue
        pass


def _generate_id_from_question(question: str) -> str:
    """Generate a valid ID from question text.

    Args:
        question: The question text

    Returns:
        A lowercase, underscored ID based on the question
    """
    # Remove punctuation and convert to lowercase
    clean = re.sub(r"[^\w\s]", "", question.lower())
    # Replace spaces with underscores
    id_str = re.sub(r"\s+", "_", clean.strip())
    # Truncate if too long
    if len(id_str) > 50:
        id_str = id_str[:50]
    # Remove trailing underscores
    id_str = id_str.rstrip("_")
    return id_str or "question"


@tool
def final_answer(result: str) -> None:
    """Return the final result to the user and stop execution.

    Call this when you have completed the task and want to return the result.
    Execution stops after calling this - no further code will run.

    Args:
        result: The final result to return to the user

    Example:
        final_answer("The answer is 42")
        # Execution stops here
    """
    # This tool exists for documentation purposes only.
    # The executor has a built-in final_answer that actually handles completion.
    # See agent.py _inject_tools_into_executor() for why this isn't injected.
    pass


@tool
def send_message(message: str) -> str:
    """Send a progress message to the user without stopping execution.

    Use this to provide status updates during long-running operations.
    Unlike final_answer(), execution continues after calling this.

    Args:
        message: The message to send to the user

    Returns:
        Confirmation that message was sent

    Example:
        send_message("Starting file analysis...")
        files = list_files("/path")
        send_message(f"Found {len(files)} files, processing...")
        final_answer("Analysis complete")
    """
    # This tool exists for documentation purposes only.
    # The executor has a built-in send_message that has access to the event_bus.
    # See agent.py _inject_tools_into_executor() for why this isn't injected.
    return f"Message sent: {message}"


@tool
def ask_user(question: str, question_type: str = "text", options: Optional[List[str]] = None) -> str:
    """Ask the user a question interactively.

    This tool allows the LLM to ask the user for input during agent execution.
    It supports three types of questions:
    - text: Freeform text input
    - yes_no: Binary yes/no question (returns "yes" or "no")
    - choice: Multiple choice from a list of options

    Args:
        question: The question to ask the user
        question_type: Type of question - "text", "yes_no", or "choice"
        options: List of options for "choice" type questions (required for choice type)

    Returns:
        User's response as a string

    Raises:
        ValueError: If not in interactive mode or invalid parameters
        RuntimeError: If user interaction fails
    """
    if not is_interactive():
        raise RuntimeError(
            "Cannot use ask_user tool in non-interactive mode. "
            "This tool requires a terminal with user input capability."
        )

    # Validate question type
    valid_types = ["text", "yes_no", "choice"]
    if question_type not in valid_types:
        raise ValueError(f"Invalid question_type '{question_type}': must be one of {', '.join(valid_types)}")

    # Validate options for choice type
    if question_type == "choice":
        if not options or len(options) < 2:
            raise ValueError(f"Invalid options {options}: must provide at least 2 options for choice type questions")

    try:
        with terminal_context() as console:
            return handle_question_by_type(question_type, question, options, console, _flush_input_buffer)
    except KeyboardInterrupt:
        raise RuntimeError("User input interrupted by keyboard interrupt")
    except Exception as e:
        raise RuntimeError(f"Failed to get user input: {e}")


@tool
def ask_user_batch(questions: List[dict]) -> dict:
    """Ask the user multiple questions at once and collect all responses.

    This tool allows the LLM to ask multiple questions in a batch, showing all questions
    upfront and collecting all answers before returning to the agent. This provides a
    better user experience for multi-field forms or related questions.

    Args:
        questions: List of question dictionaries. REQUIRED fields per question:
            - question (str, REQUIRED): The question text to display
            - type (str, REQUIRED): Question type - "text", "yes_no", or "choice"
              (can also use "question_type" for consistency with ask_user)
            - options (List[str]): Options for choice type (REQUIRED when type="choice")
            - id (str, OPTIONAL): Unique identifier for response dict key.
              If not provided, auto-generated from question text.

    Returns:
        Dictionary mapping question IDs to user responses

    Raises:
        ValueError: If not in interactive mode or invalid question structure
        RuntimeError: If user interaction fails

    Example:
        # With explicit IDs
        responses = ask_user_batch(questions=[
            {"id": "email", "question": "What is your email?", "type": "text"},
            {"id": "save", "question": "Save to file?", "type": "yes_no"},
            {"id": "format", "question": "Choose format:", "type": "choice", "options": ["json", "txt"]}
        ])
        # Returns: {"email": "user@example.com", "save": "yes", "format": "json"}

        # Without IDs (auto-generated from questions)
        responses = ask_user_batch(questions=[
            {"question": "What is your name?", "type": "text"},
            {"question": "Save to file?", "type": "yes_no"},
            {"question": "Choose format:", "type": "choice", "options": ["json", "txt", "md"]}
        ])
        # Returns: {"what_is_your_name": "Alice", "save_to_file": "yes", "choose_format": "json"}
    """
    if not is_interactive():
        raise RuntimeError(
            "Cannot use ask_user_batch tool in non-interactive mode. "
            "This tool requires a terminal with user input capability."
        )

    # Validate questions list
    if not questions or not isinstance(questions, list):
        raise ValueError(f"Invalid questions {questions}: must be a non-empty list of question dictionaries")

    # Validate each question structure and auto-generate IDs if needed
    valid_types = ["text", "yes_no", "choice"]
    validate_batch_questions(questions, valid_types)

    # Collect responses
    responses = {}

    try:
        with terminal_context() as console:
            console.print("\n[bold cyan]Please answer the following questions:[/bold cyan]\n")

            for i, q in enumerate(questions, 1):
                q_id = q["id"]
                q_text = q["question"]
                q_type = q["type"]
                options = q.get("options")

                # Show question number
                console.print(f"[dim]Question {i}/{len(questions)}[/dim]")

                # Handle question based on type
                answer = handle_question_by_type(q_type, q_text, options, console, _flush_input_buffer)
                responses[q_id] = answer

                # Add spacing between questions (except after last one)
                if i < len(questions):
                    console.print()

            console.print("\n[green]✓ All questions answered[/green]\n")

            # Write summary of answers to captured stdout so it appears in observation for LLM
            print("\nUser responses:")
            for q_id, answer in responses.items():
                print(f"  {q_id}: {answer}")

    except KeyboardInterrupt:
        raise RuntimeError("User input interrupted by keyboard interrupt")
    except Exception as e:
        raise RuntimeError(f"Failed to get user input: {e}")

    return responses


@contextmanager
def terminal_context():
    """Context manager for terminal I/O with proper stdin/stdout handling.

    Yields:
        Console: A terminal console that writes directly to real terminal
    """
    terminal_console = Console(file=sys.__stdout__, force_terminal=True)
    old_stdin = sys.stdin

    try:
        # Restore real terminal stdin for user input
        sys.stdin = sys.__stdin__

        # Pause progress spinner while showing prompts
        with paused_progress():
            yield terminal_console
    finally:
        # Restore stdin for executor
        sys.stdin = old_stdin


def ask_text_question(question: str, console: Console, flush_fn) -> str:
    """Ask a freeform text question.

    Args:
        question: Question text
        console: Rich console for output
        flush_fn: Function to flush input buffer

    Returns:
        User's text response
    """
    console.print(f"\n[cyan]Question:[/cyan] {question}")
    flush_fn()
    response = Prompt.ask("[yellow]Your answer[/yellow]", console=console)
    print(f"User answered: {response}")
    return response


def ask_yes_no_question(question: str, console: Console, flush_fn) -> str:
    """Ask a yes/no question.

    Args:
        question: Question text
        console: Rich console for output
        flush_fn: Function to flush input buffer

    Returns:
        "yes" or "no"
    """
    console.print(f"\n[cyan]Question:[/cyan] {question}")
    flush_fn()
    result = Confirm.ask("[yellow]Your answer[/yellow]", console=console)
    answer = "yes" if result else "no"
    print(f"User answered: {answer}")
    return answer


def ask_choice_question(question: str, options: List[str], console: Console, flush_fn) -> str:
    """Ask a multiple choice question with arrow key navigation.

    Args:
        question: Question text
        options: List of choice options
        console: Rich console for output
        flush_fn: Function to flush input buffer

    Returns:
        Selected option

    Raises:
        KeyboardInterrupt: If user cancels with Ctrl+C
    """
    console.print()  # Add blank line for spacing
    flush_fn()

    # Small delay to prevent rapid double-press issues
    time.sleep(0.15)

    # Questionary needs real stdout for its TUI - temporarily restore it
    old_stdout = sys.stdout
    sys.stdout = sys.__stdout__

    try:
        answer = questionary.select(
            question,
            choices=options,
            style=QUESTIONARY_STYLE,
            use_arrow_keys=True,
            use_shortcuts=True,
            use_jk_keys=True,
            instruction="(Use arrow keys or j/k, Enter to select)",
        ).ask()

        # questionary returns None on Ctrl+C
        if answer is None:
            raise KeyboardInterrupt()

        print(f"User answered: {answer}")
        return answer
    finally:
        # Restore captured stdout for executor
        sys.stdout = old_stdout


def validate_batch_questions(questions: List[dict], valid_types: List[str]) -> None:
    """Validate batch questions structure and auto-generate IDs.

    Modifies questions in-place to add auto-generated IDs where missing.

    Args:
        questions: List of question dictionaries to validate
        valid_types: List of valid question types

    Raises:
        ValueError: If validation fails
    """
    seen_ids = set()

    for i, q in enumerate(questions):
        if not isinstance(q, dict):
            raise ValueError(f"Invalid questions[{i}] {q}: must be a dictionary")

        # Check required fields
        if "question" not in q:
            raise ValueError(f"Invalid questions[{i}] {q}: missing required field 'question'")

        # Accept both 'type' and 'question_type' for compatibility
        if "type" not in q and "question_type" not in q:
            raise ValueError(f"Invalid questions[{i}] {q}: missing required field 'type' or 'question_type'")

        # Normalize to 'type' if 'question_type' was provided
        if "question_type" in q and "type" not in q:
            q["type"] = q["question_type"]

        # Auto-generate ID if not provided
        if "id" not in q:
            base_id = _generate_id_from_question(q["question"])
            q_id = base_id
            counter = 1
            while q_id in seen_ids:
                q_id = f"{base_id}_{counter}"
                counter += 1
            q["id"] = q_id
        else:
            q_id = q["id"]
            # Check for duplicate explicit IDs
            if q_id in seen_ids:
                raise ValueError(
                    f"Invalid questions[{i}].id '{q_id}': duplicate question ID - all question IDs must be unique"
                )

        seen_ids.add(q_id)

        # Validate type
        q_type = q["type"]
        if q_type not in valid_types:
            raise ValueError(f"Invalid questions[{i}].type '{q_type}': must be one of {', '.join(valid_types)}")

        # Validate options for choice type
        if q_type == "choice":
            if "options" not in q or not q["options"] or len(q["options"]) < 2:
                raise ValueError(
                    f"Invalid questions[{i}].options {q.get('options')}: must provide at least 2 options for choice type questions"
                )


def handle_question_by_type(q_type: str, q_text: str, options: Optional[List[str]], console: Console, flush_fn) -> str:
    """Handle a question based on its type.

    Args:
        q_type: Question type ("text", "yes_no", or "choice")
        q_text: Question text
        options: Options for choice questions (required if q_type == "choice")
        console: Rich console for output
        flush_fn: Function to flush input buffer

    Returns:
        User's response

    Raises:
        ValueError: If invalid question type or missing options
    """
    if q_type == "text":
        return ask_text_question(q_text, console, flush_fn)
    elif q_type == "yes_no":
        return ask_yes_no_question(q_text, console, flush_fn)
    elif q_type == "choice":
        if not options:
            raise ValueError("Options required for choice type questions")
        return ask_choice_question(q_text, options, console, flush_fn)
    else:
        raise ValueError(f"Invalid question type: {q_type}")
