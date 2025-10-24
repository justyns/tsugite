"""Interactive user input tools for Tsugite agents."""

import re
import sys
import termios
import time
from typing import List, Optional

import nest_asyncio
import questionary
from questionary import Style
from rich.console import Console
from rich.prompt import Confirm, Prompt

from ..tools import tool
from ..ui_context import paused_progress
from ..utils import is_interactive, validation_error

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
        raise validation_error(
            "question_type",
            question_type,
            f"must be one of {', '.join(valid_types)}",
        )

    # Validate options for choice type
    if question_type == "choice":
        if not options or len(options) < 2:
            raise validation_error(
                "options",
                str(options),
                "must provide at least 2 options for choice type questions",
            )

    # Create a console that writes directly to the real terminal
    # This bypasses any stdout redirection from the executor
    terminal_console = Console(file=sys.__stdout__, force_terminal=True)

    # Save current stdin (may be redirected by executor)
    old_stdin = sys.stdin

    try:
        # Restore real terminal stdin for user input
        # Only stdin needs to be restored - we use terminal_console for output
        sys.stdin = sys.__stdin__

        # Pause progress spinner while showing prompts
        with paused_progress():
            if question_type == "text":
                # Freeform text input
                terminal_console.print(f"\n[cyan]Question:[/cyan] {question}")

                # Flush input buffer to prevent accidental Enter presses
                _flush_input_buffer()

                response = Prompt.ask("[yellow]Your answer[/yellow]", console=terminal_console)

                # Write answer to captured stdout so it appears in observation for LLM
                print(f"User answered: {response}")

                return response

            elif question_type == "yes_no":
                # Yes/No question
                terminal_console.print(f"\n[cyan]Question:[/cyan] {question}")

                # Flush input buffer to prevent accidental Enter presses
                _flush_input_buffer()

                result = Confirm.ask("[yellow]Your answer[/yellow]", console=terminal_console)
                answer = "yes" if result else "no"

                # Write answer to captured stdout so it appears in observation for LLM
                print(f"User answered: {answer}")

                return answer

            elif question_type == "choice":
                # Multiple choice with arrow key navigation
                terminal_console.print()  # Add blank line for spacing

                # Flush input buffer to prevent accidental Enter presses
                _flush_input_buffer()

                # Small delay to prevent rapid double-press issues
                time.sleep(0.15)

                # Questionary needs real stdout for its TUI - temporarily restore it
                old_stdout_temp = sys.stdout
                sys.stdout = sys.__stdout__

                try:
                    # Show choice prompt with clear instructions
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
                finally:
                    # Restore captured stdout for executor
                    sys.stdout = old_stdout_temp

                # Write answer to captured stdout so it appears in observation for LLM
                print(f"User answered: {answer}")

                return answer

    except KeyboardInterrupt:
        raise RuntimeError("User input interrupted by keyboard interrupt")
    except Exception as e:
        raise RuntimeError(f"Failed to get user input: {e}")
    finally:
        # Restore stdin (for executor)
        sys.stdin = old_stdin


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
        raise validation_error(
            "questions",
            str(questions),
            "must be a non-empty list of question dictionaries",
        )

    # Validate each question structure and auto-generate IDs if needed
    valid_types = ["text", "yes_no", "choice"]
    seen_ids = set()

    for i, q in enumerate(questions):
        if not isinstance(q, dict):
            raise validation_error(
                f"questions[{i}]",
                str(q),
                "must be a dictionary",
            )

        # Check required fields
        if "question" not in q:
            raise validation_error(
                f"questions[{i}]",
                str(q),
                "missing required field 'question'",
            )
        # Accept both 'type' and 'question_type' for compatibility
        if "type" not in q and "question_type" not in q:
            raise validation_error(
                f"questions[{i}]",
                str(q),
                "missing required field 'type' or 'question_type'",
            )

        # Normalize to 'type' if 'question_type' was provided
        if "question_type" in q and "type" not in q:
            q["type"] = q["question_type"]

        # Auto-generate ID if not provided
        if "id" not in q:
            base_id = _generate_id_from_question(q["question"])
            # Handle duplicates by appending a number
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
                raise validation_error(
                    f"questions[{i}].id",
                    q_id,
                    "duplicate question ID - all question IDs must be unique",
                )

        seen_ids.add(q_id)

        # Validate type
        q_type = q["type"]
        if q_type not in valid_types:
            raise validation_error(
                f"questions[{i}].type",
                q_type,
                f"must be one of {', '.join(valid_types)}",
            )

        # Validate options for choice type
        if q_type == "choice":
            if "options" not in q or not q["options"] or len(q["options"]) < 2:
                raise validation_error(
                    f"questions[{i}].options",
                    str(q.get("options")),
                    "must provide at least 2 options for choice type questions",
                )

    # Create a console that writes directly to the real terminal
    # This bypasses any stdout redirection from the executor
    terminal_console = Console(file=sys.__stdout__, force_terminal=True)

    # Save current stdin (may be redirected by executor)
    old_stdin = sys.stdin

    # Collect responses
    responses = {}

    try:
        # Restore real terminal stdin for user input
        # Only stdin needs to be restored - we use terminal_console for output
        sys.stdin = sys.__stdin__

        # Pause progress spinner while showing prompts
        with paused_progress():
            terminal_console.print("\n[bold cyan]Please answer the following questions:[/bold cyan]\n")

            for i, q in enumerate(questions, 1):
                q_id = q["id"]
                q_text = q["question"]
                q_type = q["type"]

                # Show question number
                terminal_console.print(f"[dim]Question {i}/{len(questions)}[/dim]")

                if q_type == "text":
                    # Freeform text input
                    terminal_console.print(f"[cyan]{q_text}[/cyan]")

                    # Flush input buffer to prevent accidental Enter presses
                    _flush_input_buffer()

                    response = Prompt.ask("[yellow]Your answer[/yellow]", console=terminal_console)
                    responses[q_id] = response

                elif q_type == "yes_no":
                    # Yes/No question
                    terminal_console.print(f"[cyan]{q_text}[/cyan]")

                    # Flush input buffer to prevent accidental Enter presses
                    _flush_input_buffer()

                    result = Confirm.ask("[yellow]Your answer[/yellow]", console=terminal_console)
                    responses[q_id] = "yes" if result else "no"

                elif q_type == "choice":
                    # Multiple choice
                    options = q["options"]

                    # Flush input buffer to prevent accidental Enter presses
                    _flush_input_buffer()

                    # Small delay to prevent rapid double-press issues
                    time.sleep(0.15)

                    # Questionary needs real stdout for its TUI - temporarily restore it
                    old_stdout_temp = sys.stdout
                    sys.stdout = sys.__stdout__

                    try:
                        answer = questionary.select(
                            q_text,
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
                    finally:
                        # Restore captured stdout for executor
                        sys.stdout = old_stdout_temp

                    responses[q_id] = answer

                # Add spacing between questions (except after last one)
                if i < len(questions):
                    terminal_console.print()

            terminal_console.print("\n[green]✓ All questions answered[/green]\n")

            # Write summary of answers to captured stdout so it appears in observation for LLM
            print("\nUser responses:")
            for q_id, answer in responses.items():
                print(f"  {q_id}: {answer}")

    except KeyboardInterrupt:
        raise RuntimeError("User input interrupted by keyboard interrupt")
    except Exception as e:
        raise RuntimeError(f"Failed to get user input: {e}")
    finally:
        # Restore stdin (for executor)
        sys.stdin = old_stdin

    return responses
