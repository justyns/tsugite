"""Interactive user input tools for Tsugite agents."""

import sys
import termios
import time
from typing import List, Optional

import questionary
from questionary import Style
from rich.console import Console
from rich.prompt import Confirm, Prompt

from ..tools import tool
from ..ui_context import get_console, paused_progress
from ..utils import is_interactive, validation_error

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

    # Get console from UI context if available, otherwise create new one
    console = get_console() or Console()

    try:
        # Pause progress spinner during user interaction
        with paused_progress():
            if question_type == "text":
                # Freeform text input
                console.print(f"\n[cyan]Question:[/cyan] {question}")
                response = Prompt.ask("[yellow]Your answer[/yellow]")
                return response

            elif question_type == "yes_no":
                # Yes/No question
                console.print(f"\n[cyan]Question:[/cyan] {question}")
                result = Confirm.ask("[yellow]Your answer[/yellow]")
                return "yes" if result else "no"

            elif question_type == "choice":
                # Multiple choice with arrow key navigation
                console.print()  # Add blank line for spacing

                # Flush input buffer to prevent accidental Enter presses
                _flush_input_buffer()

                # Small delay to prevent rapid double-press issues
                time.sleep(0.15)

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

                return answer

    except KeyboardInterrupt:
        raise RuntimeError("User input interrupted by keyboard interrupt")
    except Exception as e:
        raise RuntimeError(f"Failed to get user input: {e}")
