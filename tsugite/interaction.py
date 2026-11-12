"""Interaction backends for user input across different adapters.

Provides a sync protocol that tools call to ask users questions,
with implementations for terminal (TTY), HTTP (SSE), Discord, and
non-interactive (scheduler) modes.
"""

import contextvars
from typing import List, Optional, Protocol, runtime_checkable

from tsugite.exceptions import ToolUnavailableError


@runtime_checkable
class InteractionBackend(Protocol):
    """Protocol for user interaction across adapters.

    All methods are sync — tools run on worker threads, never the event loop.
    """

    def ask_user(self, question: str, question_type: str = "text", options: Optional[List[str]] = None) -> str:
        """Ask the user a question and return their response.

        Args:
            question: The question text
            question_type: "text", "yes_no", "choice", or "approval"
            options: Options for choice/approval questions

        Returns:
            User's response as a string
        """
        ...


_interaction_backend_var: contextvars.ContextVar[Optional[InteractionBackend]] = contextvars.ContextVar(
    "interaction_backend", default=None
)


def set_interaction_backend(backend: Optional[InteractionBackend]) -> None:
    """Set the interaction backend for the current context."""
    _interaction_backend_var.set(backend)


def get_interaction_backend() -> Optional[InteractionBackend]:
    """Get the interaction backend from the current context."""
    return _interaction_backend_var.get()


class TerminalInteractionBackend:
    """Delegates to existing TTY-based interactive prompts."""

    def ask_user(self, question: str, question_type: str = "text", options: Optional[List[str]] = None) -> str:
        from tsugite.tools.interactive import _flush_input_buffer, handle_question_by_type, terminal_context

        with terminal_context() as console:
            return handle_question_by_type(question_type, question, options, console, _flush_input_buffer)


class NonInteractiveBackend:
    """Denies approvals and raises for every other question type.

    Used where no user is available to answer (scheduler, rooms, background runs).
    """

    def ask_user(self, question: str, question_type: str = "text", options: Optional[List[str]] = None) -> str:
        if question_type == "approval":
            # Never index 0, which would be "Approve".
            if options:
                for option in options:
                    if option.strip().lower() == "deny":
                        return option
            return "deny"
        raise ToolUnavailableError(
            f"No user is available in this run to answer the {question_type} question {question!r}. "
            "Do not assume an answer. Stop and report the question, or continue without it."
        )
