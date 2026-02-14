"""Interaction backends for user input across different adapters.

Provides a sync protocol that tools call to ask users questions,
with implementations for terminal (TTY), HTTP (SSE), Discord, and
non-interactive (scheduler) modes.
"""

import contextvars
from typing import List, Optional, Protocol, runtime_checkable


@runtime_checkable
class InteractionBackend(Protocol):
    """Protocol for user interaction across adapters.

    All methods are sync — tools run on worker threads, never the event loop.
    """

    def ask_user(self, question: str, question_type: str = "text", options: Optional[List[str]] = None) -> str:
        """Ask the user a question and return their response.

        Args:
            question: The question text
            question_type: "text", "yes_no", or "choice"
            options: Options for choice questions

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
        from tsugite.tools.interactive import handle_question_by_type, terminal_context, _flush_input_buffer

        with terminal_context() as console:
            return handle_question_by_type(question_type, question, options, console, _flush_input_buffer)


class NonInteractiveBackend:
    """Returns defaults for yes_no/choice, raises for freeform text.

    Used by the scheduler adapter where no user is available.
    """

    def __init__(self, default_yes_no: str = "yes", default_choice_index: int = 0):
        self._default_yes_no = default_yes_no
        self._default_choice_index = default_choice_index

    def ask_user(self, question: str, question_type: str = "text", options: Optional[List[str]] = None) -> str:
        if question_type == "yes_no":
            return self._default_yes_no
        if question_type == "choice":
            if not options:
                raise RuntimeError("Cannot ask choice questions without options in non-interactive mode.")
            idx = min(self._default_choice_index, len(options) - 1)
            return options[idx]
        raise RuntimeError(
            "Cannot ask freeform text questions in non-interactive mode (scheduler). "
            "Use yes_no or choice question types instead."
        )
