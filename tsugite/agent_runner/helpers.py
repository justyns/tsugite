"""Shared helper functions for agent execution."""

import threading
from typing import Any, Optional

from rich.console import Console

from tsugite.console import get_stderr_console

# Console for warnings and debug output (stderr)
_stderr_console = get_stderr_console()

# Thread-local storage for tracking currently executing agent
_current_agent_context = threading.local()


def set_current_agent(name: str) -> None:
    """Set the name of the currently executing agent in thread-local storage."""
    _current_agent_context.name = name


def get_current_agent() -> Optional[str]:
    """Get the name of the currently executing agent from thread-local storage."""
    return getattr(_current_agent_context, "name", None)


def clear_current_agent() -> None:
    """Clear the currently executing agent from thread-local storage."""
    if hasattr(_current_agent_context, "name"):
        delattr(_current_agent_context, "name")


def get_display_console(custom_logger: Optional[Any]) -> Console:
    """Get console for displaying output, with fallback to stderr.

    Args:
        custom_logger: Custom logger instance (may be None)

    Returns:
        Console instance to use for output
    """
    if custom_logger and hasattr(custom_logger, "console"):
        return custom_logger.console
    return _stderr_console


def get_ui_handler(custom_logger: Optional[Any]) -> Optional[Any]:
    """Safely get UI handler from custom logger.

    Args:
        custom_logger: Custom logger instance (may be None)

    Returns:
        UI handler if available, None otherwise
    """
    return custom_logger.ui_handler if custom_logger and hasattr(custom_logger, "ui_handler") else None
