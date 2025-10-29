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


def set_multistep_ui_context(custom_logger: Optional[Any], step_number: int, step_name: str, total_steps: int) -> None:
    """Set multistep context in UI handler if available.

    Args:
        custom_logger: Custom logger instance (may be None)
        step_number: Current step number
        step_name: Name of current step
        total_steps: Total number of steps
    """
    ui_handler = get_ui_handler(custom_logger)
    if ui_handler:
        ui_handler.set_multistep_context(step_number, step_name, total_steps)


def clear_multistep_ui_context(custom_logger: Optional[Any]) -> None:
    """Clear multistep context from UI handler if available.

    Args:
        custom_logger: Custom logger instance (may be None)
    """
    ui_handler = get_ui_handler(custom_logger)
    if ui_handler:
        ui_handler.clear_multistep_context()


def print_step_progress(
    custom_logger: Optional[Any], step_header: str, message: str, debug: bool = False, style: str = "cyan"
) -> None:
    """Print step progress message using event system.

    Args:
        custom_logger: Custom logger instance (may be None)
        step_header: Step header string
        message: Message to display
        debug: Whether debug mode is active (skips output if True)
        style: Rich style string (e.g., "cyan", "green", "yellow")
    """
    if debug:
        return

    # Emit as StepProgressEvent through event bus
    ui_handler = get_ui_handler(custom_logger)
    if ui_handler:
        from tsugite.events import EventBus, StepProgressEvent

        event_bus = EventBus()
        event_bus.subscribe(ui_handler.handle_event)
        event_bus.emit(StepProgressEvent(message=f"{step_header} {message}", style=style))
