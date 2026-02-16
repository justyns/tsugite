"""Context for UI state access from tools using contextvars."""

import contextvars
from contextlib import contextmanager
from typing import Any, Generator, Optional

from rich.console import Console
from rich.progress import Progress

# Context variables for UI context (can be propagated to threads)
_console_var: contextvars.ContextVar[Optional[Console]] = contextvars.ContextVar("console", default=None)
_progress_var: contextvars.ContextVar[Optional[Progress]] = contextvars.ContextVar("progress", default=None)
_ui_handler_var: contextvars.ContextVar[Optional[Any]] = contextvars.ContextVar("ui_handler", default=None)
_event_bus_var: contextvars.ContextVar[Optional[Any]] = contextvars.ContextVar("event_bus", default=None)


def set_ui_context(
    console: Optional[Console] = None,
    progress: Optional[Progress] = None,
    ui_handler: Optional[Any] = None,
    event_bus: Optional[Any] = None,
) -> None:
    """Store UI context in context variables.

    Args:
        console: Rich console instance
        progress: Rich progress instance (spinner)
        ui_handler: UI handler instance (e.g., CustomUIHandler)
        event_bus: Event bus instance for emitting events
    """
    _console_var.set(console)
    _progress_var.set(progress)
    _ui_handler_var.set(ui_handler)
    _event_bus_var.set(event_bus)


def get_console() -> Optional[Console]:
    """Get the console from context variables."""
    return _console_var.get()


def get_progress() -> Optional[Progress]:
    """Get the progress spinner from context variables."""
    return _progress_var.get()


def get_ui_handler() -> Optional[Any]:
    """Get the UI handler from context variables.

    Returns:
        UI handler instance if available, None otherwise
    """
    return _ui_handler_var.get()


def get_event_bus() -> Optional[Any]:
    """Get the event bus from context variables.

    Returns:
        EventBus instance if available, None otherwise
    """
    return _event_bus_var.get()


def clear_ui_context() -> None:
    """Clear UI context from context variables."""
    _console_var.set(None)
    _progress_var.set(None)
    _ui_handler_var.set(None)
    _event_bus_var.set(None)


@contextmanager
def paused_progress() -> Generator[None, None, None]:
    """Context manager to temporarily pause UI elements for user input.

    This pauses either:
    - Live Display (if a UI handler with pause_for_input exists)
    - Progress spinner (fallback for older UI modes)

    This is useful for interactive operations that need user input
    without UI elements interfering with the display.

    Yields:
        None
    """
    ui_handler = get_ui_handler()
    progress = get_progress()

    # Check if UI handler has a pause_for_input method (Live Display)
    if ui_handler is not None and hasattr(ui_handler, "pause_for_input"):
        # Use the UI handler's pause mechanism (for Live Display)
        with ui_handler.pause_for_input():
            yield
    else:
        # Fallback: pause old-style progress spinner
        if progress is not None:
            progress.stop()

        try:
            yield
        finally:
            if progress is not None:
                progress.start()
