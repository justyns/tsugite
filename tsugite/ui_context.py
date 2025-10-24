"""Thread-local context for UI state access from tools."""

import threading
from contextlib import contextmanager
from typing import Any, Generator, Optional

from rich.console import Console
from rich.progress import Progress

# Thread-local storage for UI context
_context = threading.local()


def set_ui_context(
    console: Optional[Console] = None,
    progress: Optional[Progress] = None,
    ui_handler: Optional[Any] = None,
) -> None:
    """Store UI context in thread-local storage.

    Args:
        console: Rich console instance
        progress: Rich progress instance (spinner)
        ui_handler: UI handler instance (e.g., TextualUIHandler or CustomUIHandler)
    """
    _context.console = console
    _context.progress = progress
    _context.ui_handler = ui_handler


def get_console() -> Optional[Console]:
    """Get the console from thread-local storage."""
    return getattr(_context, "console", None)


def get_progress() -> Optional[Progress]:
    """Get the progress spinner from thread-local storage."""
    return getattr(_context, "progress", None)


def get_ui_handler() -> Optional[Any]:
    """Get the UI handler from thread-local storage.

    Returns:
        UI handler instance if available (e.g., TextualUIHandler), None otherwise
    """
    return getattr(_context, "ui_handler", None)


def clear_ui_context() -> None:
    """Clear UI context from thread-local storage."""
    _context.console = None
    _context.progress = None
    _context.ui_handler = None


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
