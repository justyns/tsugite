"""Thread-local context for UI state access from tools."""

import threading
from contextlib import contextmanager
from typing import Generator, Optional

from rich.console import Console
from rich.progress import Progress

# Thread-local storage for UI context
_context = threading.local()


def set_ui_context(console: Optional[Console] = None, progress: Optional[Progress] = None) -> None:
    """Store UI context in thread-local storage.

    Args:
        console: Rich console instance
        progress: Rich progress instance (spinner)
    """
    _context.console = console
    _context.progress = progress


def get_console() -> Optional[Console]:
    """Get the console from thread-local storage."""
    return getattr(_context, "console", None)


def get_progress() -> Optional[Progress]:
    """Get the progress spinner from thread-local storage."""
    return getattr(_context, "progress", None)


def clear_ui_context() -> None:
    """Clear UI context from thread-local storage."""
    _context.console = None
    _context.progress = None


@contextmanager
def paused_progress() -> Generator[None, None, None]:
    """Context manager to temporarily pause the progress spinner.

    This is useful for interactive operations that need user input
    without the spinner interfering with the display.

    Yields:
        None
    """
    progress = get_progress()

    if progress is not None:
        # Stop the progress temporarily
        progress.stop()

    try:
        yield
    finally:
        if progress is not None:
            # Restart the progress
            progress.start()
