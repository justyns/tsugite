"""Loading animation utilities for CLI."""

import threading
import time
from contextlib import contextmanager
from typing import Generator

from rich.console import Console
from rich.live import Live
from rich.spinner import Spinner


class LoadingAnimation:
    """Manages loading animations during LLM calls."""

    def __init__(self, console: Console):
        self.console = console
        self.stop_event = threading.Event()
        self.thread = None

    def _animate_spinner(self, message: str) -> None:
        """Run spinner animation in a separate thread."""
        spinner = Spinner("dots", text=message, style="cyan")

        with Live(spinner, console=self.console, refresh_per_second=10):
            while not self.stop_event.is_set():
                time.sleep(0.1)

    def _animate_simple(self, message: str) -> None:
        """Simple text-based animation for non-color mode."""
        dots = ""
        while not self.stop_event.is_set():
            for i in range(4):
                if self.stop_event.is_set():
                    break
                dots = "." * i
                self.console.print(f"\r{message}{dots}   ", end="")
                time.sleep(0.5)

    def start(self, message: str = "Waiting for LLM response") -> None:
        """Start the loading animation."""
        self.stop_event.clear()

        if self.console.no_color:
            # Simple text animation for no-color mode
            self.thread = threading.Thread(target=self._animate_simple, args=(message,))
        else:
            # Rich spinner animation for color mode
            self.thread = threading.Thread(target=self._animate_spinner, args=(message,))

        self.thread.daemon = True
        self.thread.start()

    def stop(self) -> None:
        """Stop the loading animation."""
        if self.thread and self.thread.is_alive():
            self.stop_event.set()
            self.thread.join(timeout=1.0)

            # Clear the animation line
            if self.console.no_color:
                self.console.print("\r" + " " * 50 + "\r", end="")


@contextmanager
def loading_animation(
    console: Console, message: str = "Waiting for LLM response", enabled: bool = True
) -> Generator[None, None, None]:
    """Context manager for showing loading animation during LLM calls.

    Args:
        console: Rich console instance
        message: Message to show with the animation
        enabled: Whether to show animation (disabled for non-interactive mode)
    """
    if not enabled:
        yield
        return

    animation = LoadingAnimation(console)
    animation_started = False

    try:
        try:
            animation.start(message)
            animation_started = True
        except Exception:
            # If animation start fails, continue without animation
            pass

        yield

    finally:
        if animation_started:
            try:
                animation.stop()
            except Exception:
                # Silently ignore animation cleanup failures
                pass
