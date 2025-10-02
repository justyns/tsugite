"""Loading animation utilities for CLI."""

import threading
import time
from contextlib import contextmanager
from typing import Generator

from rich.console import Console
from rich.live import Live
from rich.spinner import Spinner


class LoadingAnimation:
    def __init__(self, console: Console):
        self.console = console
        self.stop_event = threading.Event()
        self.thread = None

    def _animate_spinner(self, message: str) -> None:
        spinner = Spinner("dots", text=message, style="cyan")

        with Live(spinner, console=self.console, refresh_per_second=10, transient=True):
            while not self.stop_event.is_set():
                time.sleep(0.1)

        # Force clear the spinner line after Live context exits
        time.sleep(0.05)  # Small delay to ensure Live has fully cleaned up
        self.console.print("\r" + " " * 80 + "\r", end="")

    def _animate_simple(self, message: str) -> None:
        dots = ""
        while not self.stop_event.is_set():
            for i in range(4):
                if self.stop_event.is_set():
                    break
                dots = "." * i
                self.console.print(f"\r{message}{dots}   ", end="")
                time.sleep(0.5)

    def start(self, message: str = "Waiting for LLM response") -> None:
        self.stop_event.clear()

        if self.console.no_color:
            self.thread = threading.Thread(target=self._animate_simple, args=(message,))
        else:
            self.thread = threading.Thread(target=self._animate_spinner, args=(message,))

        self.thread.daemon = True
        self.thread.start()

    def stop(self) -> None:
        if self.thread and self.thread.is_alive():
            self.stop_event.set()
            self.thread.join(timeout=1.0)

            # Force clear the animation line for both modes
            # Small delay to ensure thread cleanup is complete
            time.sleep(0.05)
            self.console.print("\r" + " " * 80 + "\r", end="")


@contextmanager
def loading_animation(
    console: Console, message: str = "Waiting for LLM response", enabled: bool = True
) -> Generator[None, None, None]:
    """Context manager for showing loading animation during LLM calls."""
    if not enabled:
        yield
        return

    animation = LoadingAnimation(console)
    animation_started = False

    try:
        animation.start(message)
        animation_started = True
    except Exception:
        pass

    try:
        yield
    finally:
        if animation_started:
            try:
                animation.stop()
            except Exception:
                pass
