"""Helper functions for creating and managing UI loggers."""

from contextlib import contextmanager
from typing import Generator

from rich.console import Console

from tsugite.ui.base import CustomUIHandler, CustomUILogger
from tsugite.ui.plain import PlainUIHandler
from tsugite.ui_context import clear_ui_context, set_ui_context


@contextmanager
def custom_agent_ui(
    console: Console,
    show_code: bool = True,
    show_observations: bool = True,
    show_progress: bool = True,
    show_llm_messages: bool = False,
    show_execution_results: bool = True,
    show_execution_logs: bool = True,
    show_panels: bool = True,
) -> Generator[CustomUILogger, None, None]:
    """Context manager for custom agent UI.

    Args:
        console: Rich console instance
        show_code: Whether to display executed code
        show_observations: Whether to display tool observations
        show_progress: Whether to show progress spinner
        show_llm_messages: Whether to show LLM reasoning messages
        show_execution_results: Whether to show code execution results
        show_execution_logs: Whether to show execution logs
        show_panels: Whether to show Rich panels (borders and decorations)

    Yields:
        CustomUILogger: Logger instance with ui_handler and console
    """
    ui_handler = CustomUIHandler(
        console,
        show_code=show_code,
        show_observations=show_observations,
        show_llm_messages=show_llm_messages,
        show_execution_results=show_execution_results,
        show_execution_logs=show_execution_logs,
        show_panels=show_panels,
    )
    logger = CustomUILogger(ui_handler, console)

    # Store console and ui_handler in thread-local for ask_user integration
    set_ui_context(console=console, progress=None, ui_handler=ui_handler)

    try:
        if show_progress:
            with ui_handler.progress_context():
                yield logger
        else:
            yield logger
    finally:
        clear_ui_context()


def create_silent_logger() -> CustomUILogger:
    """Create a minimal logger for silent execution.

    Returns a logger with a console writing to /dev/null, so it produces no output.

    Returns:
        CustomUILogger with silent console
    """
    silent_console = Console(file=open("/dev/null", "w"))
    silent_handler = CustomUIHandler(
        silent_console,
        show_code=False,
        show_observations=False,
        show_llm_messages=False,
        show_execution_results=False,
        show_execution_logs=False,
        show_panels=False,
    )
    return CustomUILogger(silent_handler, silent_console)


def create_plain_logger() -> CustomUILogger:
    """Create a plain text logger for copy-paste friendly output.

    Returns a logger using PlainUIHandler with no colors, panels, or emojis.
    Ideal for:
    - Piped output
    - Copy-paste workflows
    - Screen readers
    - Logs and automation

    Returns:
        CustomUILogger with PlainUIHandler
    """
    plain_handler = PlainUIHandler()
    return CustomUILogger(plain_handler, plain_handler.console)


def create_live_template_logger(interactive: bool = True) -> CustomUILogger:
    """Create logger using Live Template handler with Tree and interactive prompts.

    Returns a logger using LiveTemplateHandler with Rich Live Display.
    Features:
    - Live Display with multi-panel layout
    - Tree visualization of execution steps
    - Interactive prompts (optional)
    - Real-time updates without scrolling

    Args:
        interactive: Enable interactive prompts during execution

    Returns:
        CustomUILogger with LiveTemplateHandler
    """
    from tsugite.ui.live_template import LiveTemplateHandler

    handler = LiveTemplateHandler(interactive=interactive)
    return CustomUILogger(handler, handler.console)
