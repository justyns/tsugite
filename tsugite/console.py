"""Centralized console creation utilities."""

import sys

from rich.console import Console


def get_stderr_console(no_color: bool = False) -> Console:
    """Get console for error/warning output to stderr.

    Args:
        no_color: Disable color output

    Returns:
        Console instance configured for stderr
    """
    return Console(file=sys.stderr, no_color=no_color)


def get_stdout_console(no_color: bool = False) -> Console:
    """Get console for standard output to stdout.

    Args:
        no_color: Disable color output

    Returns:
        Console instance configured for stdout
    """
    return Console(file=sys.stdout, no_color=no_color)


def get_output_console() -> Console:
    """Get console for standard output (alias for get_stdout_console with no_color=True)."""
    return get_stdout_console(no_color=True)


def get_error_console(headless: bool, console: Console) -> Console:
    """Get console for error output based on mode.

    Args:
        headless: Whether running in headless mode
        console: Default console to use in non-headless mode

    Returns:
        Console for error output (no-color stderr in headless, console otherwise)
    """
    return get_stderr_console(no_color=True) if headless else console
