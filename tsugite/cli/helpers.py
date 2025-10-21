"""CLI helper functions."""

import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console

from tsugite.constants import TSUGITE_LOGO_NARROW, TSUGITE_LOGO_WIDE


def get_logo(console: Console) -> str:
    """Get appropriate logo based on terminal width."""
    return TSUGITE_LOGO_NARROW if console.width < 80 else TSUGITE_LOGO_WIDE


def get_error_console(headless: bool, console: Console) -> Console:
    """Get console for error output based on mode."""
    return Console(file=sys.stderr, no_color=True) if headless else console


def get_output_console() -> Console:
    """Get console for standard output."""
    return Console(file=sys.stdout, no_color=True)


def print_plain_section(console: Console, title: str, content: str, style: str = "") -> None:
    """Print a plain text section with simple separators.

    Args:
        console: Rich console instance
        title: Section title
        content: Section content
        style: Optional Rich style for content (e.g., "cyan", "green")
    """
    console.print()
    console.rule(title if not style else f"[{style}]{title}[/{style}]", style="dim")
    if style:
        console.print(f"[{style}]{content}[/{style}]")
    else:
        console.print(content)
    console.print()


def print_plain_info(console: Console, title: str, items: dict, style: str = "cyan") -> None:
    """Print plain text information list.

    Args:
        console: Rich console instance
        title: Section title
        items: Dict of label: value pairs
        style: Optional Rich style for labels
    """
    console.print()
    console.rule(f"[bold]{title}[/bold]", style="dim")
    for label, value in items.items():
        if style:
            console.print(f"[{style}]{label}:[/{style}] {value}")
        else:
            console.print(f"{label}: {value}")
    console.print()


def parse_cli_arguments(args: List[str]) -> tuple[List[str], str]:
    """Parse CLI arguments into agent references and prompt.

    Args:
        args: List of positional arguments from CLI

    Returns:
        Tuple of (agent_refs, prompt)

    Examples:
        ["+a", "+b", "task"] -> (["+a", "+b"], "task")
        ["+a", "create", "ticket"] -> (["+a"], "create ticket")
        ["agent.md", "helper.md", "do", "work"] -> (["agent.md", "helper.md"], "do work")
    """
    if not args:
        raise ValueError("No arguments provided")

    agents = []
    prompt_parts = []

    for i, arg in enumerate(args):
        # Check if this looks like an agent reference
        # Exclude arguments containing @ (file references) from being treated as agents
        # Also exclude arguments with spaces unless they're file paths (contain /)
        has_file_reference = "@" in arg
        has_path_separator = "/" in arg
        has_spaces = " " in arg
        is_agent = (
            (arg.startswith("+") or arg.endswith(".md") or has_path_separator)
            and not has_file_reference
            and not (has_spaces and not has_path_separator)
        )

        if is_agent and not prompt_parts:
            # Still collecting agents
            agents.append(arg)
        else:
            # First non-agent arg or after we started collecting prompt
            prompt_parts.append(arg)

    if not agents:
        # Default to builtin-default for auto-discovery
        agents = ["+builtin-default"]
        # All args become the prompt
        prompt = " ".join(args)
    else:
        prompt = " ".join(prompt_parts)

    return agents, prompt


@contextmanager
def agent_context(agent_path: str, root: Optional[str], console: Console):
    """Validate agent path and optionally change working directory."""

    original_cwd = None

    try:
        if root:
            root_path = Path(root)
            if not root_path.exists():
                console.print(f"[red]Working directory not found: {root}[/red]")
                raise typer.Exit(1)
            original_cwd = os.getcwd()
            os.chdir(str(root_path))

        agent_file = Path(agent_path)
        if not agent_file.exists():
            console.print(f"[red]Agent file not found: {agent_path}[/red]")
            raise typer.Exit(1)

        if agent_file.suffix != ".md":
            console.print(f"[red]Agent file must be a .md file: {agent_path}[/red]")
            raise typer.Exit(1)

        yield agent_file.resolve()

    finally:
        if original_cwd:
            os.chdir(original_cwd)
