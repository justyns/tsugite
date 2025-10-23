"""CLI helper functions."""

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, List, Optional, Tuple

import typer
from rich.console import Console

# Re-export console utilities for backwards compatibility
from tsugite.console import get_error_console, get_output_console  # noqa: F401
from tsugite.constants import TSUGITE_LOGO_NARROW, TSUGITE_LOGO_WIDE


def get_logo(console: Console) -> str:
    """Get appropriate logo based on terminal width."""
    return TSUGITE_LOGO_NARROW if console.width < 80 else TSUGITE_LOGO_WIDE


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


def resolve_attachments_with_error_handling(
    attachments: List[str],
    base_dir: Path,
    refresh_cache: bool,
    console: Console,
    error_context: str = "Attachment",
) -> List[Tuple[str, str]]:
    """Resolve attachments with error handling.

    Args:
        attachments: List of attachment names/paths
        base_dir: Base directory for resolving paths
        refresh_cache: Whether to refresh cached content
        console: Console for error messages
        error_context: Context for error message (e.g., "Agent attachment" or "Attachment")

    Returns:
        List of (name, content) tuples

    Raises:
        typer.Exit: If attachment resolution fails
    """
    from tsugite.utils import resolve_attachments

    try:
        return resolve_attachments(attachments, base_dir, refresh_cache)
    except ValueError as e:
        console.print(f"[red]{error_context} error: {e}[/red]")
        raise typer.Exit(1)


def assemble_prompt_with_attachments(
    prompt: str,
    agent_attachments: Optional[List[str]],
    cli_attachments: Optional[List[str]],
    base_dir: Path,
    refresh_cache: bool,
    console: Console,
) -> Tuple[str, List[str]]:
    """Resolve all attachments and assemble final prompt with proper ordering.

    Args:
        prompt: Base prompt text
        agent_attachments: Attachments from agent definition
        cli_attachments: Attachments from CLI (-f flag)
        base_dir: Base directory for resolving paths
        refresh_cache: Whether to refresh cached content
        console: Console for error messages

    Returns:
        Tuple of (assembled_prompt, expanded_file_list)

    Raises:
        typer.Exit: If attachment or file reference resolution fails
    """
    from tsugite.utils import expand_file_references

    # Resolve agent attachments
    agent_attachment_contents = (
        resolve_attachments_with_error_handling(agent_attachments, base_dir, refresh_cache, console, "Agent attachment")
        if agent_attachments
        else []
    )

    # Resolve CLI attachments
    cli_attachment_contents = (
        resolve_attachments_with_error_handling(cli_attachments, base_dir, refresh_cache, console, "Attachment")
        if cli_attachments
        else []
    )

    # Expand @filename references in prompt
    try:
        prompt, expanded_files = expand_file_references(prompt, base_dir)
    except ValueError as e:
        console.print(f"[red]File reference error: {e}[/red]")
        raise typer.Exit(1)

    # Assemble all attachments in proper order: agent -> CLI -> file refs -> prompt
    all_attachments = agent_attachment_contents + cli_attachment_contents

    if all_attachments:
        attachment_sections = [
            f"<Attachment: {name}>\n{content}\n</Attachment: {name}>" for name, content in all_attachments
        ]
        prompt = "\n\n".join(attachment_sections) + "\n\n" + prompt

    return prompt, expanded_files


def load_and_validate_agent(agent_path: str, console: Console) -> Tuple[Any, Path, str]:
    """Load and validate an agent from path or builtin name.

    Consolidates agent loading logic used across run, render, and chat commands.
    Handles both builtin agents (e.g., "+builtin-default") and file-based agents.

    Args:
        agent_path: Path to agent file or builtin reference (e.g., "+builtin-default", "agent.md")
        console: Console for error messages

    Returns:
        Tuple of (agent_object, agent_file_path, display_name)

    Raises:
        typer.Exit: If agent cannot be loaded or validated

    Examples:
        >>> agent, path, name = load_and_validate_agent("+builtin-default", console)
        >>> agent, path, name = load_and_validate_agent("agents/my_agent.md", console)
    """
    from tsugite.agent_composition import resolve_agent_reference
    from tsugite.builtin_agents import (
        get_builtin_chat_assistant,
        get_builtin_default_agent,
        is_builtin_agent_path,
    )
    from tsugite.md_agents import parse_agent_file

    # Use resolve_agent_reference to handle +name shorthand and builtin agents
    try:
        base_dir = Path.cwd()
        resolved_path = resolve_agent_reference(agent_path, base_dir)
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)

    # Check if this is a builtin agent path
    if is_builtin_agent_path(resolved_path):
        agent_name = str(resolved_path).strip("<>")

        # Get the valid builtin agent
        if agent_name == "builtin-default":
            agent = get_builtin_default_agent()
        elif agent_name == "builtin-chat-assistant":
            agent = get_builtin_chat_assistant()
        else:
            # Should never reach here due to resolve_agent_reference validation
            console.print(f"[red]Unknown builtin agent: {agent_name}[/red]")
            raise typer.Exit(1)

        agent_display_name = agent_name
        agent_file_path = resolved_path
    else:
        # Regular file-based agent
        agent_file_path = resolved_path
        if not agent_file_path.exists():
            console.print(f"[red]Agent file not found: {agent_file_path}[/red]")
            raise typer.Exit(1)

        if agent_file_path.suffix != ".md":
            console.print(f"[red]Agent file must be a .md file: {agent_file_path}[/red]")
            raise typer.Exit(1)

        # Use parse_agent_file to properly resolve inheritance
        agent = parse_agent_file(agent_file_path)
        agent_display_name = agent_file_path.name

    return agent, agent_file_path, agent_display_name


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

    # Check if common CLI options appear in positional args (common user error)
    common_options = [
        "--ui",
        "--model",
        "--verbose",
        "--debug",
        "--silent",
        "--headless",
        "--plain",
        "--stream",
        "--native-ui",
        "--non-interactive",
        "--no-color",
        "--show-reasoning",
        "--no-show-reasoning",
        "--trust-mcp-code",
        "--attachment",
        "-f",
        "--with-agents",
        "--root",
        "--history-dir",
        "--log-json",
        "--dry-run",
        "--refresh-cache",
        "--docker",
        "--keep",
        "--container",
        "--network",
    ]

    misplaced_options = [arg for arg in args if arg in common_options]
    if misplaced_options:
        option_str = ", ".join(misplaced_options)
        raise ValueError(
            f"Options must come before the prompt or agent name.\n"
            f"Found: {option_str}\n\n"
            f"Correct usage:\n"
            f'  tsugite run --ui minimal +agent "prompt"\n'
            f'  tsugite run +agent "prompt" --ui minimal\n\n'
            f"Incorrect:\n"
            f'  tsugite run +agent --ui minimal "prompt"'
        )

    agents = []
    prompt_parts = []

    for i, arg in enumerate(args):
        # Check if this looks like an agent reference
        # Exclude arguments containing @ (file references) from being treated as agents
        has_file_reference = "@" in arg
        has_path_separator = "/" in arg
        has_spaces = " " in arg

        # Agent detection logic:
        # - Starts with + (shorthand like +agent) OR
        # - Ends with .md (agent file) BUT no spaces (to avoid "text in file.md" prompts) OR
        # - Has path separator BUT no spaces (file path like path/to/agent.md)
        # - Must not contain @ (file reference marker)
        is_agent = (
            arg.startswith("+") or (arg.endswith(".md") and not has_spaces) or (has_path_separator and not has_spaces)
        ) and not has_file_reference

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
def change_to_root_directory(root: Optional[str], console: Console):
    """Context manager for temporarily changing to a root directory.

    Args:
        root: Optional path to root directory
        console: Console for error messages

    Yields:
        None

    Raises:
        typer.Exit: If root directory doesn't exist
    """
    original_cwd = None

    try:
        if root:
            root_path = Path(root)
            if not root_path.exists():
                console.print(f"[red]Working directory not found: {root}[/red]")
                raise typer.Exit(1)
            original_cwd = os.getcwd()
            os.chdir(str(root_path))

        yield

    finally:
        if original_cwd:
            os.chdir(original_cwd)


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
