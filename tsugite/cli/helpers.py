"""CLI helper functions."""

import hashlib
import io
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, List, Optional, Tuple

import typer
from rich.console import Console

# Re-export console utilities for backwards compatibility
from tsugite.console import get_error_console, get_output_console  # noqa: F401
from tsugite.constants import TSUGITE_LOGO_NARROW, TSUGITE_LOGO_WIDE

MIN_WIDTH_FOR_WIDE_LOGO = 80
STDIN_ATTACHMENT_NAME = "stdin"


def deduplicate_attachments(attachments: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """Deduplicate attachments by canonical path and content hash.

    This prevents the same file from being sent multiple times to the LLM when:
    - Referenced through different paths (symlinks, relative vs absolute)
    - Specified multiple times across different sources (agent, CLI, file refs)
    - Identical content with different names (renamed/moved copies)

    Args:
        attachments: List of (name, content) tuples

    Returns:
        Deduplicated list of (name, content) tuples where duplicate files are combined
        with their aliases shown in the name: "file.txt (also: symlink.txt, @other.txt)"

    Examples:
        >>> deduplicate_attachments([("file.txt", "content"), ("symlink.txt", "content")])
        [("file.txt (also: symlink.txt)", "content")]
    """
    seen_paths = {}  # canonical_path -> {name, content, aliases, order}
    seen_hashes = {}  # content_hash -> canonical_path
    order_counter = 0

    for name, content in attachments:
        # Try to resolve as file path
        canonical = None
        try:
            # Attempt to resolve path (handles symlinks and relative paths)
            path = Path(name)
            if path.exists():
                canonical = str(path.resolve())
        except (OSError, ValueError):
            # Not a valid file path or can't resolve - treat as non-path attachment
            pass

        # If we couldn't resolve to a canonical path, use name as-is for deduplication
        if canonical is None:
            canonical = name

        # Check if already seen by path
        if canonical in seen_paths:
            # Add this name as an alias
            seen_paths[canonical]["aliases"].append(name)
            continue

        # Check by content hash (catches renamed/moved copies)
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        if content_hash in seen_hashes:
            existing_canonical = seen_hashes[content_hash]
            seen_paths[existing_canonical]["aliases"].append(name)
            continue

        # New unique attachment - preserve order of first occurrence
        seen_paths[canonical] = {"name": name, "content": content, "aliases": [], "order": order_counter}
        seen_hashes[content_hash] = canonical
        order_counter += 1

    # Build result preserving order of first occurrence
    result = []
    for entry in sorted(seen_paths.values(), key=lambda x: x["order"]):
        if entry["aliases"]:
            # Combine name with aliases
            aliases_str = ", ".join(entry["aliases"])
            combined_name = f"{entry['name']} (also: {aliases_str})"
        else:
            combined_name = entry["name"]
        result.append((combined_name, entry["content"]))

    return result


def get_logo(console: Console) -> str:
    """Get appropriate logo based on terminal width."""
    return TSUGITE_LOGO_NARROW if console.width < MIN_WIDTH_FOR_WIDE_LOGO else TSUGITE_LOGO_WIDE


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
    # Use simple header if no_color is enabled (to avoid ANSI codes from rule)
    if console.no_color:
        console.print(title)
        console.print("-" * len(title))
    else:
        console.rule(f"[bold]{title}[/bold]", style="dim")
    for label, value in items.items():
        if style and not console.no_color:
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
        return resolve_attachments(attachments, refresh_cache)
    except ValueError as e:
        console.print(f"[red]{error_context} error: {e}[/red]")
        raise typer.Exit(1)


def inject_auto_context_if_enabled(
    agent_attachments: Optional[List[str]],
    agent_auto_context: Optional[bool],
    cli_override: Optional[bool] = None,
) -> Optional[List[str]]:
    """Inject auto-context attachment if enabled in config or agent.

    Args:
        agent_attachments: Current agent attachments list
        agent_auto_context: Agent's auto_context setting (None = use config default)
        cli_override: CLI flag override (None = use precedence, True/False = force)

    Returns:
        Updated attachments list with auto-context prepended if enabled, or original list
    """
    from tsugite.config import load_config

    config = load_config()

    # Determine if auto-context should be enabled
    # Priority: CLI override > agent setting > config default
    if cli_override is not None:
        should_enable = cli_override
    elif agent_auto_context is not None:
        should_enable = agent_auto_context
    else:
        should_enable = config.auto_context_enabled

    if not should_enable:
        return agent_attachments

    # Prepend auto-context to attachments list
    attachments = list(agent_attachments) if agent_attachments else []

    # Only add if not already present
    if "auto-context" not in attachments:
        attachments.insert(0, "auto-context")

    return attachments


def assemble_prompt_with_attachments(
    prompt: str,
    agent_attachments: Optional[List[str]],
    cli_attachments: Optional[List[str]],
    base_dir: Path,
    refresh_cache: bool,
    console: Console,
    stdin_attachment: Optional[Tuple[str, str]] = None,
) -> Tuple[str, List[Tuple[str, str]]]:
    """Resolve all attachments and file references, returning combined attachment tuples.

    Args:
        prompt: Base prompt text
        agent_attachments: Attachments from agent definition
        cli_attachments: Attachments from CLI (-f flag)
        base_dir: Base directory for resolving paths
        refresh_cache: Whether to refresh cached content
        console: Console for error messages
        stdin_attachment: Optional stdin content as (name, content) tuple

    Returns:
        Tuple of (updated_prompt, combined_attachment_tuples)
        where attachment_tuples is a list of (name, content) tuples

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

    # Expand @filename references in prompt (returns tuples now)
    try:
        updated_prompt, file_attachment_tuples = expand_file_references(prompt, base_dir)
    except ValueError as e:
        console.print(f"[red]File reference error: {e}[/red]")
        raise typer.Exit(1)

    # Combine all attachments in proper order: agent -> CLI -> file refs -> stdin
    all_attachments = agent_attachment_contents + cli_attachment_contents + file_attachment_tuples

    if stdin_attachment:
        all_attachments.append(stdin_attachment)

    deduplicated_attachments = deduplicate_attachments(all_attachments)

    return updated_prompt, deduplicated_attachments


def load_and_validate_agent(agent_path: str, console: Console) -> Tuple[Any, Path, str]:
    """Load and validate an agent from path or builtin name.

    Consolidates agent loading logic used across run, render, and chat commands.
    Handles both package-provided agents (e.g., "+default") and file-based agents.

    Args:
        agent_path: Path to agent file or agent reference (e.g., "+default", "agent.md")
        console: Console for error messages

    Returns:
        Tuple of (agent_object, agent_file_path, display_name)

    Raises:
        typer.Exit: If agent cannot be loaded or validated

    Examples:
        >>> agent, path, name = load_and_validate_agent("+default", console)
        >>> agent, path, name = load_and_validate_agent("agents/my_agent.md", console)
    """
    from tsugite.agent_composition import resolve_agent_reference
    from tsugite.md_agents import parse_agent_file

    # Use resolve_agent_reference to handle +name shorthand and builtin agents
    try:
        base_dir = Path.cwd()
        resolved_path = resolve_agent_reference(agent_path, base_dir)
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)

    # All agents are now file-based (including built-ins)
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


def _validate_common_option_placement(args: List[str]) -> None:
    """Check if common CLI options appear in positional args (common user error).

    Args:
        args: List of positional arguments from CLI

    Raises:
        ValueError: If common options are found in positional arguments
    """
    common_options = [
        "--ui",
        "--model",
        "--verbose",
        "--debug",
        "--final-only",
        "--quiet",
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


def _parse_agent_refs(args: List[str]) -> tuple[List[str], List[str]]:
    """Parse agent references from arguments.

    Args:
        args: List of positional arguments from CLI

    Returns:
        Tuple of (agent_refs, remaining_prompt_parts)
    """
    agents = []
    prompt_parts = []

    for arg in args:
        has_file_reference = "@" in arg
        has_path_separator = "/" in arg
        has_spaces = " " in arg

        is_agent = (
            arg.startswith("+") or (arg.endswith(".md") and not has_spaces) or (has_path_separator and not has_spaces)
        ) and not has_file_reference

        if is_agent and not prompt_parts:
            agents.append(arg)
        else:
            prompt_parts.append(arg)

    return agents, prompt_parts


def _check_stdin_data() -> Optional[tuple[str, str]]:
    """Check for stdin data and return as attachment if present.

    Returns:
        (STDIN_ATTACHMENT_NAME, content) if stdin has data, None otherwise
    """
    from tsugite.utils import has_stdin_data, read_stdin

    try:
        if has_stdin_data():
            stdin_content = read_stdin()
            if stdin_content.strip():
                return (STDIN_ATTACHMENT_NAME, stdin_content)
    except (OSError, io.UnsupportedOperation):
        # In test environments or special contexts, stdin may not support fileno()
        pass
    return None


def parse_cli_arguments(
    args: List[str], allow_empty_agents: bool = False, check_stdin: bool = True
) -> tuple[List[str], str, Optional[tuple[str, str]]]:
    """Parse CLI arguments into agent references, prompt, and optional stdin.

    Args:
        args: List of positional arguments from CLI
        allow_empty_agents: If True, allow returning empty agent list (for continuation mode)
        check_stdin: If True, check for stdin data and read it

    Returns:
        Tuple of (agent_refs, prompt, stdin_attachment)
        stdin_attachment is None or ("stdin", content)

    Examples:
        ["+a", "+b", "task"] -> (["+a", "+b"], "task", None)
        ["+a", "create", "ticket"] -> (["+a"], "create ticket", None)
        ["agent.md", "helper.md", "do", "work"] -> (["agent.md", "helper.md"], "do work", None)
        ["task"], allow_empty_agents=True -> ([], "task", None)
        ["task"] + stdin data -> (["+default"], "task", ("stdin", "data"))
    """
    if not args:
        raise ValueError("No arguments provided")

    _validate_common_option_placement(args)

    agents, prompt_parts = _parse_agent_refs(args)

    if not agents:
        if allow_empty_agents:
            agents = []
            prompt = " ".join(args)
        else:
            agents = ["+default"]
            prompt = " ".join(args)
    else:
        prompt = " ".join(prompt_parts)

    stdin_attachment = None
    if check_stdin and prompt:
        stdin_attachment = _check_stdin_data()

    return agents, prompt, stdin_attachment


def _validate_and_change_to_root(root: Optional[str], console: Console) -> Optional[str]:
    """Validate root directory and change to it if provided.

    Args:
        root: Optional path to root directory
        console: Console for error messages

    Returns:
        Original working directory path if changed, None otherwise

    Raises:
        typer.Exit: If root directory doesn't exist
    """
    if not root:
        return None

    root_path = Path(root)
    if not root_path.exists():
        console.print(f"[red]Working directory not found: {root}[/red]")
        raise typer.Exit(1)

    original_cwd = os.getcwd()
    os.chdir(str(root_path))
    return original_cwd


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
    original_cwd = _validate_and_change_to_root(root, console)

    try:
        yield
    finally:
        if original_cwd:
            os.chdir(original_cwd)


@contextmanager
def agent_context(agent_path: str, root: Optional[str], console: Console):
    """Validate agent path and optionally change working directory."""
    original_cwd = _validate_and_change_to_root(root, console)

    try:
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
