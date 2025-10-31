"""Common utilities for Tsugite."""

import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml


def parse_yaml_frontmatter(content: str, label: str = "content") -> Tuple[Dict[str, Any], str]:
    """Parse YAML frontmatter from markdown content.

    Args:
        content: Markdown content with YAML frontmatter
        label: Description of content type for error messages

    Returns:
        Tuple of (metadata dict, markdown content)

    Raises:
        ValueError: If frontmatter is missing or invalid
    """
    if not content.startswith("---"):
        raise ValueError(f"{label} must start with YAML frontmatter")

    parts = content.split("---", 2)
    if len(parts) < 3:
        raise ValueError(f"Invalid YAML frontmatter format in {label.lower()}")

    try:
        metadata = yaml.safe_load(parts[1]) or {}
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML frontmatter in {label.lower()}: {e}") from e

    markdown_content = parts[2].strip()
    return metadata, markdown_content


def standardize_error_message(operation: str, target: str, error: Exception) -> str:
    return f"Failed to {operation} {target}: {error}"


def tool_error(tool_name: str, operation: str, details: str) -> RuntimeError:
    return RuntimeError(f"Tool '{tool_name}' failed to {operation}: {details}")


def validation_error(item_type: str, item_name: str, issue: str) -> ValueError:
    return ValueError(f"Invalid {item_type} '{item_name}': {issue}")


def execute_shell_command(command: str, timeout: int = 30, shell: bool = True) -> str:
    """Execute a shell command and return its output.

    Args:
        command: Shell command to execute
        timeout: Maximum execution time in seconds
        shell: Whether to use shell execution

    Returns:
        Command output including stdout, stderr, and exit code

    Raises:
        RuntimeError: If command execution fails or times out
    """
    try:
        if shell:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        else:
            cmd_parts = shlex.split(command)
            result = subprocess.run(
                cmd_parts,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )

        output = ""
        if result.stdout:
            output += result.stdout
        if result.stderr:
            if output:
                output += "\n" + result.stderr
            else:
                output = result.stderr

        if result.returncode != 0:
            output += f"\n[Exit code: {result.returncode}]"

        return output or "[No output]"

    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"Command timed out after {timeout} seconds") from exc
    except Exception as e:
        raise RuntimeError(f"Command execution failed: {e}") from e


def is_interactive() -> bool:
    """Check if running in an interactive terminal (TTY).

    Returns:
        True if running in an interactive terminal, False otherwise
    """
    return sys.stdin.isatty()


def has_stdin_data() -> bool:
    """Check if stdin has data available (pipe or redirect).

    Returns:
        True if stdin has data, False if interactive terminal or no data
    """
    import select

    if sys.stdin.isatty():
        return False

    ready, _, _ = select.select([sys.stdin], [], [], 0.0)
    return bool(ready)


def read_stdin() -> str:
    """Read all data from stdin.

    Returns:
        Content from stdin as string
    """
    return sys.stdin.read()


def should_use_plain_output() -> bool:
    """Detect if plain output mode should be used (no panels/boxes).

    Plain output is used when:
    - NO_COLOR environment variable is set
    - stdout is not a TTY (output is piped/redirected)

    Returns:
        True if plain output should be used, False otherwise
    """
    if os.environ.get("NO_COLOR"):
        return True

    if not sys.stdout.isatty():
        return True

    return False


def ensure_file_exists(path: Path, context: str = "File") -> Path:
    """Ensure a file exists and return its resolved path.

    Args:
        path: Path to validate
        context: Context for error message (e.g., "Agent file", "Config file")

    Returns:
        Resolved absolute path

    Raises:
        ValueError: If path doesn't exist or is not a file
    """
    if not path.exists():
        raise ValueError(f"{context} not found: {path}")
    if not path.is_file():
        raise ValueError(f"{context} is not a file: {path}")
    return path.resolve()


def ensure_dir_exists(path: Path, context: str = "Directory") -> Path:
    """Ensure a directory exists and return its resolved path.

    Args:
        path: Path to validate
        context: Context for error message (e.g., "Working directory", "Cache directory")

    Returns:
        Resolved absolute path

    Raises:
        ValueError: If path doesn't exist or is not a directory
    """
    if not path.exists():
        raise ValueError(f"{context} not found: {path}")
    if not path.is_dir():
        raise ValueError(f"{context} is not a directory: {path}")
    return path.resolve()


def resolve_attachments(attachment_refs: List[str], refresh_cache: bool = False) -> List[Tuple[str, str]]:
    """Resolve attachment references to their content using handler system.

    Args:
        attachment_refs: List of attachment aliases
        refresh_cache: If True, bypass cache and re-fetch content

    Returns:
        List of (alias, content) tuples

    Raises:
        ValueError: If an attachment cannot be resolved
    """
    from tsugite.attachments import get_attachment, get_handler
    from tsugite.cache import get_cached_content, save_to_cache

    resolved = []

    for ref in attachment_refs:
        # Get attachment from registry
        result = get_attachment(ref)
        handler = None

        # If not in registry, try to find a handler that can handle it directly
        if result is None:
            try:
                handler = get_handler(ref)
                # Handler found, use ref as source
                source = ref
                content = None
            except ValueError:
                # No handler found either
                raise ValueError(f"Attachment not found: '{ref}'")
        else:
            source, content = result

        # If inline content, use it directly
        if content is not None:
            resolved.append((ref, content))
            continue

        # For file/URL references, check cache first
        if not refresh_cache:
            cached = get_cached_content(source)
            if cached:
                resolved.append((ref, cached))
                continue

        # Fetch content via handler
        try:
            if handler is None:
                handler = get_handler(source)

            # Check if handler supports multiple attachments (like AutoContextHandler)
            if hasattr(handler, "fetch_multiple"):
                # Fetch multiple attachments and add all to resolved list
                multiple_attachments = handler.fetch_multiple(source)
                for name, content in multiple_attachments:
                    # Cache each attachment separately
                    cache_key = f"{source}:{name}"
                    save_to_cache(cache_key, content)
                    resolved.append((name, content))
            else:
                # Single attachment - use normal fetch
                fetched_content = handler.fetch(source)

                # Save to cache
                save_to_cache(source, fetched_content)

                resolved.append((ref, fetched_content))
        except Exception as e:
            raise ValueError(f"Failed to fetch attachment '{ref}' from {source}: {e}") from e

    return resolved


def expand_file_references(prompt: str, base_dir: Path) -> Tuple[str, List[Tuple[str, str]]]:
    """Expand @filename references in prompt by reading file contents.

    Finds patterns like @filename or @"path with spaces.txt", reads their contents,
    and returns them as attachment tuples. The @filename references in the prompt are
    replaced with just the filename (without @).

    Args:
        prompt: User prompt potentially containing @filename references
        base_dir: Base directory to resolve relative paths from

    Returns:
        Tuple of (updated_prompt, list_of_file_attachment_tuples)
        where each tuple is (relative_path, content)

    Raises:
        ValueError: If a referenced file cannot be read

    Examples:
        >>> expand_file_references("Analyze @test.py", Path("/tmp"))
        ("Analyze test.py", [("test.py", "code content")])
    """
    # Pattern matches @filename or @"quoted filename"
    # Group 1: quoted path, Group 2: unquoted path
    # Unquoted paths must start with valid filename characters (not special symbols like #, $, etc.)
    pattern = r'@(?:"([^"]+)"|([a-zA-Z0-9_./\-][^\s]*))'

    file_attachments = []

    def collect_file_ref(match: re.Match) -> str:
        # Extract filename from either quoted or unquoted group
        filename = match.group(1) or match.group(2)
        file_path = Path(filename)

        # Resolve relative paths from base_dir
        if not file_path.is_absolute():
            file_path = base_dir / file_path

        # Check if file exists and is readable
        if not file_path.exists():
            raise ValueError(f"File not found: {filename}")

        if not file_path.is_file():
            raise ValueError(f"Path is not a file: {filename}")

        try:
            content = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"File is not a text file or has encoding issues: {filename}") from exc
        except PermissionError as exc:
            raise ValueError(f"Permission denied reading file: {filename}") from exc
        except Exception as e:
            raise ValueError(f"Failed to read file {filename}: {e}") from e

        # Store as attachment tuple (relative path, content)
        file_attachments.append((filename, content))

        # Replace @filename with just the filename in the prompt
        return filename

    # Replace all @filename references and collect contents as tuples
    updated_prompt = re.sub(pattern, collect_file_ref, prompt)

    return updated_prompt, file_attachments


async def cleanup_pending_tasks() -> None:
    """Clean up any pending asyncio tasks.

    This is used to properly clean up background tasks (like LiteLLM's logging tasks)
    before the event loop shuts down, preventing RuntimeWarning about pending tasks.

    Should be called in finally blocks of async functions that use asyncio.run().
    """
    import asyncio

    # Get all tasks except the current one
    current_task = asyncio.current_task()
    all_tasks = asyncio.all_tasks()
    pending_tasks = [task for task in all_tasks if task is not current_task and not task.done()]

    if not pending_tasks:
        return

    # Cancel all pending tasks
    for task in pending_tasks:
        task.cancel()

    # Wait for all tasks to be cancelled
    # Use return_exceptions=True to suppress CancelledError
    await asyncio.gather(*pending_tasks, return_exceptions=True)
