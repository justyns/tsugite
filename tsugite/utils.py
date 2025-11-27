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


def resolve_attachments(attachment_refs: List[str], refresh_cache: bool = False) -> List["Attachment"]:
    """Resolve attachment references to their content using handler system.

    Args:
        attachment_refs: List of attachment aliases
        refresh_cache: If True, bypass cache and re-fetch content

    Returns:
        List of Attachment objects

    Raises:
        ValueError: If an attachment cannot be resolved
    """
    from tsugite.attachments import get_attachment, get_handler
    from tsugite.attachments.base import Attachment, AttachmentContentType
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

        # If inline content, use it directly as text attachment
        if content is not None:
            resolved.append(
                Attachment(
                    name=ref,
                    content=content,
                    content_type=AttachmentContentType.TEXT,
                    mime_type="text/plain",
                    source_url=None,
                )
            )
            continue

        # For file/URL references, check cache first (for text attachments only)
        if not refresh_cache:
            cached = get_cached_content(source)
            if cached:
                # Cached content is text
                resolved.append(
                    Attachment(
                        name=ref,
                        content=cached,
                        content_type=AttachmentContentType.TEXT,
                        mime_type="text/plain",
                        source_url=None,
                    )
                )
                continue

        # Fetch content via handler
        try:
            if handler is None:
                handler = get_handler(source)

            # Check if handler supports multiple attachments (like AutoContextHandler)
            if hasattr(handler, "fetch_multiple"):
                # Fetch multiple attachments and add all to resolved list
                multiple_attachments = handler.fetch_multiple(source)
                for attachment in multiple_attachments:
                    # Cache text content
                    if attachment.content_type == AttachmentContentType.TEXT and attachment.content:
                        cache_key = f"{source}:{attachment.name}"
                        save_to_cache(cache_key, attachment.content)
                    resolved.append(attachment)
            else:
                # Single attachment - use normal fetch (returns Attachment object)
                fetched_attachment = handler.fetch(source)

                # Cache text content only (binary attachments with URLs don't need caching)
                if fetched_attachment.content_type == AttachmentContentType.TEXT and fetched_attachment.content:
                    save_to_cache(source, fetched_attachment.content)

                resolved.append(fetched_attachment)
        except Exception as e:
            raise ValueError(f"Failed to fetch attachment '{ref}' from {source}: {e}") from e

    return resolved


def expand_file_references(prompt: str, base_dir: Path) -> Tuple[str, List["Attachment"]]:
    """Expand @filename references in prompt by reading file contents.

    Finds patterns like @filename or @"path with spaces.txt", reads their contents,
    and returns them as Attachment objects. The @filename references in the prompt are
    replaced with just the filename (without @).

    Args:
        prompt: User prompt potentially containing @filename references
        base_dir: Base directory to resolve relative paths from

    Returns:
        Tuple of (updated_prompt, list_of_file_attachments)

    Raises:
        ValueError: If a referenced file cannot be read

    Examples:
        >>> expand_file_references("Analyze @test.py", Path("/tmp"))
        ("Analyze test.py", [Attachment(...)])
    """
    from tsugite.attachments.base import Attachment, AttachmentContentType

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

        # Store as Attachment object
        file_attachments.append(
            Attachment(
                name=filename,
                content=content,
                content_type=AttachmentContentType.TEXT,
                mime_type="text/plain",
            )
        )

        # Replace @filename with just the filename in the prompt
        return filename

    # Replace all @filename references and collect contents as Attachments
    updated_prompt = re.sub(pattern, collect_file_ref, prompt)

    return updated_prompt, file_attachments
