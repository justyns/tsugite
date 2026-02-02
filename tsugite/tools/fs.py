"""File system tools for Tsugite agents."""

from pathlib import Path
from typing import Any, Dict, List, Optional

import pathspec

from ..tools import tool


@tool
def read_file(path: str, start_line: Optional[int] = None, end_line: Optional[int] = None) -> str:
    """Read content from a file, optionally with line range.

    Args:
        path: Path to the file to read
        start_line: Starting line number (1-indexed, 0 also accepted). If provided, returns numbered lines.
        end_line: Ending line number (1-indexed, inclusive). Defaults to end of file.

    Returns:
        If start_line is None: Full file content as plain text
        If start_line is provided: Numbered lines in format "LINE_NUM: content"
    """
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if file_path.is_dir():
        raise IsADirectoryError(f"Path is a directory: {path}")

    try:
        if start_line is not None:
            # Accept 0-based indexing (treat 0 as 1 for convenience)
            if start_line < 0:
                raise ValueError("start_line must be >= 0")
            if start_line == 0:
                start_line = 1

            if end_line is not None and end_line < start_line:
                raise ValueError(f"end_line ({end_line}) must be >= start_line ({start_line})")

        content = file_path.read_text(encoding="utf-8")

        from tsugite.events.helpers import emit_file_read_event

        emit_file_read_event(str(file_path), content, "tool_call")

        # If no line range specified, return entire file (backward compatible)
        if start_line is None:
            return content

        lines = content.splitlines()
        total_lines = len(lines)

        # Adjust end_line if not specified or beyond file length
        if end_line is None:
            end_line = total_lines
        else:
            end_line = min(end_line, total_lines)

        # Extract requested range (convert to 0-indexed)
        start_idx = start_line - 1
        end_idx = end_line

        if start_idx >= total_lines:
            return f"File only has {total_lines} lines, but start_line is {start_line}"

        selected_lines = lines[start_idx:end_idx]

        # Format with line numbers
        formatted_lines = [f"{i + start_line}: {line}" for i, line in enumerate(selected_lines)]

        return "\n".join(formatted_lines)

    except Exception as e:
        if isinstance(e, (FileNotFoundError, IsADirectoryError, ValueError)):
            raise
        raise RuntimeError(f"Failed to read file {path}: {e}") from e


@tool
def write_file(path: str, content: str) -> str:
    """Write content to a file.

    Args:
        path: Path to the file to write
        content: Content to write to the file
    """
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        file_path.write_text(content, encoding="utf-8")
        return f"Successfully wrote {len(content)} characters to {path}"
    except Exception as e:
        raise RuntimeError(f"Failed to write file {path}: {e}") from e


def _build_gitignore_matcher(base_path: Path) -> Optional[pathspec.PathSpec]:
    """Build a pathspec matcher from .gitignore files.

    Walks up the directory tree from base_path to find .gitignore files
    and combines their patterns into a single matcher.

    Args:
        base_path: Starting directory path

    Returns:
        PathSpec matcher or None if no .gitignore files found
    """
    patterns = []
    current = base_path.resolve()

    # Walk up the directory tree to find .gitignore files
    while True:
        gitignore_path = current / ".gitignore"
        if gitignore_path.exists() and gitignore_path.is_file():
            try:
                with gitignore_path.open("r", encoding="utf-8") as f:
                    # Parse gitignore patterns (skip empty lines and comments)
                    for line in f:
                        line = line.rstrip("\n\r")
                        if line and not line.startswith("#"):
                            patterns.append(line)
            except Exception:
                pass

        # Stop at git root (if .git exists) or filesystem root
        if (current / ".git").exists() or current.parent == current:
            break

        current = current.parent

    # Always exclude .git/ directory
    patterns.append(".git/")

    if not patterns:
        return None

    return pathspec.PathSpec.from_lines(pathspec.patterns.GitWildMatchPattern, patterns)


@tool
def list_files(path: str = ".", pattern: str = "*", respect_gitignore: bool = True) -> List[str]:
    """List files in a directory with optional pattern matching.

    Args:
        path: Directory path to list files from
        pattern: Glob pattern to match files
        respect_gitignore: If True (default), respects .gitignore files and excludes .git/ directory.
                          Follows the behavior of modern tools like ripgrep and fd.
    """
    dir_path = Path(path)

    if not dir_path.exists():
        raise FileNotFoundError(f"Directory not found: {path}")

    if not dir_path.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {path}")

    try:
        # Build gitignore matcher if requested
        gitignore_spec = None
        if respect_gitignore:
            gitignore_spec = _build_gitignore_matcher(dir_path)

        files = []
        for item in dir_path.glob(pattern):
            if item.is_file():
                rel_path = str(item.relative_to(dir_path))

                # Filter through gitignore if enabled
                if gitignore_spec and gitignore_spec.match_file(rel_path):
                    continue

                files.append(rel_path)

        return sorted(files)
    except Exception as e:
        raise RuntimeError(f"Failed to list files in directory {path}: {e}") from e


@tool
def file_exists(path: str) -> bool:
    """Check if a file exists.

    Args:
        path: Path to check for existence
    """
    return Path(path).exists()


@tool
def create_directory(path: str) -> str:
    """Create a directory and any necessary parent directories.

    Args:
        path: Directory path to create
    """
    dir_path = Path(path)

    try:
        dir_path.mkdir(parents=True, exist_ok=True)
        return f"Successfully created directory: {path}"
    except Exception as e:
        raise RuntimeError(f"Failed to create directory {path}: {e}") from e


@tool
def get_file_info(path: str) -> Dict[str, Any]:
    """Get file metadata without reading full content.

    Args:
        path: Path to the file

    Returns:
        Dictionary with file metadata:
        - line_count: Total number of lines
        - size_bytes: File size in bytes
        - last_modified: Last modification timestamp (ISO format)
        - exists: Whether file exists
        - is_directory: Whether path is a directory
    """
    import datetime

    file_path = Path(path)

    info = {
        "exists": file_path.exists(),
        "is_directory": False,
        "line_count": 0,
        "size_bytes": 0,
        "last_modified": None,
    }

    if not file_path.exists():
        return info

    info["is_directory"] = file_path.is_dir()

    if file_path.is_dir():
        return info

    try:
        # Get file stats
        stats = file_path.stat()
        info["size_bytes"] = stats.st_size
        info["last_modified"] = datetime.datetime.fromtimestamp(stats.st_mtime).isoformat()

        # Count lines
        content = file_path.read_text(encoding="utf-8")
        info["line_count"] = len(content.splitlines())

        return info

    except Exception as e:
        raise RuntimeError(f"Failed to get info for file {path}: {e}") from e


def _detect_line_ending(content: str) -> str:
    """Detect line ending style in content."""
    return "\r\n" if "\r\n" in content else "\n"


def _preserve_line_ending(original_content: str, modified_content: str) -> str:
    """Preserve original line endings in modified content."""
    original_ending = _detect_line_ending(original_content)
    if original_ending == "\r\n" and "\r\n" not in modified_content:
        return modified_content.replace("\n", "\r\n")
    if original_ending == "\n" and "\r\n" in modified_content:
        return modified_content.replace("\r\n", "\n")
    return modified_content


def _apply_exact_replacement(
    content: str, search: str, replace: str, expected_count: int
) -> tuple[str, int, str | None]:
    """Apply exact string replacement.

    Returns:
        Tuple of (new_content, match_count, error_message)
        error_message is None on success
    """
    if not search:
        return content, 0, "Search string cannot be empty"

    if search == replace:
        return content, 0, "Search and replace strings must be different"

    match_count = content.count(search)

    if match_count == 0:
        return (
            content,
            0,
            "No matches found. Ensure old_string matches file content exactly (including whitespace/indentation). "
            "Use read_file to verify.",
        )

    if match_count != expected_count:
        if expected_count == 1:
            error = (
                f"Found {match_count} matches but expected 1. "
                "Either add more context to make the match unique or use expected_replacements parameter."
            )
        else:
            error = f"Found {match_count} matches but expected {expected_count}."
        return content, match_count, error

    new_content = content.replace(search, replace)
    return new_content, match_count, None


@tool
def edit_file(
    path: str,
    old_string: Optional[str] = None,
    new_string: Optional[str] = None,
    expected_replacements: int = 1,
    edits: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Edit a file with single or multiple exact string replacements.

    Two modes of operation:

    **Single edit mode** - Use old_string and new_string:
    - Applies one exact string replacement

    **Batch edit mode** - Use edits parameter:
    - Apply multiple edits sequentially
    - Atomic: if any edit fails, none are applied
    - Each edit operates on the result of the previous edit

    Args:
        path: Path to the file to edit
        old_string: Text to find (for single edit mode)
        new_string: Replacement text (for single edit mode)
        expected_replacements: Expected match count (default: 1, for single edit mode)
        edits: List of edit dicts (for batch edit mode)
            Each dict: {"old_string": str, "new_string": str, "expected_replacements": int}

    Returns:
        Success message with number of replacements made

    Examples:
        Single edit:
            edit_file("config.py", old_string="DEBUG = True", new_string="DEBUG = False")

        Batch edits:
            edit_file("config.py", edits=[
                {"old_string": "DEBUG = True", "new_string": "DEBUG = False"},
                {"old_string": "TIMEOUT = 30", "new_string": "TIMEOUT = 60"}
            ])

    Raises:
        ValueError: If parameters are invalid or conflicting
        RuntimeError: If edits fail
    """
    single_mode = old_string is not None
    batch_mode = edits is not None

    if single_mode and batch_mode:
        raise ValueError("Provide either old_string/new_string OR edits, not both")
    if not single_mode and not batch_mode:
        raise ValueError("Must provide either old_string/new_string OR edits")

    if single_mode and new_string is None:
        raise ValueError("new_string is required when using old_string")

    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if file_path.is_dir():
        raise IsADirectoryError(f"Path is a directory: {path}")

    try:
        original_content = file_path.read_text(encoding="utf-8")
        current_content = original_content.replace("\r\n", "\n")

        if single_mode:
            normalized_old = old_string.replace("\r\n", "\n")
            normalized_new = new_string.replace("\r\n", "\n")

            new_content, match_count, error = _apply_exact_replacement(
                current_content, normalized_old, normalized_new, expected_replacements
            )

            if error:
                raise RuntimeError(f"Failed to edit {path}: {error}")

            total_edits = 1
            total_replacements = match_count

        else:
            if not edits:
                raise ValueError("edits list cannot be empty")

            total_replacements = 0
            for i, edit in enumerate(edits, 1):
                if "old_string" not in edit or "new_string" not in edit:
                    raise ValueError(f"Edit #{i} missing required 'old_string' or 'new_string'")

                old_str = edit["old_string"].replace("\r\n", "\n")
                new_str = edit["new_string"].replace("\r\n", "\n")
                expected = edit.get("expected_replacements", 1)

                current_content, match_count, error = _apply_exact_replacement(
                    current_content, old_str, new_str, expected
                )

                if error:
                    raise RuntimeError(f"Edit #{i} failed: {error}")

                total_replacements += match_count

            new_content = current_content
            total_edits = len(edits)

        final_content = _preserve_line_ending(original_content, new_content)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(final_content, encoding="utf-8")

        if batch_mode:
            return f"Successfully applied {total_edits} edit(s) to {path} ({total_replacements} total replacements)"
        else:
            return f"Successfully edited {path}: {total_replacements} replacement(s) made"

    except Exception as e:
        if isinstance(e, (FileNotFoundError, IsADirectoryError, RuntimeError, ValueError)):
            raise
        raise RuntimeError(f"Failed to edit file {path}: {e}") from e
