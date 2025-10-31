"""File system tools for Tsugite agents."""

from pathlib import Path
from typing import Any, Dict, List, Optional

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
        # If no line range specified, read entire file (backward compatible)
        if start_line is None:
            return file_path.read_text(encoding="utf-8")

        # Line range mode - return numbered lines
        # Accept 0-based indexing (treat 0 as 1 for convenience)
        if start_line < 0:
            raise ValueError("start_line must be >= 0")
        if start_line == 0:
            start_line = 1

        if end_line is not None and end_line < start_line:
            raise ValueError(f"end_line ({end_line}) must be >= start_line ({start_line})")

        lines = file_path.read_text(encoding="utf-8").splitlines()
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


@tool
def list_files(path: str = ".", pattern: str = "*") -> List[str]:
    """List files in a directory with optional pattern matching.

    Args:
        path: Directory path to list files from
        pattern: Glob pattern to match files
    """
    dir_path = Path(path)

    if not dir_path.exists():
        raise FileNotFoundError(f"Directory not found: {path}")

    if not dir_path.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {path}")

    try:
        files = []
        for item in dir_path.glob(pattern):
            if item.is_file():
                files.append(str(item.relative_to(dir_path)))

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


@tool
def edit_file(
    path: str,
    old_string: Optional[str] = None,
    new_string: Optional[str] = None,
    expected_replacements: int = 1,
    edits: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Edit a file with single or multiple replacements.

    Two modes of operation:

    **Single edit mode** - Use old_string and new_string:
    - Applies one replacement with smart matching strategies
    - Strategies: exact, line-trimmed, block-anchor, whitespace-normalized, indentation-flexible

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
    from .edit_strategies import apply_replacement, preserve_line_ending

    # Validate mode selection
    single_mode = old_string is not None
    batch_mode = edits is not None

    if single_mode and batch_mode:
        raise ValueError("Provide either old_string/new_string OR edits, not both")
    if not single_mode and not batch_mode:
        raise ValueError("Must provide either old_string/new_string OR edits")

    if single_mode and new_string is None:
        raise ValueError("new_string is required when using old_string")

    # Validate file
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if file_path.is_dir():
        raise IsADirectoryError(f"Path is a directory: {path}")

    try:
        # Read original content
        original_content = file_path.read_text(encoding="utf-8")

        # Normalize to \n for processing
        current_content = original_content.replace("\r\n", "\n")

        if single_mode:
            # Single edit mode
            normalized_old = old_string.replace("\r\n", "\n")
            normalized_new = new_string.replace("\r\n", "\n")

            new_content, match_count, error = apply_replacement(
                current_content, normalized_old, normalized_new, expected_replacements
            )

            if error:
                raise RuntimeError(f"Failed to edit {path}: {error}")

            total_edits = 1
            total_replacements = match_count

        else:
            # Batch edit mode
            if not edits:
                raise ValueError("edits list cannot be empty")

            total_replacements = 0
            for i, edit in enumerate(edits, 1):
                if "old_string" not in edit or "new_string" not in edit:
                    raise ValueError(f"Edit #{i} missing required 'old_string' or 'new_string'")

                old_str = edit["old_string"].replace("\r\n", "\n")
                new_str = edit["new_string"].replace("\r\n", "\n")
                expected = edit.get("expected_replacements", 1)

                current_content, match_count, error = apply_replacement(current_content, old_str, new_str, expected)

                if error:
                    raise RuntimeError(f"Edit #{i} failed: {error}")

                total_replacements += match_count

            new_content = current_content
            total_edits = len(edits)

        # Restore original line endings
        final_content = preserve_line_ending(original_content, new_content)

        # Create parent directories if needed
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Write updated content
        file_path.write_text(final_content, encoding="utf-8")

        if batch_mode:
            return f"Successfully applied {total_edits} edit(s) to {path} ({total_replacements} total replacements)"
        else:
            return f"Successfully edited {path}: {total_replacements} replacement(s) made"

    except Exception as e:
        if isinstance(e, (FileNotFoundError, IsADirectoryError, RuntimeError, ValueError)):
            raise
        raise RuntimeError(f"Failed to edit file {path}: {e}") from e
