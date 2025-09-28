"""File system tools for Tsugite agents."""

from pathlib import Path
from typing import List
from ..tools import tool


@tool
def read_file(path: str) -> str:
    """Read content from a file.

    Args:
        path: Path to the file to read
    """
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if file_path.is_dir():
        raise IsADirectoryError(f"Path is a directory: {path}")

    try:
        return file_path.read_text(encoding="utf-8")
    except Exception as e:
        raise RuntimeError(f"Failed to read file {path}: {e}")


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
        raise RuntimeError(f"Failed to write file {path}: {e}")


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
        raise RuntimeError(f"Failed to list files in {path}: {e}")


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
        raise RuntimeError(f"Failed to create directory {path}: {e}")