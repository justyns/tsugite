"""Common utilities for Tsugite."""

import re
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

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
        raise ValueError(f"Invalid YAML frontmatter in {label.lower()}: {e}")

    markdown_content = parts[2].strip()
    return metadata, markdown_content


def lazy_import(module_path: str, attr_name: str = None, level: int = 1) -> Callable:
    """Create a lazy import function to avoid circular dependencies.

    Args:
        module_path: Python module path (e.g., "..agent_runner")
        attr_name: Specific attribute to import (if None, imports module)
        level: Import level for relative imports

    Returns:
        Function that performs the import when called
    """

    def _import():
        if attr_name:
            if module_path.startswith("."):
                # Relative import
                module = __import__(module_path, fromlist=[attr_name], level=level)
            else:
                # Absolute import
                module = __import__(module_path, fromlist=[attr_name])
            return getattr(module, attr_name)
        else:
            if module_path.startswith("."):
                return __import__(module_path, level=level)
            else:
                return __import__(module_path)

    return _import


def standardize_error_message(operation: str, target: str, error: Exception) -> str:
    return f"Failed to {operation} {target}: {error}"


def tool_error(tool_name: str, operation: str, details: str) -> RuntimeError:
    return RuntimeError(f"Tool '{tool_name}' failed to {operation}: {details}")


def validation_error(item_type: str, item_name: str, issue: str) -> ValueError:
    return ValueError(f"Invalid {item_type} '{item_name}': {issue}")


def is_interactive() -> bool:
    """Check if running in an interactive terminal (TTY).

    Returns:
        True if running in an interactive terminal, False otherwise
    """
    return sys.stdin.isatty()


def expand_file_references(prompt: str, base_dir: Path) -> Tuple[str, List[str]]:
    """Expand @filename references in prompt by reading and injecting file contents.

    Finds patterns like @filename or @"path with spaces.txt", reads their contents,
    and prepends them to the prompt. The @filename references in the prompt are
    replaced with just the filename (without @).

    Args:
        prompt: User prompt potentially containing @filename references
        base_dir: Base directory to resolve relative paths from

    Returns:
        Tuple of (expanded_prompt, list_of_expanded_files)

    Raises:
        ValueError: If a referenced file cannot be read

    Examples:
        >>> expand_file_references("Analyze @test.py", Path("/tmp"))
        ("<File: test.py>\\ncode\\n</File: test.py>\\n\\nAnalyze test.py", ["test.py"])
    """
    # Pattern matches @filename or @"quoted filename"
    # Group 1: quoted path, Group 2: unquoted path
    # Unquoted paths must start with valid filename characters (not special symbols like #, $, etc.)
    pattern = r'@(?:"([^"]+)"|([a-zA-Z0-9_./\-][^\s]*))'

    file_contents = []
    expanded_files = []

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
        except UnicodeDecodeError:
            raise ValueError(f"File is not a text file or has encoding issues: {filename}")
        except PermissionError:
            raise ValueError(f"Permission denied reading file: {filename}")
        except Exception as e:
            raise ValueError(f"Failed to read file {filename}: {e}")

        # Track expanded files and their contents
        expanded_files.append(filename)
        file_contents.append(f"<File: {filename}>\n{content}\n</File: {filename}>")

        # Replace @filename with just the filename in the prompt
        return filename

    # Replace all @filename references and collect contents
    updated_prompt = re.sub(pattern, collect_file_ref, prompt)

    # If we found files, prepend their contents to the prompt
    if file_contents:
        files_section = "\n\n".join(file_contents)
        expanded_prompt = f"{files_section}\n\n{updated_prompt}"
    else:
        expanded_prompt = updated_prompt

    return expanded_prompt, expanded_files
