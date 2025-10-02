"""Common utilities for Tsugite."""

from typing import Any, Callable, Dict, Tuple

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
