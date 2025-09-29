"""Common utilities for Tsugite."""

import yaml
from typing import Dict, Any, Tuple, Callable
from pathlib import Path


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
    """Create standardized error messages across tools.

    Args:
        operation: The operation being performed (e.g., "read", "write", "parse")
        target: The target of the operation (e.g., file path, tool name)
        error: The original exception

    Returns:
        Formatted error message
    """
    return f"Failed to {operation} {target}: {error}"


def tool_error(tool_name: str, operation: str, details: str) -> RuntimeError:
    """Create standardized tool execution errors.

    Args:
        tool_name: Name of the tool
        operation: What the tool was trying to do
        details: Error details

    Returns:
        RuntimeError with standardized message
    """
    return RuntimeError(f"Tool '{tool_name}' failed to {operation}: {details}")


def validation_error(item_type: str, item_name: str, issue: str) -> ValueError:
    """Create standardized validation errors.

    Args:
        item_type: Type of item being validated (e.g., "agent", "parameter")
        item_name: Name/identifier of the item
        issue: Description of the validation issue

    Returns:
        ValueError with standardized message
    """
    return ValueError(f"Invalid {item_type} '{item_name}': {issue}")
