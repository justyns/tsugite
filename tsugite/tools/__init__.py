"""Tool registry for Tsugite agents."""

import inspect
from dataclasses import dataclass
from typing import Any, Callable, Dict, List

from ..utils import tool_error, validation_error


@dataclass
class ToolInfo:
    """Information about a registered tool."""

    name: str
    func: Callable
    description: str
    parameters: Dict[str, Any]


# Global tool registry
_tools: Dict[str, ToolInfo] = {}


def tool(func: Callable) -> Callable:
    """Register a function as a tool."""
    # Extract function signature and docstring
    sig = inspect.signature(func)
    doc = func.__doc__ or "No description available"

    # Extract parameter info
    parameters = {}
    for param_name, param in sig.parameters.items():
        parameters[param_name] = {
            "type": (param.annotation if param.annotation != inspect.Parameter.empty else str),
            "default": (param.default if param.default != inspect.Parameter.empty else None),
            "required": param.default == inspect.Parameter.empty,
        }

    tool_info = ToolInfo(
        name=func.__name__,
        func=func,
        description=doc.split("\n")[0].strip(),  # First line of docstring
        parameters=parameters,
    )

    _tools[func.__name__] = tool_info
    return func


def get_tool(name: str) -> ToolInfo:
    """Get a registered tool by name."""
    if name not in _tools:
        # Provide helpful error message
        from ..shell_tool_config import get_custom_tools_config_path

        error_parts = ["not found"]

        # Check if it might be a custom tool
        config_path = get_custom_tools_config_path()
        if config_path.exists():
            error_parts.append(f"Check if '{name}' is defined in {config_path}")
        else:
            error_parts.append(f"Custom tools config not found at {config_path}")

        # Suggest similar tool names
        all_tools = list(_tools.keys())
        similar = [t for t in all_tools if name.lower() in t.lower() or t.lower() in name.lower()]
        if similar:
            error_parts.append(f"Did you mean: {', '.join(similar[:3])}?")

        error_parts.append("Run 'tsugite tools list' to see all available tools")

        raise validation_error("tool", name, ". ".join(error_parts))
    return _tools[name]


def call_tool(name: str, **kwargs) -> Any:
    """Call a tool with the given arguments."""
    tool_info = get_tool(name)

    # Validate required parameters
    for param_name, param_info in tool_info.parameters.items():
        if param_info["required"] and param_name not in kwargs:
            raise validation_error("parameter", param_name, f"missing for tool '{name}'")

    try:
        return tool_info.func(**kwargs)
    except Exception as e:
        raise tool_error(name, "execute", str(e))


def list_tools() -> List[str]:
    """List all registered tool names."""
    return list(_tools.keys())


def get_tools_by_category(category: str) -> List[str]:
    """Get all tool names in a specific category.

    Args:
        category: Category name (e.g., 'fs', 'http', 'shell')

    Returns:
        List of tool names in the category
    """
    category_tools = []
    for tool_name, tool_info in _tools.items():
        module = tool_info.func.__module__.split(".")[-1]
        if module == category:
            category_tools.append(tool_name)

    return sorted(category_tools)


def _expand_single_spec(spec: str, strict: bool = True) -> List[str]:
    """Expand a single tool specification to tool names.

    Args:
        spec: Tool specification (name, @category, or glob pattern)
        strict: If True, raise error when spec matches nothing. If False, return empty list.

    Returns:
        List of matching tool names
    """
    import fnmatch

    if spec.startswith("@"):
        # Category reference: @fs, @http, etc.
        category = spec[1:]
        category_tools = get_tools_by_category(category)
        if not category_tools and strict:
            raise validation_error("tool category", category, "not found or empty")
        return category_tools
    elif "*" in spec or "?" in spec or "[" in spec:
        # Glob pattern
        all_tool_names = list_tools()
        matches = fnmatch.filter(all_tool_names, spec)
        if not matches and strict:
            raise validation_error("tool pattern", spec, "matched no tools")
        return matches
    else:
        # Regular tool name
        if spec not in _tools:
            if strict:
                available = ", ".join(list(_tools.keys())) if _tools else "none"
                raise validation_error("tool", spec, f"not found. Available: {available}")
            return []
        return [spec]


def expand_tool_specs(tool_specs: List[str]) -> List[str]:
    """Expand tool specifications to actual tool names.

    Supports:
    - Regular tool names: 'read_file' -> ['read_file']
    - Category references: '@fs' -> all tools in fs category
    - Glob patterns: '*_file' -> all tools matching pattern
    - Exclusions: '-tool_name', '-@category', '-pattern*' -> remove matching tools

    Args:
        tool_specs: List of tool specifications (names, @category, globs, or exclusions with -)

    Returns:
        Expanded list of unique tool names

    Examples:
        >>> expand_tool_specs(['read_file', 'write_file'])
        ['read_file', 'write_file']
        >>> expand_tool_specs(['@fs'])
        ['create_directory', 'file_exists', 'list_files', 'read_file', 'write_file']
        >>> expand_tool_specs(['*_file'])
        ['read_file', 'write_file']
        >>> expand_tool_specs(['@fs', '-*_directory'])
        ['file_exists', 'list_files', 'read_file', 'write_file']
        >>> expand_tool_specs(['@http', '-web_search'])
        ['check_url', 'download_file', 'fetch_json', 'fetch_text', 'post_json']
    """
    # Separate inclusions and exclusions
    inclusions = []
    exclusions = []

    for spec in tool_specs:
        if spec.startswith("-"):
            exclusions.append(spec[1:])  # Strip the - prefix
        else:
            inclusions.append(spec)

    # Expand inclusions (strict - must match something)
    expanded = []
    for spec in inclusions:
        expanded.extend(_expand_single_spec(spec, strict=True))

    # Expand exclusions (non-strict - silently ignore if nothing matches)
    excluded_tools = set()
    for spec in exclusions:
        excluded_tools.update(_expand_single_spec(spec, strict=False))

    # Apply exclusions
    result_with_exclusions = [tool for tool in expanded if tool not in excluded_tools]

    # Return unique tools while preserving order
    seen = set()
    result = []
    for tool in result_with_exclusions:
        if tool not in seen:
            seen.add(tool)
            result.append(tool)

    return result


def load_custom_shell_tools() -> None:
    """Load custom shell tools from config file.

    This is called automatically at module import time to register
    user-defined shell tools from custom_tools.yaml.
    """
    import os
    import sys

    try:
        from ..shell_tool_config import get_custom_tools_config_path, load_custom_tools_config
        from .shell_tools import register_shell_tools

        config_path = get_custom_tools_config_path()

        # Only try to load if config file exists
        if not config_path.exists():
            # Silently skip if no custom tools configured
            return

        definitions = load_custom_tools_config()
        if definitions:
            register_shell_tools(definitions)

            # Show helpful message if verbose mode enabled
            if os.environ.get("TSUGITE_VERBOSE") or os.environ.get("TSUGITE_DEBUG"):
                tool_names = [d.name for d in definitions]
                print(
                    f"✓ Loaded {len(definitions)} custom tool(s): {', '.join(tool_names)}",
                    file=sys.stderr,
                )
        else:
            # Config exists but no tools defined
            if os.environ.get("TSUGITE_VERBOSE") or os.environ.get("TSUGITE_DEBUG"):
                print(f"⚠ Custom tools config exists but no tools defined: {config_path}", file=sys.stderr)

    except Exception as e:
        # Don't fail startup if custom tools can't be loaded, but show clear error
        print(f"⚠ Failed to load custom tools: {e}", file=sys.stderr)
        print(f"  Config file: {get_custom_tools_config_path()}", file=sys.stderr)
        print("  Use 'tsugite tools validate' to check your config", file=sys.stderr)


# Import tool modules at the end to avoid circular imports
# (they need to import 'tool' decorator from this module)
from . import agents as agents  # noqa: E402
from . import fs as fs  # noqa: E402
from . import http as http  # noqa: E402
from . import interactive as interactive  # noqa: E402
from . import shell as shell  # noqa: E402
from . import tasks as tasks  # noqa: E402

# Load custom shell tools after built-in tools
load_custom_shell_tools()
