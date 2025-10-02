"""Tool registry for Tsugite agents."""

import functools
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
        available = ", ".join(list(_tools.keys())) if _tools else "none"
        raise validation_error("tool", name, f"not found. Available: {available}")
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


def describe_tool(name: str) -> str:
    """Get description and parameters for a tool."""
    tool_info = get_tool(name)

    desc = f"{tool_info.name}: {tool_info.description}\n"
    desc += "Parameters:\n"

    for param_name, param_info in tool_info.parameters.items():
        required = " (required)" if param_info["required"] else f" (default: {param_info['default']})"
        desc += f"  - {param_name}: {param_info['type'].__name__}{required}\n"

    return desc


# Import all tool modules to register them
from . import agents, fs, http, shell, tasks
