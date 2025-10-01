"""Bridge between Tsugite's tool registry and smolagents Tool class."""

import inspect
from functools import wraps
from typing import Iterable, List, Optional
from smolagents import tool as smol_tool
from tsugite.tools import get_tool, call_tool, list_tools


def create_smolagents_tool_from_tsugite(tool_name: str):
    """Create a smolagents tool from a Tsugite tool using the @tool decorator.

    This approach lets smolagents handle the tool interface properly.
    """
    tool_info = get_tool(tool_name)

    # Create a wrapper function with the correct signature
    sig = inspect.signature(tool_info.func)

    @wraps(tool_info.func)
    def wrapper_func(*args, **kwargs):
        try:
            bound_args = sig.bind_partial(*args, **kwargs)
            return call_tool(tool_name, **bound_args.arguments)
        except Exception as e:
            return f"Tool '{tool_name}' failed: {e}"

    setattr(wrapper_func, "__signature__", sig)

    # Apply smolagents tool decorator
    return smol_tool(wrapper_func)


def get_smolagents_tools(tool_names: Optional[Iterable[str]] = None) -> List:
    """Convert Tsugite tools to smolagents tools.

    Args:
        tool_names: List of tool names to convert. If None, converts all registered tools.

    Returns:
        List of smolagents Tool instances.
    """
    if tool_names is None:
        tool_names = list_tools()

    smol_tools = []
    for tool_name in tool_names:
        try:
            smol_tools.append(create_smolagents_tool_from_tsugite(tool_name))
        except Exception as e:
            print(f"Warning: Could not convert tool '{tool_name}': {e}")

    return smol_tools


def create_smolagents_tool_from_function(func):
    """Create a smolagents tool directly from a Tsugite-style function.

    This bypasses the Tsugite registry and creates a smolagents tool directly.
    Useful for dynamically creating tools or when you want smolagents native behavior.

    Args:
        func: Function with type hints and docstring following Tsugite conventions.

    Returns:
        smolagents Tool instance.
    """
    return smol_tool(func)
