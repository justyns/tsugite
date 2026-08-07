"""Tool system for agents.

Provides a simple Tool class and converters to wrap:
- Existing tsugite tools
- Custom functions
"""

import asyncio
import inspect
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

_JSON_TO_PYTHON_TYPE = {
    "string": "str",
    "integer": "int",
    "number": "float",
    "boolean": "bool",
    "array": "list",
    "object": "dict",
}

_PYTHON_TO_JSON_TYPE = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}

# Placeholder shown in a tool's usage example, by JSON schema type. Strings are
# absent because their example is derived from the parameter name.
_EXAMPLE_VALUES = {
    "integer": "42",
    "number": "3.14",
    "boolean": "True",
    "array": '["item1", "item2"]',
    "object": '{"key": "value"}',
}


@dataclass
class Tool:
    """A tool that agents can use.

    Built by `create_tool_from_function` or `create_tool_from_tsugite` rather
    than constructed directly.
    """

    name: str
    description: str
    parameters: Dict[str, Any]
    function: Callable
    # Dotted argument paths this tool declares as always sensitive; the executor
    # redacts them before recording or emitting the call.
    sensitive_paths: tuple[str, ...] = ()

    def to_code_prompt(self) -> str:
        """Format the tool as the keyword-only Python function the agent sees in the system prompt."""
        props = self.parameters.get("properties", {})
        required = self.parameters.get("required", [])

        params = []
        param_docs = []
        example_args = []
        for param_name, param_info in props.items():
            declared_type = param_info.get("type")
            params.append(f"{param_name}: {_JSON_TO_PYTHON_TYPE.get(declared_type, 'Any')}")

            required_marker = " (required)" if param_name in required else ""
            param_docs.append(f"        {param_name}: {param_info.get('description', '')}{required_marker}")

            if declared_type in (None, "string"):
                example_value = f'"{param_name}_value"'
            else:
                example_value = _EXAMPLE_VALUES.get(declared_type, "value")
            example_args.append(f"{param_name}={example_value}")

        param_str = f"*, {', '.join(params)}" if params else ""
        param_doc_str = "\n".join(param_docs) if param_docs else "        No parameters"
        usage_example = f"result = {self.name}({', '.join(example_args)})"

        return f'''def {self.name}({param_str}) -> Any:
    """{self.description}

    Args:
{param_doc_str}

    Usage:
        {usage_example}
    """
    pass
'''

    async def execute(self, **kwargs) -> Any:
        """Execute the tool, awaiting the wrapped function only when it is async."""
        if asyncio.iscoroutinefunction(self.function):
            return await self.function(**kwargs)
        return self.function(**kwargs)


def create_tool_from_function(func: Callable, name: Optional[str] = None, description: Optional[str] = None) -> Tool:
    """Create a Tool from a Python function.

    Extracts parameter info from function signature and docstring.

    Args:
        func: The function to wrap
        name: Tool name (defaults to function name)
        description: Tool description (defaults to docstring)

    Returns:
        Tool: Wrapped function

    Example:
        def multiply(a: int, b: int) -> int:
            '''Multiply two numbers'''
            return a * b

        tool = create_tool_from_function(multiply)
    """
    tool_name = name or func.__name__
    tool_description = description or (func.__doc__ or "").strip().split("\n")[0]

    parameters = {"type": "object", "properties": {}, "required": []}

    for param_name, param in inspect.signature(func).parameters.items():
        if param_name in ("self", "cls"):
            continue

        # An unannotated param, or one with no JSON Schema equivalent, gets no type
        # constraint: still valid JSON Schema, and it accepts any value.
        json_type = _PYTHON_TO_JSON_TYPE.get(param.annotation)
        parameters["properties"][param_name] = {"type": json_type} if json_type else {}

        if param.default == inspect.Parameter.empty:
            parameters["required"].append(param_name)

    return Tool(
        name=tool_name,
        description=tool_description,
        parameters=parameters,
        function=func,
        # Declared via @tool(sensitive_args=...), which stamps the function.
        sensitive_paths=tuple(getattr(func, "_sensitive_args", ()) or ()),
    )


def create_tool_from_tsugite(tool_name: str) -> Tool:
    """Convert existing tsugite tool to Tool object.

    Tsugite has its own tool registry. This function wraps those
    tools in our Tool interface.

    Args:
        tool_name: Name of tool in tsugite registry

    Returns:
        Tool: Wrapped tsugite tool

    Example:
        tool = create_tool_from_tsugite("read_file")
        result = await tool.execute(file_path="/path/to/file")
    """
    from tsugite.tools import call_tool, get_tool

    tool_info = get_tool(tool_name)

    async def tool_wrapper(**kwargs):
        result = call_tool(tool_name, **kwargs)
        if inspect.iscoroutine(result):
            return await result
        return result

    # Carry the registry's signature and sensitive-arg declaration onto the wrapper,
    # so it looks like any other @tool-decorated function downstream.
    tool_wrapper.__signature__ = inspect.signature(tool_info.func)
    tool_wrapper._sensitive_args = tool_info.sensitive_args

    return create_tool_from_function(
        tool_wrapper,
        name=tool_name,
        description=tool_info.description,
    )
