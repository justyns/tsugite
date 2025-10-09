"""Tool system for agents.

Provides a simple Tool class and converters to wrap:
- Existing tsugite tools
- MCP tools
- Custom functions
"""

import asyncio
import inspect
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional


@dataclass
class Tool:
    """A tool that agents can use.

    Tools are Python functions with:
    - name: Identifier
    - description: What it does
    - parameters: JSON schema of arguments
    - function: The actual callable

    Example:
        def add(a: int, b: int) -> int:
            '''Add two numbers'''
            return a + b

        tool = Tool(
            name="add",
            description="Add two numbers",
            parameters={
                "type": "object",
                "properties": {
                    "a": {"type": "integer"},
                    "b": {"type": "integer"}
                },
                "required": ["a", "b"]
            },
            function=add
        )
    """

    name: str
    description: str
    parameters: Dict[str, Any]
    function: Callable

    def to_code_prompt(self) -> str:
        """Format tool as Python function for system prompt.

        The agent sees tools as Python functions it can call.

        Returns:
            str: Python function signature with docstring

        Example output:
            def add(a: int, b: int) -> Any:
                '''Add two numbers

                Args:
                    a: First number
                    b: Second number
                '''
                pass
        """
        # Extract parameter info from JSON schema
        props = self.parameters.get("properties", {})
        required = self.parameters.get("required", [])

        # Build function signature
        params = []
        for param_name, param_info in props.items():
            # Get type (default to Any)
            param_type = param_info.get("type", "Any")

            # Convert JSON schema types to Python types
            type_map = {
                "string": "str",
                "integer": "int",
                "number": "float",
                "boolean": "bool",
                "array": "list",
                "object": "dict",
            }
            python_type = type_map.get(param_type, "Any")

            # Add to params list
            params.append(f"{param_name}: {python_type}")

        param_str = ", ".join(params)

        # Build docstring with parameter descriptions
        param_docs = []
        for param_name, param_info in props.items():
            desc = param_info.get("description", "")
            required_marker = " (required)" if param_name in required else ""
            param_docs.append(f"        {param_name}: {desc}{required_marker}")

        param_doc_str = "\n".join(param_docs) if param_docs else "        No parameters"

        # Build full function definition
        return f'''def {self.name}({param_str}) -> Any:
    """{self.description}

    Args:
{param_doc_str}
    """
    pass
'''

    async def execute(self, **kwargs) -> Any:
        """Execute the tool with given arguments.

        Handles both sync and async functions.

        Args:
            **kwargs: Arguments to pass to the tool

        Returns:
            Tool execution result
        """
        # Check if function is async
        if asyncio.iscoroutinefunction(self.function):
            return await self.function(**kwargs)
        else:
            # Run sync function
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
    # Get name and description
    tool_name = name or func.__name__
    tool_description = description or (func.__doc__ or "").strip().split("\n")[0]

    # Extract parameters from signature
    sig = inspect.signature(func)
    parameters = {"type": "object", "properties": {}, "required": []}

    for param_name, param in sig.parameters.items():
        # Skip self/cls
        if param_name in ("self", "cls"):
            continue

        # Get type annotation and convert to JSON schema type
        if param.annotation != inspect.Parameter.empty:
            # Convert Python type to JSON schema type
            type_map = {
                str: "string",
                int: "integer",
                float: "number",
                bool: "boolean",
                list: "array",
                dict: "object",
            }
            param_type = type_map.get(param.annotation)

            if param_type:
                # Known type - add type constraint
                parameters["properties"][param_name] = {"type": param_type}
            else:
                # Unknown type - omit type constraint (accepts any value, but valid JSON Schema)
                parameters["properties"][param_name] = {}
        else:
            # No annotation - omit type constraint
            parameters["properties"][param_name] = {}

        # Check if required (no default value)
        if param.default == inspect.Parameter.empty:
            parameters["required"].append(param_name)

    return Tool(
        name=tool_name,
        description=tool_description,
        parameters=parameters,
        function=func,
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

    # Get tool info from registry
    tool_info = get_tool(tool_name)

    # Create async wrapper function that preserves signature
    sig = inspect.signature(tool_info.func)

    async def tool_wrapper(**kwargs):
        """Wrapper that calls tsugite tool."""
        # call_tool might be sync or async, handle both
        result = call_tool(tool_name, **kwargs)
        # If result is a coroutine, await it
        if inspect.iscoroutine(result):
            return await result
        return result

    # Set the signature on the wrapper
    tool_wrapper.__signature__ = sig

    # Create Tool using function converter
    return create_tool_from_function(
        tool_wrapper,
        name=tool_name,
        description=tool_info.description,
    )
