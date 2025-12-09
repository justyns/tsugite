"""MCP server exposing Tsugite tools to external clients."""

import asyncio
import json
from typing import Any, Dict, List, Optional

from mcp.server import Server
from mcp.types import TextContent, Tool

# Tools to expose via MCP
EXPOSED_TOOLS = [
    "memory_store",
    "memory_search",
    "memory_list",
    "memory_get",
    "memory_update",
    "memory_delete",
    "memory_count",
    "spawn_agent",
    "list_agents",
]

server = Server("tsugite")


def _python_type_to_json_schema(python_type: Any) -> Dict[str, Any]:
    """Convert Python type annotation to JSON schema type."""
    if python_type is None or python_type is type(None):
        return {"type": "null"}

    type_name = getattr(python_type, "__name__", str(python_type))

    type_map = {
        "str": {"type": "string"},
        "int": {"type": "integer"},
        "float": {"type": "number"},
        "bool": {"type": "boolean"},
        "list": {"type": "array"},
        "dict": {"type": "object"},
        "List": {"type": "array"},
        "Dict": {"type": "object"},
        "Any": {},
    }

    # Handle Optional types
    origin = getattr(python_type, "__origin__", None)
    if origin is not None:
        origin_name = getattr(origin, "__name__", str(origin))
        if origin_name == "Optional" or str(origin) == "typing.Union":
            args = getattr(python_type, "__args__", ())
            if len(args) == 2 and type(None) in args:
                non_none = [a for a in args if a is not type(None)][0]
                return _python_type_to_json_schema(non_none)
        if origin_name in ("List", "list"):
            return {"type": "array"}
        if origin_name in ("Dict", "dict"):
            return {"type": "object"}

    return type_map.get(type_name, {"type": "string"})


def _tool_info_to_json_schema(tool_info) -> Dict[str, Any]:
    """Convert ToolInfo parameters to JSON schema."""
    properties = {}
    required = []

    for param_name, param_info in tool_info.parameters.items():
        param_type = param_info.get("type", str)
        schema = _python_type_to_json_schema(param_type)

        if param_info.get("default") is not None:
            schema["default"] = param_info["default"]

        properties[param_name] = schema

        if param_info.get("required", False):
            required.append(param_name)

    return {
        "type": "object",
        "properties": properties,
        "required": required,
    }


def get_mcp_tools() -> List[Tool]:
    """Convert tsugite tools to MCP Tool format."""
    from tsugite.tools import get_tool

    tools = []
    for name in EXPOSED_TOOLS:
        try:
            tool_info = get_tool(name)
            input_schema = _tool_info_to_json_schema(tool_info)

            tools.append(
                Tool(
                    name=tool_info.name,
                    description=tool_info.description,
                    inputSchema=input_schema,
                )
            )
        except ValueError:
            pass  # Tool not found, skip

    return tools


async def execute_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
    """Execute a tsugite tool and return MCP-formatted result."""
    from tsugite.tools import get_tool

    if name not in EXPOSED_TOOLS:
        return [TextContent(type="text", text=json.dumps({"error": f"Tool '{name}' not exposed via MCP"}))]

    try:
        tool_info = get_tool(name)
        result = tool_info.func(**arguments)

        if asyncio.iscoroutine(result):
            result = await result

        return [TextContent(type="text", text=json.dumps(result, default=str))]
    except Exception as e:
        return [TextContent(type="text", text=json.dumps({"error": str(e)}))]


@server.list_tools()
async def handle_list_tools() -> List[Tool]:
    """MCP handler: list available tools."""
    return get_mcp_tools()


@server.call_tool()
async def handle_call_tool(name: str, arguments: Optional[Dict[str, Any]] = None) -> List[TextContent]:
    """MCP handler: execute a tool."""
    return await execute_tool(name, arguments or {})


async def _run_stdio_async():
    """Async implementation of stdio server."""
    from mcp.server.stdio import stdio_server

    async with stdio_server() as (read_stream, write_stream):
        init_options = server.create_initialization_options()
        await server.run(read_stream, write_stream, init_options)


def run_stdio():
    """Run MCP server over stdio transport."""
    asyncio.run(_run_stdio_async())
