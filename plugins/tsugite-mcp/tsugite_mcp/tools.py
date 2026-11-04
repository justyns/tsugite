"""Agent-facing tools for connected MCP servers (the `tsugite.plugins` entry point).

Each tool of a connected server is registered as `mcp_<server>_<tool>` in the category
`mcp_<server>`, so an agent enables one server with `@mcp_<server>` or all with `mcp_*`.
"""

import inspect
import keyword
import logging
import re
from pathlib import Path

from tsugite.exceptions import ToolUnavailableError
from tsugite.tools import _tools, call_on_loop, tool
from tsugite.ui_surfaces import register_ui_surface

CALL_TIMEOUT = 120
JSON_TYPES = {"string": str, "integer": int, "number": float, "boolean": bool, "array": list, "object": dict}

logger = logging.getLogger(__name__)

register_ui_surface(
    kind="connectors",
    label="Connectors",
    icon="plug",
    entry="ui/connectors.html",
    assets=Path(__file__).parent / "ui",
    nav=True,
)

_adapter = None
_registered: dict[str, list[str]] = {}


def set_adapter(adapter) -> None:
    global _adapter
    _adapter = adapter


def clear_tools(server: str) -> None:
    for name in _registered.pop(server, []):
        _tools.pop(name, None)


def _caller(server: str, name: str):
    def call(**kwargs):
        if _adapter is None:
            raise ToolUnavailableError("MCP tools need the tsugite daemon with the mcp plugin")
        arguments = {key: value for key, value in kwargs.items() if value is not None}
        return call_on_loop(_adapter.loop, _adapter.call_tool, server, name, arguments, timeout=CALL_TIMEOUT)

    return call


def set_tools(server: str, listed) -> list[dict]:
    """Replace `server`'s registered tools with the MCP tools in `listed`. Returns what registered."""
    clear_tools(server)
    registered = []
    for spec in listed:
        schema = spec.inputSchema or {}
        properties = schema.get("properties", {})
        bad = [
            key
            for key, prop in properties.items()
            if not (isinstance(prop, dict) and key.isidentifier() and not keyword.iskeyword(key))
        ]
        if bad:
            logger.warning("mcp %s: skipping tool %s, argument %r is not usable", server, spec.name, bad[0])
            continue
        safe = re.sub(r"\W", "_", spec.name)
        name = f"mcp_{server}_{safe}"
        if name in _registered.get(server, []):
            logger.warning("mcp %s: skipping tool %s, its name collides with another tool", server, spec.name)
            continue
        required = set(schema.get("required", []))
        parameters = [
            inspect.Parameter(
                key,
                inspect.Parameter.KEYWORD_ONLY,
                default=inspect.Parameter.empty if key in required else None,
                annotation=JSON_TYPES.get(str(prop.get("type")), inspect.Parameter.empty),
            )
            for key, prop in sorted(properties.items(), key=lambda item: item[0] not in required)
        ]
        call = _caller(server, spec.name)
        call.__name__ = name
        call.__doc__ = spec.description or f"{spec.name} on MCP server {server}"
        call.__signature__ = inspect.Signature(parameters)
        tool(category=f"mcp_{server}")(call)
        _registered.setdefault(server, []).append(call.__name__)
        registered.append({"name": call.__name__, "description": call.__doc__.split("\n")[0].strip()})
    return registered
