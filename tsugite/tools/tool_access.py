import inspect
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, List, Optional

from tsugite.approval import request_approval
from tsugite.events.helpers import current_session_id
from tsugite.permissions import get_permissions

from . import get_tool, iter_tool_infos, tool

TOOL_ACCESS_TOOLS = ["search_tools", "load_tool", "request_tool_access"]
MAX_RESULTS = 20
MAX_REMEMBERED = 256


@dataclass
class ToolAccess:
    agent_name: str
    allowed: set[str]
    # Shared with every turn of this agent in this session, through _remembered.
    loaded: list[str]
    denied: set[str]
    # The running TsugiteAgent, set by the runner so a loaded tool joins the live session.
    agent: Any = None


current_access: ContextVar[Optional[ToolAccess]] = ContextVar("tool_access", default=None)
# (session_id, agent_name) -> (loaded, denied), least recently used first.
_remembered: dict[tuple, tuple[list[str], set[str]]] = {}


def start_turn(agent_name: str, described: List[str], deferred: List[str]) -> ToolAccess:
    key = (current_session_id(), agent_name)
    loaded, denied = _remembered[key] = _remembered.pop(key, None) or ([], set())
    while len(_remembered) > MAX_REMEMBERED:
        del _remembered[next(iter(_remembered))]
    access = ToolAccess(agent_name, set(described) | set(deferred) | set(loaded), loaded, denied)
    current_access.set(access)
    return access


def get_access() -> ToolAccess:
    access = current_access.get()
    if access is None:
        raise RuntimeError("Tool access is only available to agents with deferred_tools set")
    return access


def _load(access: ToolAccess, names: List[str]) -> dict:
    from tsugite.core.tools import create_tool_from_tsugite

    created = {n: create_tool_from_tsugite(n) for n in names}
    new = [name for name in names if name not in access.loaded]
    access.loaded.extend(new)
    if access.agent is not None:
        have = {t.name for t in access.agent.tools}
        access.agent.tools.extend(created[n] for n in new if n not in have)
        access.agent._inject_tools_into_executor()
    return created


@tool(parent_only=True)
def search_tools(query: str) -> List[dict]:
    """Search every installed tool by name or description, including ones not in your tool list.

    Each match has a status: "available" tools can be loaded with load_tool, "not allowed"
    tools need request_tool_access first.

    Args:
        query: Words to look for in tool names and descriptions

    Example:
        search_tools("calendar")
    """
    access = get_access()
    words = query.lower().split()
    matches = []
    for info in sorted(iter_tool_infos(), key=lambda i: i.name):
        text = f"{info.name} {info.description} {info.category or ''}".lower()
        if all(word in text for word in words):
            status = "available" if info.name in access.allowed else "not allowed"
            matches.append({"name": info.name, "summary": info.description, "status": status})
    return matches[:MAX_RESULTS]


@tool(parent_only=True)
def load_tool(name: str) -> str:
    """Load an available tool, or one you already have, and return its signature and full documentation.

    Args:
        name: Tool name from search_tools with status "available"

    Example:
        load_tool("schedule_list")
    """
    access = get_access()
    if name not in access.allowed:
        raise PermissionError(f"Tool '{name}' is not allowed for this agent. Call request_tool_access first.")
    signature = _load(access, [name])[name].to_code_prompt()
    return f"{signature}\nFull documentation:\n{inspect.getdoc(get_tool(name).func) or ''}"


@tool(parent_only=True)
def request_tool_access(tools: List[str], reason: str) -> str:
    """Ask the user to let you use tools that are not allowed for this agent.

    Approved tools load into this session immediately. A denied tool is not asked for again.

    Args:
        tools: Tool names from search_tools
        reason: Why you need them, shown to the user

    Example:
        request_tool_access(["mcp_calendar_create_event"], "Add the meeting you asked for")
    """
    access = get_access()
    infos = {i.name: i for i in iter_tool_infos()}
    unknown = [t for t in tools if t not in infos]
    if unknown:
        raise ValueError(f"Invalid tools {unknown}: not installed. Use search_tools to list tool names")
    ask = [t for t in tools if t not in access.allowed and t not in access.denied]
    if ask:
        lines = []
        for name in ask:
            category = infos[name].category or ""
            server = f" (MCP server {category.removeprefix('mcp_')})" if category.startswith("mcp_") else ""
            lines.append(f"- {name}{server}: {infos[name].description}")
        perms = get_permissions()
        decision = request_approval(
            f"Allow agent '{access.agent_name}' to use these tools?",
            allow_always=perms is not None,
            detail=f"Reason: {reason}\n\n" + "\n".join(lines) + "\n\nApprove allows them for this session only.",
        )
        if decision == "always":
            for name in ask:
                perms.add(f"agents.{access.agent_name}.tools", name)
        if decision in ("approve", "always"):
            access.allowed.update(ask)
        else:
            access.denied.update(ask)
    granted = [t for t in tools if t in access.allowed]
    _load(access, granted)
    return f"Loaded: {granted or 'none'}. Denied: {[t for t in tools if t not in granted] or 'none'}."
