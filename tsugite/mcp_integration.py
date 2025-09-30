"""MCP server integration for loading tools from MCP servers."""

import os
from typing import List, Optional
from smolagents import ToolCollection
from mcp import StdioServerParameters
from .mcp_config import MCPServerConfig


def convert_to_server_params(server_config: MCPServerConfig):
    """Convert MCPServerConfig to smolagents-compatible server parameters.

    Args:
        server_config: MCP server configuration

    Returns:
        StdioServerParameters for stdio servers, or dict for HTTP servers
    """
    if server_config.is_stdio():
        env = server_config.env or {}
        full_env = {**os.environ, **env}

        return StdioServerParameters(
            command=server_config.command,
            args=server_config.args or [],
            env=full_env,
        )
    elif server_config.is_http():
        return {"url": server_config.url, "transport": "streamable-http"}
    else:
        raise ValueError(f"Unknown server type: {server_config.type}")


def load_mcp_tools(
    server_name: str,
    server_config: MCPServerConfig,
    allowed_tools: Optional[List[str]] = None,
    trust_remote_code: bool = False,
) -> List:
    """Load tools from an MCP server with optional filtering.

    Args:
        server_name: Name of the MCP server (for logging)
        server_config: MCP server configuration
        allowed_tools: If provided, only load these specific tools. If None, load all.
        trust_remote_code: Whether to trust remote code execution

    Returns:
        List of smolagents Tool objects

    Raises:
        RuntimeError: If connection fails or tools cannot be loaded
    """
    try:
        server_params = convert_to_server_params(server_config)

        with ToolCollection.from_mcp(server_params, trust_remote_code=trust_remote_code) as tool_collection:
            all_tools = list(tool_collection.tools)

            if allowed_tools is None:
                return all_tools

            tool_dict = {tool.name: tool for tool in all_tools}
            filtered_tools = []

            for tool_name in allowed_tools:
                if tool_name in tool_dict:
                    filtered_tools.append(tool_dict[tool_name])
                else:
                    available_tools = ", ".join(tool_dict.keys())
                    print(
                        f"Warning: Tool '{tool_name}' not found in MCP server '{server_name}'. "
                        f"Available tools: {available_tools}"
                    )

            return filtered_tools

    except Exception as e:
        raise RuntimeError(f"Failed to load tools from MCP server '{server_name}': {e}")


def load_all_mcp_tools(
    mcp_servers_config: dict,
    global_mcp_config: dict,
    trust_remote_code: bool = False,
) -> List:
    """Load tools from multiple MCP servers.

    Args:
        mcp_servers_config: Dict mapping server names to optional tool lists (from agent config)
        global_mcp_config: Dict mapping server names to MCPServerConfig objects
        trust_remote_code: Whether to trust remote code execution

    Returns:
        List of all loaded tools from all servers
    """
    all_tools = []

    for server_name, allowed_tools in mcp_servers_config.items():
        if server_name not in global_mcp_config:
            print(f"Warning: MCP server '{server_name}' not found in config. Skipping.")
            continue

        server_config = global_mcp_config[server_name]

        try:
            tools = load_mcp_tools(server_name, server_config, allowed_tools, trust_remote_code)
            all_tools.extend(tools)
            tool_names = [t.name for t in tools]
            print(f"Loaded {len(tools)} tools from MCP server '{server_name}': {', '.join(tool_names)}")
        except RuntimeError as e:
            print(f"Warning: {e}. Continuing without tools from this server.")
            continue

    return all_tools
