"""MCP server configuration loading and management."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from .xdg import get_xdg_config_path, get_xdg_write_path


@dataclass
class MCPServerConfig:
    """Configuration for a single MCP server."""

    name: str
    command: Optional[str] = None
    args: Optional[List[str]] = None
    env: Optional[Dict[str, str]] = None
    url: Optional[str] = None
    type: Optional[str] = None

    def __post_init__(self):
        if self.type is None:
            if self.command:
                self.type = "stdio"
            elif self.url:
                self.type = "http"
            else:
                raise ValueError(f"Server '{self.name}' must have either 'command' or 'url' specified")

        if self.type == "stdio" and not self.command:
            raise ValueError(f"Stdio server '{self.name}' must have 'command' specified")

        if self.type == "http" and not self.url:
            raise ValueError(f"HTTP server '{self.name}' must have 'url' specified")

    def is_stdio(self) -> bool:
        return self.type == "stdio"

    def is_http(self) -> bool:
        return self.type == "http"


def get_default_config_path() -> Path:
    return get_xdg_config_path("mcp.json")


def get_config_path_for_write() -> Path:
    return get_xdg_write_path("mcp.json")


def load_mcp_config(path: Optional[Path] = None) -> Dict[str, MCPServerConfig]:
    """Load MCP server configurations from JSON file.

    Args:
        path: Path to mcp.json file. If None, uses default ~/.tsugite/mcp.json

    Returns:
        Dictionary mapping server names to MCPServerConfig objects.
        Returns empty dict if file doesn't exist or is invalid.
    """
    if path is None:
        path = get_default_config_path()

    if not path.exists():
        return {}

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if "mcpServers" not in data:
            print(f"Warning: No 'mcpServers' key found in {path}")
            return {}

        servers = {}
        for name, config in data["mcpServers"].items():
            try:
                servers[name] = MCPServerConfig(
                    name=name,
                    command=config.get("command"),
                    args=config.get("args"),
                    env=config.get("env"),
                    url=config.get("url"),
                    type=config.get("type"),
                )
            except ValueError as e:
                print(f"Warning: Invalid config for server '{name}': {e}")
                continue

        return servers

    except json.JSONDecodeError as e:
        print(f"Warning: Failed to parse MCP config at {path}: {e}")
        return {}
    except Exception as e:
        print(f"Warning: Failed to load MCP config from {path}: {e}")
        return {}


def save_mcp_config(servers: Dict[str, MCPServerConfig], path: Optional[Path] = None) -> None:
    """Save MCP server configurations to JSON file.

    Args:
        servers: Dictionary mapping server names to MCPServerConfig objects
        path: Path to mcp.json file. If None, uses appropriate config location

    Raises:
        IOError: If file cannot be written
    """
    if path is None:
        path = get_config_path_for_write()

    # Ensure directory exists
    path.parent.mkdir(parents=True, exist_ok=True)

    # Convert MCPServerConfig objects to JSON-serializable dicts
    config_data = {"mcpServers": {}}

    for name, server in servers.items():
        server_dict = {}

        if server.is_stdio():
            server_dict["command"] = server.command
            if server.args:
                server_dict["args"] = server.args
            if server.env:
                server_dict["env"] = server.env
        else:  # HTTP
            server_dict["type"] = "http"
            server_dict["url"] = server.url

        config_data["mcpServers"][name] = server_dict

    # Write with pretty formatting
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=2)


def add_server_to_config(server: MCPServerConfig, path: Optional[Path] = None, overwrite: bool = False) -> bool:
    """Add or update an MCP server in the configuration file.

    Args:
        server: MCPServerConfig object to add
        path: Path to mcp.json file. If None, uses appropriate config location
        overwrite: If True, overwrite existing server. If False, raise error if exists.

    Returns:
        True if server was added/updated successfully

    Raises:
        ValueError: If server already exists and overwrite is False
    """
    if path is None:
        path = get_config_path_for_write()

    # Load existing config
    servers = load_mcp_config(path)

    # Check if server already exists
    if server.name in servers and not overwrite:
        raise ValueError(f"Server '{server.name}' already exists. Use --force to overwrite.")

    # Add/update server
    servers[server.name] = server

    # Save config
    save_mcp_config(servers, path)

    return True
