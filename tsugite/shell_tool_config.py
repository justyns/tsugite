"""Configuration loader for custom shell tools."""

from pathlib import Path
from typing import Dict, List, Optional

import yaml

from .tools.shell_tools import ShellToolDefinition, ShellToolParameter
from .xdg import get_xdg_config_path


def get_custom_tools_config_path() -> Path:
    """Get the path to custom_tools.yaml config file."""
    return get_xdg_config_path("custom_tools.yaml")


def load_custom_tools_config(path: Optional[Path] = None) -> List[ShellToolDefinition]:
    """Load custom tool definitions from YAML config.

    Args:
        path: Path to custom_tools.yaml. If None, uses default XDG path.

    Returns:
        List of ShellToolDefinition objects

    Example YAML:
        tools:
          - name: file_search
            description: Search files with ripgrep
            command: "rg {pattern} {path}"
            timeout: 30
            parameters:
              pattern:
                type: str
                description: Search pattern
                required: true
              path:
                type: str
                description: Directory to search
                default: "."
    """
    if path is None:
        path = get_custom_tools_config_path()

    if not path.exists():
        return []

    try:
        with open(path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        if not config or "tools" not in config:
            return []

        definitions = []
        for tool_def in config["tools"]:
            # Parse parameters
            parameters = {}
            for param_name, param_config in tool_def.get("parameters", {}).items():
                # Handle multiple formats
                if isinstance(param_config, dict):
                    # Full dict format
                    param_type = param_config.get("type", "str")
                    description = param_config.get("description", "")
                    required = param_config.get("required", False)
                    default = param_config.get("default")
                    flag = param_config.get("flag")
                elif isinstance(param_config, str):
                    # Simple string format - assume it's the type
                    param_type = param_config if param_config else "str"
                    description = ""
                    required = False
                    default = None
                    flag = None
                elif param_config is None:
                    # Just parameter name, no config
                    param_type = "str"
                    description = ""
                    required = False
                    default = None
                    flag = None
                else:
                    # Value provided - infer type and use as default
                    if isinstance(param_config, bool):
                        param_type = "bool"
                        default = param_config
                    elif isinstance(param_config, int):
                        param_type = "int"
                        default = param_config
                    elif isinstance(param_config, float):
                        param_type = "float"
                        default = param_config
                    else:
                        param_type = "str"
                        default = str(param_config)
                    description = ""
                    required = False
                    flag = None

                parameters[param_name] = ShellToolParameter(
                    name=param_name,
                    type=param_type,
                    description=description,
                    required=required,
                    default=default,
                    flag=flag,
                )

            # Create definition
            definition = ShellToolDefinition(
                name=tool_def["name"],
                description=tool_def.get("description", ""),
                command=tool_def["command"],
                parameters=parameters,
                timeout=tool_def.get("timeout", 30),
                safe_mode=tool_def.get("safe_mode", True),
                shell=tool_def.get("shell", True),
            )

            definitions.append(definition)

        return definitions

    except yaml.YAMLError as e:
        raise ValueError(f"Failed to parse custom tools config: {e}") from e
    except Exception as e:
        raise RuntimeError(f"Failed to load custom tools config: {e}") from e


def save_custom_tools_config(definitions: List[ShellToolDefinition], path: Optional[Path] = None) -> None:
    """Save custom tool definitions to YAML config.

    Args:
        definitions: List of tool definitions to save
        path: Path to custom_tools.yaml. If None, uses default XDG path.
    """
    if path is None:
        path = get_custom_tools_config_path()

    # Ensure parent directory exists
    path.parent.mkdir(parents=True, exist_ok=True)

    # Convert definitions to YAML-friendly format
    tools_config = []
    for definition in definitions:
        tool_dict = {
            "name": definition.name,
            "description": definition.description,
            "command": definition.command,
            "timeout": definition.timeout,
            "safe_mode": definition.safe_mode,
            "shell": definition.shell,
            "parameters": {},
        }

        # Convert parameters
        for param_name, param_def in definition.parameters.items():
            tool_dict["parameters"][param_name] = {
                "type": param_def.type,
                "description": param_def.description,
                "required": param_def.required,
            }

            if param_def.default is not None:
                tool_dict["parameters"][param_name]["default"] = param_def.default

            if param_def.flag:
                tool_dict["parameters"][param_name]["flag"] = param_def.flag

        tools_config.append(tool_dict)

    config = {"tools": tools_config}

    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)


def parse_tool_definition_from_dict(tool_dict: Dict) -> ShellToolDefinition:
    """Parse a tool definition from a dictionary (e.g., from agent frontmatter).

    Args:
        tool_dict: Dictionary with tool configuration

    Returns:
        ShellToolDefinition object

    Example:
        {
            "name": "file_search",
            "command": "rg {pattern} {path}",
            "parameters": {
                "pattern": {"type": "str", "required": true},
                "path": ".",  # Infer type from default value
                "recursive": true  # Infer bool from value
            }
        }
    """
    parameters = {}
    for param_name, param_config in tool_dict.get("parameters", {}).items():
        if isinstance(param_config, dict):
            # Full dict format
            param_type = param_config.get("type", "str")
            description = param_config.get("description", "")
            required = param_config.get("required", False)
            default = param_config.get("default")
            flag = param_config.get("flag")
        elif isinstance(param_config, str):
            # String - assume it's either type name or default value
            # If it looks like a type name (str, int, bool, float), use it as type
            if param_config in ("str", "int", "bool", "float"):
                param_type = param_config
                description = ""
                required = False
                default = None
                flag = None
            else:
                # Otherwise it's a default value
                param_type = "str"
                description = ""
                required = False
                default = param_config
                flag = None
        elif param_config is None:
            # Just parameter name
            param_type = "str"
            description = ""
            required = False
            default = None
            flag = None
        else:
            # Value provided - infer type and use as default
            if isinstance(param_config, bool):
                param_type = "bool"
                default = param_config
            elif isinstance(param_config, int):
                param_type = "int"
                default = param_config
            elif isinstance(param_config, float):
                param_type = "float"
                default = param_config
            else:
                param_type = "str"
                default = str(param_config)
            description = ""
            required = False
            flag = None

        parameters[param_name] = ShellToolParameter(
            name=param_name,
            type=param_type,
            description=description,
            required=required,
            default=default,
            flag=flag,
        )

    return ShellToolDefinition(
        name=tool_dict["name"],
        description=tool_dict.get("description", ""),
        command=tool_dict["command"],
        parameters=parameters,
        timeout=tool_dict.get("timeout", 30),
        safe_mode=tool_dict.get("safe_mode", True),
        shell=tool_dict.get("shell", True),
    )
