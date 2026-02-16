"""Configuration loader for custom shell tools."""

from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from .config import get_xdg_config_path
from .tools.shell_tools import ShellToolDefinition, ShellToolParameter


def get_custom_tools_config_path() -> Path:
    """Get the path to custom_tools.yaml config file."""
    return get_xdg_config_path("custom_tools.yaml")


def _parse_parameter(param_name: str, param_config: Any) -> ShellToolParameter:
    """Parse a single parameter config into a ShellToolParameter.

    Handles multiple input formats:
    - dict: Full config with type, description, required, default, flag
    - str: Type name (str/int/bool/float) or default string value
    - None: String parameter with no default
    - bool/int/float: Infer type from value, use as default
    """
    if isinstance(param_config, dict):
        param_type = param_config.get("type", "str")
        description = param_config.get("description", "")
        required = param_config.get("required", False)
        default = param_config.get("default")
        flag = param_config.get("flag")
    elif isinstance(param_config, str):
        # If it looks like a type name, use it as type; otherwise it's a default value
        if param_config in ("str", "int", "bool", "float"):
            param_type = param_config
            default = None
        else:
            param_type = "str"
            default = param_config
        description = ""
        required = False
        flag = None
    elif param_config is None:
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

    return ShellToolParameter(
        name=param_name,
        type=param_type,
        description=description,
        required=required,
        default=default,
        flag=flag,
    )


def _parse_parameters(raw_params: Dict) -> Dict[str, ShellToolParameter]:
    """Parse a parameters dict into ShellToolParameter objects."""
    return {name: _parse_parameter(name, config) for name, config in raw_params.items()}


def load_custom_tools_config(path: Optional[Path] = None) -> List[ShellToolDefinition]:
    """Load custom tool definitions from YAML config.

    Args:
        path: Path to custom_tools.yaml. If None, uses default XDG path.

    Returns:
        List of ShellToolDefinition objects
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
            definition = ShellToolDefinition(
                name=tool_def["name"],
                description=tool_def.get("description", ""),
                command=tool_def["command"],
                parameters=_parse_parameters(tool_def.get("parameters", {})),
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

    path.parent.mkdir(parents=True, exist_ok=True)

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
    """
    return ShellToolDefinition(
        name=tool_dict["name"],
        description=tool_dict.get("description", ""),
        command=tool_dict["command"],
        parameters=_parse_parameters(tool_dict.get("parameters", {})),
        timeout=tool_dict.get("timeout", 30),
        safe_mode=tool_dict.get("safe_mode", True),
        shell=tool_dict.get("shell", True),
    )
