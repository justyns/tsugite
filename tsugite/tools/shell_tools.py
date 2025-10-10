"""Shell-based custom tools for Tsugite agents.

This module provides a lightweight system for defining tools that wrap shell commands,
allowing users to create custom tools without writing Python code.
"""

import shlex
import subprocess
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from tsugite.tools import tool


@dataclass
class ShellToolParameter:
    """Definition of a parameter for a shell tool."""

    name: str
    type: str  # "str", "int", "bool", "float"
    description: str = ""
    required: bool = False
    default: Any = None
    flag: Optional[str] = None  # For boolean flags (e.g., "-s" for case_sensitive)

    def validate(self, value: Any) -> Any:
        """Validate and convert parameter value to correct type."""
        if value is None:
            if self.required:
                raise ValueError(f"Required parameter '{self.name}' is missing")
            return self.default

        # Type conversion
        if self.type == "str":
            return str(value)
        elif self.type == "int":
            try:
                return int(value)
            except (ValueError, TypeError):
                raise ValueError(f"Parameter '{self.name}' must be an integer")
        elif self.type == "float":
            try:
                return float(value)
            except (ValueError, TypeError):
                raise ValueError(f"Parameter '{self.name}' must be a float")
        elif self.type == "bool":
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                return value.lower() in ("true", "yes", "1", "y")
            return bool(value)
        else:
            raise ValueError(f"Unknown parameter type: {self.type}")


@dataclass
class ShellToolDefinition:
    """Definition of a shell-based tool."""

    name: str
    description: str
    command: str  # Template string with {param} placeholders
    parameters: Dict[str, ShellToolParameter] = field(default_factory=dict)
    timeout: int = 30
    safe_mode: bool = True  # Use run_safe vs run
    shell: bool = True  # Execute via shell


def interpolate_command(command_template: str, params: Dict[str, Any]) -> str:
    """Interpolate parameters into command template.

    Args:
        command_template: Command string with {param} placeholders
        params: Dictionary of parameter values

    Returns:
        Interpolated command string

    Example:
        >>> interpolate_command("rg {pattern} {path}", {"pattern": "test", "path": "."})
        'rg test .'
    """
    # Handle boolean flags specially
    interpolated = command_template

    # First, handle any boolean parameters with flags
    for key, value in params.items():
        # If value is boolean and the key appears with a flag placeholder
        if isinstance(value, bool):
            # Replace {key} with flag or empty string
            flag_value = params.get(f"_flag_{key}", "")
            if value and flag_value:
                interpolated = interpolated.replace(f"{{{key}}}", flag_value)
            else:
                interpolated = interpolated.replace(f"{{{key}}}", "")

    # Then handle regular parameter substitution
    try:
        interpolated = interpolated.format(**params)
    except KeyError as e:
        raise ValueError(f"Missing required parameter in command template: {e}")

    return interpolated.strip()


def create_shell_tool_function(definition: ShellToolDefinition):
    """Create a Python function that executes a shell tool.

    Args:
        definition: Shell tool definition

    Returns:
        Callable function that can be registered as a tool
    """
    import inspect

    def shell_tool_func(**kwargs) -> str:
        """Dynamically generated shell tool function."""
        # Validate and process parameters
        processed_params = {}

        for param_name, param_def in definition.parameters.items():
            value = kwargs.get(param_name)
            validated_value = param_def.validate(value)

            # Handle boolean flags
            if param_def.type == "bool" and param_def.flag:
                if validated_value:
                    processed_params[f"_flag_{param_name}"] = param_def.flag
                processed_params[param_name] = validated_value
            else:
                processed_params[param_name] = validated_value

        # Interpolate command
        try:
            command = interpolate_command(definition.command, processed_params)
        except Exception as e:
            raise RuntimeError(f"Failed to build command: {e}")

        # Execute command
        try:
            if definition.shell:
                result = subprocess.run(
                    command,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=definition.timeout,
                    check=False,
                )
            else:
                cmd_parts = shlex.split(command)
                result = subprocess.run(
                    cmd_parts,
                    capture_output=True,
                    text=True,
                    timeout=definition.timeout,
                    check=False,
                )

            # Combine stdout and stderr
            output = ""
            if result.stdout:
                output += result.stdout
            if result.stderr:
                if output:
                    output += "\n" + result.stderr
                else:
                    output = result.stderr

            if result.returncode != 0:
                output += f"\n[Exit code: {result.returncode}]"

            return output or "[No output]"

        except subprocess.TimeoutExpired:
            raise RuntimeError(f"Command timed out after {definition.timeout} seconds")
        except Exception as e:
            raise RuntimeError(f"Command execution failed: {e}")

    # Set function metadata for tool registration
    shell_tool_func.__name__ = definition.name
    shell_tool_func.__doc__ = definition.description

    # Build proper signature with actual parameters
    # This is critical for @tool decorator to work correctly
    params = []
    annotations = {}

    for param_name, param_def in definition.parameters.items():
        # Map our types to Python types
        if param_def.type == "str":
            param_type = str
        elif param_def.type == "int":
            param_type = int
        elif param_def.type == "bool":
            param_type = bool
        elif param_def.type == "float":
            param_type = float
        else:
            param_type = str

        annotations[param_name] = param_type

        # Create Parameter with default if not required
        if param_def.required:
            param = inspect.Parameter(
                param_name,
                inspect.Parameter.KEYWORD_ONLY,
                annotation=param_type,
            )
        else:
            param = inspect.Parameter(
                param_name,
                inspect.Parameter.KEYWORD_ONLY,
                default=param_def.default,
                annotation=param_type,
            )
        params.append(param)

    # Create new signature and assign it
    new_sig = inspect.Signature(params)
    shell_tool_func.__signature__ = new_sig
    shell_tool_func.__annotations__ = annotations

    return shell_tool_func


def register_shell_tool(definition: ShellToolDefinition) -> None:
    """Register a shell tool definition as a tool.

    Args:
        definition: Shell tool definition to register
    """
    func = create_shell_tool_function(definition)
    tool(func)


def register_shell_tools(definitions: List[ShellToolDefinition]) -> None:
    """Register multiple shell tool definitions.

    Args:
        definitions: List of shell tool definitions to register
    """
    for definition in definitions:
        register_shell_tool(definition)
