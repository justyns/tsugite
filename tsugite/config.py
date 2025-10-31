"""Tsugite configuration management."""

import json
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from .xdg import get_xdg_config_path


class Config(BaseModel):
    """Tsugite configuration."""

    model_config = ConfigDict(
        extra="allow",
        str_strip_whitespace=True,
    )

    default_model: Optional[str] = None
    model_aliases: Dict[str, str] = Field(default_factory=dict)
    default_base_agent: Optional[str] = None
    chat_theme: str = "gruvbox"
    history_enabled: bool = True
    history_dir: Optional[Path] = None
    machine_name: Optional[str] = None
    max_history_days: Optional[int] = None
    auto_context_enabled: bool = True
    auto_context_files: List[str] = Field(default_factory=lambda: [".tsugite/CONTEXT.md", "AGENTS.md", "CLAUDE.md"])
    auto_context_include_global: bool = True


def get_config_path() -> Path:
    return get_xdg_config_path("config.json")


def load_config(path: Optional[Path] = None) -> Config:
    """Load Tsugite configuration from JSON file.

    Args:
        path: Path to config.json file. If None, uses default path

    Returns:
        Config object with loaded settings. Returns default config if file doesn't exist.
    """
    if path is None:
        path = get_config_path()

    if not path.exists():
        return Config()

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Convert history_dir string to Path if present
        if "history_dir" in data and data["history_dir"]:
            data["history_dir"] = Path(data["history_dir"])

        # Use Pydantic's model_validate for validation and construction
        return Config.model_validate(data)

    except json.JSONDecodeError as e:
        print(f"Warning: Failed to parse config at {path}: {e}")
        return Config()
    except Exception as e:
        print(f"Warning: Failed to load config from {path}: {e}")
        return Config()


def save_config(config: Config, path: Optional[Path] = None) -> None:
    """Save Tsugite configuration to JSON file.

    Args:
        config: Config object to save
        path: Path to config.json file. If None, uses default path

    Raises:
        IOError: If file cannot be written
    """
    if path is None:
        path = get_config_path()

    path.parent.mkdir(parents=True, exist_ok=True)

    # Use Pydantic's model_dump to serialize, excluding None values for cleaner output
    config_data = config.model_dump(exclude_none=True, mode="json")

    # Convert Path to string for JSON serialization
    if "history_dir" in config_data and config_data["history_dir"]:
        config_data["history_dir"] = str(config_data["history_dir"])

    # Remove empty model_aliases dict for cleaner output
    if "model_aliases" in config_data and not config_data["model_aliases"]:
        del config_data["model_aliases"]

    with open(path, "w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=2)


def set_default_model(model: str, path: Optional[Path] = None) -> None:
    """Set the default model in configuration.

    Args:
        model: Model string (e.g., "ollama:qwen2.5-coder:7b")
        path: Path to config.json file. If None, uses default path
    """
    config = load_config(path)
    config.default_model = model
    save_config(config, path)


def add_model_alias(alias: str, model: str, path: Optional[Path] = None) -> None:
    """Add or update a model alias in configuration.

    Args:
        alias: Alias name (e.g., "cheap")
        model: Model string (e.g., "openai:gpt-4o-mini")
        path: Path to config.json file. If None, uses default path
    """
    config = load_config(path)
    config.model_aliases[alias] = model
    save_config(config, path)


def remove_model_alias(alias: str, path: Optional[Path] = None) -> bool:
    """Remove a model alias from configuration.

    Args:
        alias: Alias name to remove
        path: Path to config.json file. If None, uses default path

    Returns:
        True if alias was removed, False if it didn't exist
    """
    config = load_config(path)
    if alias in config.model_aliases:
        del config.model_aliases[alias]
        save_config(config, path)
        return True
    return False


def get_model_alias(alias: str, path: Optional[Path] = None) -> Optional[str]:
    """Get the model string for an alias.

    Args:
        alias: Alias name to look up
        path: Path to config.json file. If None, uses default path

    Returns:
        Model string if alias exists, None otherwise
    """
    config = load_config(path)
    return config.model_aliases.get(alias)


def set_default_base_agent(base_agent: Optional[str], path: Optional[Path] = None) -> None:
    """Set the default base agent in configuration.

    Args:
        base_agent: Base agent name (e.g., "default") or None to disable
        path: Path to config.json file. If None, uses default path
    """
    config = load_config(path)
    config.default_base_agent = base_agent
    save_config(config, path)


def set_chat_theme(theme: str, path: Optional[Path] = None) -> None:
    """Set the chat UI theme in configuration.

    Args:
        theme: Theme name (e.g., "gruvbox", "nord", "tokyo-night")
        path: Path to config.json file. If None, uses default path
    """
    config = load_config(path)
    config.chat_theme = theme
    save_config(config, path)


def set_auto_context_enabled(enabled: bool, path: Optional[Path] = None) -> None:
    """Set whether auto-context is enabled in configuration.

    Args:
        enabled: True to enable auto-context, False to disable
        path: Path to config.json file. If None, uses default path
    """
    config = load_config(path)
    config.auto_context_enabled = enabled
    save_config(config, path)


def get_chat_theme(path: Optional[Path] = None) -> str:
    """Get the chat UI theme from configuration.

    Args:
        path: Path to config.json file. If None, uses default path

    Returns:
        Theme name (defaults to "gruvbox")
    """
    config = load_config(path)
    return config.chat_theme
