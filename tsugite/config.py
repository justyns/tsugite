"""Tsugite configuration management."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

from .xdg import get_xdg_config_path


@dataclass
class Config:
    """Tsugite configuration."""

    default_model: Optional[str] = None
    model_aliases: Dict[str, str] = field(default_factory=dict)
    default_base_agent: Optional[str] = None
    chat_theme: str = "gruvbox"
    history_enabled: bool = True
    history_dir: Optional[Path] = None
    machine_name: Optional[str] = None
    max_history_days: Optional[int] = None

    def __post_init__(self):
        if self.model_aliases is None:
            self.model_aliases = {}


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

        history_dir = None
        if "history_dir" in data and data["history_dir"]:
            history_dir = Path(data["history_dir"])

        return Config(
            default_model=data.get("default_model"),
            model_aliases=data.get("model_aliases", {}),
            default_base_agent=data.get("default_base_agent"),
            chat_theme=data.get("chat_theme", "gruvbox"),
            history_enabled=data.get("history_enabled", True),
            history_dir=history_dir,
            machine_name=data.get("machine_name"),
            max_history_days=data.get("max_history_days"),
        )

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

    config_data = {}

    if config.default_model:
        config_data["default_model"] = config.default_model

    if config.model_aliases:
        config_data["model_aliases"] = config.model_aliases

    if config.default_base_agent is not None:
        config_data["default_base_agent"] = config.default_base_agent

    if config.chat_theme:
        config_data["chat_theme"] = config.chat_theme

    # Always save history_enabled (defaults to True)
    config_data["history_enabled"] = config.history_enabled

    if config.history_dir:
        config_data["history_dir"] = str(config.history_dir)

    if config.machine_name:
        config_data["machine_name"] = config.machine_name

    if config.max_history_days is not None:
        config_data["max_history_days"] = config.max_history_days

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


def get_chat_theme(path: Optional[Path] = None) -> str:
    """Get the chat UI theme from configuration.

    Args:
        path: Path to config.json file. If None, uses default path

    Returns:
        Theme name (defaults to "gruvbox")
    """
    config = load_config(path)
    return config.chat_theme
