"""Tsugite configuration management."""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional


@dataclass
class Config:
    """Tsugite configuration."""

    default_model: Optional[str] = None
    model_aliases: Dict[str, str] = field(default_factory=dict)
    default_base_agent: Optional[str] = None

    def __post_init__(self):
        if self.model_aliases is None:
            self.model_aliases = {}


def get_config_path() -> Path:
    """Get the configuration file path.

    Checks locations in order of precedence:
    1. ~/.tsugite/config.json
    2. $XDG_CONFIG_HOME/tsugite/config.json (if XDG_CONFIG_HOME is set)
    3. ~/.config/tsugite/config.json (XDG default)

    Returns the first existing file, or the preferred location for new files.
    """
    home_tsugite_path = Path.home() / ".tsugite" / "config.json"
    if home_tsugite_path.exists():
        return home_tsugite_path

    xdg_config = os.environ.get("XDG_CONFIG_HOME")
    if xdg_config:
        xdg_path = Path(xdg_config) / "tsugite" / "config.json"
        if xdg_path.exists():
            return xdg_path

    default_path = Path.home() / ".config" / "tsugite" / "config.json"
    if default_path.exists():
        return default_path

    if xdg_config:
        return Path(xdg_config) / "tsugite" / "config.json"
    return default_path


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
        with open(path, "r") as f:
            data = json.load(f)

        return Config(
            default_model=data.get("default_model"),
            model_aliases=data.get("model_aliases", {}),
            default_base_agent=data.get("default_base_agent"),
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

    with open(path, "w") as f:
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
