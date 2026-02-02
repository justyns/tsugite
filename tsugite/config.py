"""Tsugite configuration management."""

import json
import os
from pathlib import Path
from typing import Callable, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

# XDG Base Directory utilities


def get_xdg_config_path(filename: str) -> Path:
    """Get XDG-compliant config file path.

    Checks locations in order of precedence:
    1. $XDG_CONFIG_HOME/tsugite/{filename} (if XDG_CONFIG_HOME is set)
    2. ~/.config/tsugite/{filename} (XDG default)

    Returns the first existing file, or the preferred location for new files.

    Args:
        filename: Name of the config file (e.g., "config.json", "mcp.json")

    Returns:
        Path to config file
    """
    xdg_config = os.environ.get("XDG_CONFIG_HOME")
    if xdg_config:
        xdg_path = Path(xdg_config) / "tsugite" / filename
        if xdg_path.exists():
            return xdg_path

    default_path = Path.home() / ".config" / "tsugite" / filename
    if default_path.exists():
        return default_path

    if xdg_config:
        return Path(xdg_config) / "tsugite" / filename
    return default_path


def get_xdg_write_path(filename: str) -> Path:
    """Get config path for writing operations.

    Uses XDG location ($XDG_CONFIG_HOME or ~/.config).

    Args:
        filename: Name of the config file

    Returns:
        Path where config should be written
    """
    xdg_config = os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config"))
    return Path(xdg_config) / "tsugite" / filename


def get_xdg_cache_path(subdir: str = "") -> Path:
    """Get XDG-compliant cache directory path.

    Uses XDG Base Directory specification for cache:
    - $XDG_CACHE_HOME/tsugite/{subdir} (if XDG_CACHE_HOME is set)
    - ~/.cache/tsugite/{subdir} (XDG default)

    Args:
        subdir: Optional subdirectory within tsugite cache (e.g., "attachments")

    Returns:
        Path to cache directory
    """
    xdg_base = os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache"))
    path = Path(xdg_base) / "tsugite"
    if subdir:
        path = path / subdir
    return path


def get_xdg_data_path(subdir: str = "") -> Path:
    """Get XDG-compliant data directory path.

    Uses XDG Base Directory specification for data:
    - $XDG_DATA_HOME/tsugite/{subdir} (if XDG_DATA_HOME is set)
    - ~/.local/share/tsugite/{subdir} (XDG default)

    Args:
        subdir: Optional subdirectory within tsugite data (e.g., "history")

    Returns:
        Path to data directory
    """
    xdg_base = os.environ.get("XDG_DATA_HOME", str(Path.home() / ".local/share"))
    path = Path(xdg_base) / "tsugite"
    if subdir:
        path = path / subdir
    return path


class Config(BaseModel):
    """Tsugite configuration."""

    model_config = ConfigDict(
        extra="allow",
        str_strip_whitespace=True,
    )

    default_model: Optional[str] = None
    model_aliases: Dict[str, str] = Field(default_factory=dict)
    default_base_agent: Optional[str] = None
    default_workspace: Optional[str] = None
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


def update_config(path: Optional[Path], updater: Callable[[Config], None]) -> None:
    """Load config, apply update function, and save.

    This is a generic helper for modifying configuration. Use with a lambda
    or function that modifies the config object in place.

    Args:
        path: Path to config.json file. If None, uses default path
        updater: Function that modifies the config object

    Examples:
        # Set default model
        update_config(None, lambda cfg: setattr(cfg, "default_model", "ollama:qwen2.5-coder:7b"))

        # Add model alias
        update_config(None, lambda cfg: cfg.model_aliases.update({"cheap": "openai:gpt-4o-mini"}))

        # Set chat theme
        update_config(None, lambda cfg: setattr(cfg, "chat_theme", "nord"))
    """
    config = load_config(path)
    updater(config)
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


def get_chat_theme(path: Optional[Path] = None) -> str:
    """Get the chat UI theme from configuration.

    Args:
        path: Path to config.json file. If None, uses default path

    Returns:
        Theme name (defaults to "gruvbox")
    """
    config = load_config(path)
    return config.chat_theme
