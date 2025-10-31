"""XDG Base Directory utilities for config file management."""

import os
from pathlib import Path


def get_xdg_config_path(filename: str, legacy_dir: bool = True) -> Path:
    """Get XDG-compliant config file path.

    Checks locations in order of precedence:
    1. ~/.tsugite/{filename} (if legacy_dir is True)
    2. $XDG_CONFIG_HOME/tsugite/{filename} (if XDG_CONFIG_HOME is set)
    3. ~/.config/tsugite/{filename} (XDG default)

    Returns the first existing file, or the preferred location for new files.

    Args:
        filename: Name of the config file (e.g., "config.json", "mcp.json")
        legacy_dir: Whether to check ~/.tsugite first for backwards compatibility

    Returns:
        Path to config file
    """
    if legacy_dir:
        home_tsugite_path = Path.home() / ".tsugite" / filename
        if home_tsugite_path.exists():
            return home_tsugite_path

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


def get_xdg_write_path(filename: str, legacy_dir: bool = True) -> Path:
    """Get config path for writing operations.

    Respects existing config location:
    - If ~/.tsugite/{filename} exists and legacy_dir is True, use it
    - Otherwise, use XDG location ($XDG_CONFIG_HOME or ~/.config)

    Args:
        filename: Name of the config file
        legacy_dir: Whether to check ~/.tsugite first for backwards compatibility

    Returns:
        Path where config should be written
    """
    if legacy_dir:
        home_tsugite_path = Path.home() / ".tsugite" / filename
        if home_tsugite_path.exists():
            return home_tsugite_path

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
    xdg_cache = os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache"))
    cache_path = Path(xdg_cache) / "tsugite"
    if subdir:
        cache_path = cache_path / subdir
    return cache_path


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
    xdg_data = os.environ.get("XDG_DATA_HOME", str(Path.home() / ".local" / "share"))
    data_path = Path(xdg_data) / "tsugite"
    if subdir:
        data_path = data_path / subdir
    return data_path
