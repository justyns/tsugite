"""Daemon configuration models."""

import os
from pathlib import Path
from typing import Dict, List, Literal, Optional

import yaml
from pydantic import BaseModel, Field


class AgentConfig(BaseModel):
    """Configuration for a single agent."""

    workspace_dir: Path
    agent_file: str
    memory_enabled: bool = True
    memory_inject_days: int = 2
    context_limit: int = 128000  # Model's context window (tokens)
    memory_extraction_interval: int = 10  # Run memory extraction every N messages


class DiscordBotConfig(BaseModel):
    """Configuration for a single Discord bot."""

    name: str
    token: str
    agent: str  # References agents key
    command_prefix: str = "!"
    dm_policy: Literal["allowlist", "open"] = "allowlist"
    allow_from: List[str] = Field(default_factory=list)


class DaemonConfig(BaseModel):
    """Main daemon configuration."""

    state_dir: Path = Field(default_factory=lambda: Path.home() / ".tsugite-daemon")
    log_level: str = "info"
    agents: Dict[str, AgentConfig]
    discord_bots: List[DiscordBotConfig] = Field(default_factory=list)


def load_daemon_config(path: Optional[Path] = None) -> DaemonConfig:
    """Load daemon config from YAML.

    Args:
        path: Path to daemon config file. If None, uses default XDG location

    Returns:
        DaemonConfig instance

    Raises:
        ValueError: If config file not found or invalid
    """
    if path is None:
        from tsugite.config import get_xdg_config_path

        path = get_xdg_config_path("daemon.yaml", legacy_dir=False)

    if not path.exists():
        raise ValueError(f"Daemon config not found: {path}")

    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    # Expand environment variables in tokens
    for bot in data.get("discord_bots", []):
        if "token" in bot:
            bot["token"] = os.path.expandvars(bot["token"])

    # Expand workspace_dir paths
    for agent_name, agent_data in data.get("agents", {}).items():
        if "workspace_dir" in agent_data:
            agent_data["workspace_dir"] = Path(agent_data["workspace_dir"]).expanduser()

    # Expand state_dir
    if "state_dir" in data:
        data["state_dir"] = Path(data["state_dir"]).expanduser()

    return DaemonConfig.model_validate(data)
