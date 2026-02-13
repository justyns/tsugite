"""Daemon configuration models."""

import os
from pathlib import Path
from typing import Dict, List, Literal, Optional

import yaml
from pydantic import BaseModel, Field


def _get_default_state_dir() -> Path:
    """Get the default state directory for daemon."""
    from tsugite.config import get_xdg_data_path

    return get_xdg_data_path("daemon")


class AgentConfig(BaseModel):
    """Configuration for a single agent."""

    workspace_dir: Path
    agent_file: str
    context_limit: int = 128000  # Model's context window (tokens)


class DiscordBotConfig(BaseModel):
    """Configuration for a single Discord bot."""

    name: str
    token: str
    agent: str  # References agents key
    command_prefix: str = "!"
    dm_policy: Literal["allowlist", "open"] = "allowlist"
    allow_from: List[str] = Field(default_factory=list)


class HTTPConfig(BaseModel):
    """Configuration for the HTTP API server."""

    enabled: bool = False
    host: str = "127.0.0.1"
    port: int = 8374
    auth_tokens: List[str] = Field(default_factory=list)


class DaemonConfig(BaseModel):
    """Main daemon configuration."""

    state_dir: Path = Field(default_factory=_get_default_state_dir)
    log_level: str = "info"
    agents: Dict[str, AgentConfig]
    discord_bots: List[DiscordBotConfig] = Field(default_factory=list)
    http: Optional[HTTPConfig] = None


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

        path = get_xdg_config_path("daemon.yaml")

    if not path.exists():
        raise ValueError(f"Daemon config not found: {path}")

    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    # Expand environment variables in tokens
    for bot in data.get("discord_bots", []):
        if "token" in bot:
            bot["token"] = os.path.expandvars(bot["token"])

    # Expand environment variables in HTTP auth tokens
    http_data = data.get("http")
    if http_data and "auth_tokens" in http_data:
        http_data["auth_tokens"] = [os.path.expandvars(t) for t in http_data["auth_tokens"]]

    # Expand workspace_dir paths
    for agent_name, agent_data in data.get("agents", {}).items():
        if "workspace_dir" in agent_data:
            agent_data["workspace_dir"] = Path(agent_data["workspace_dir"]).expanduser()

    # Expand state_dir
    if "state_dir" in data:
        data["state_dir"] = Path(data["state_dir"]).expanduser()

    return DaemonConfig.model_validate(data)


def save_daemon_config(config: DaemonConfig, path: Optional[Path] = None) -> Path:
    """Save daemon config to YAML file.

    Args:
        config: DaemonConfig instance to save
        path: Path to save config. If None, uses default XDG location

    Returns:
        Path where config was saved
    """
    if path is None:
        from tsugite.config import get_xdg_write_path

        path = get_xdg_write_path("daemon.yaml")

    path.parent.mkdir(parents=True, exist_ok=True)

    config_data = {
        "state_dir": str(config.state_dir),
        "log_level": config.log_level,
        "agents": {
            name: {
                "workspace_dir": str(agent_cfg.workspace_dir),
                "agent_file": agent_cfg.agent_file,
                "context_limit": agent_cfg.context_limit,
            }
            for name, agent_cfg in config.agents.items()
        },
        "discord_bots": [
            {
                "name": bot.name,
                "token": bot.token,
                "agent": bot.agent,
                "command_prefix": bot.command_prefix,
                "dm_policy": bot.dm_policy,
                "allow_from": bot.allow_from,
            }
            for bot in config.discord_bots
        ],
    }

    if config.http:
        config_data["http"] = {
            "enabled": config.http.enabled,
            "host": config.http.host,
            "port": config.http.port,
            "auth_tokens": config.http.auth_tokens,
        }

    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config_data, f, default_flow_style=False, sort_keys=False)

    return path
