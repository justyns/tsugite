"""Daemon configuration models."""

import os
from pathlib import Path
from typing import Dict, List, Literal, Optional

import yaml
from pydantic import BaseModel, Field, model_validator


def _get_default_state_dir() -> Path:
    """Get the default state directory for daemon."""
    from tsugite.config import get_xdg_data_path

    return get_xdg_data_path("daemon")


class AgentConfig(BaseModel):
    """Configuration for a single agent."""

    workspace_dir: Path
    agent_file: str
    context_limit: int = 128000  # Model's context window (tokens)
    model: Optional[str] = None
    compaction_model: Optional[str] = None
    max_turns: Optional[int] = None


class DiscordBotConfig(BaseModel):
    """Configuration for a single Discord bot."""

    name: str
    token: str
    agent: str  # References agents key
    command_prefix: str = "!"
    dm_policy: Literal["allowlist", "open"] = "allowlist"
    allow_from: List[str] = Field(default_factory=list)


class NotificationChannelConfig(BaseModel):
    """Configuration for a notification channel (discord DM or webhook)."""

    type: Literal["discord", "webhook"]
    # Discord fields
    user_id: Optional[str] = None
    bot: Optional[str] = None
    # Webhook fields
    url: Optional[str] = None
    method: str = "POST"
    headers: Dict[str, str] = Field(default_factory=dict)
    body_template: Optional[str] = None

    @model_validator(mode="after")
    def _validate_required_fields(self):
        if self.type == "discord":
            if not self.user_id or not self.bot:
                raise ValueError("Discord notification channels require 'user_id' and 'bot'")
        elif self.type == "webhook":
            if not self.url:
                raise ValueError("Webhook notification channels require 'url'")
        return self


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
    notification_channels: Dict[str, NotificationChannelConfig] = Field(default_factory=dict)
    identity_links: Dict[str, List[str]] = Field(default_factory=dict)


def _expand_env_vars(data: dict, *keys: str) -> None:
    """Expand environment variables in the specified string-valued keys in-place."""
    for key in keys:
        if key in data and isinstance(data[key], str):
            data[key] = os.path.expandvars(data[key])


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

    for bot in data.get("discord_bots", []):
        _expand_env_vars(bot, "token")

    http_data = data.get("http")
    if http_data and "auth_tokens" in http_data:
        http_data["auth_tokens"] = [os.path.expandvars(t) for t in http_data["auth_tokens"]]

    for agent_data in data.get("agents", {}).values():
        if "workspace_dir" in agent_data:
            agent_data["workspace_dir"] = Path(agent_data["workspace_dir"]).expanduser()

    for channel in data.get("notification_channels", {}).values():
        _expand_env_vars(channel, "url", "body_template")
        if "headers" in channel:
            channel["headers"] = {k: os.path.expandvars(v) for k, v in channel["headers"].items()}

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

    # mode="json" ensures Path objects become strings
    config_data = config.model_dump(exclude_none=True, mode="json")

    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config_data, f, default_flow_style=False, sort_keys=False)

    return path
