"""Daemon configuration models."""

import os
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import yaml
from pydantic import BaseModel, Field, model_validator


def _get_default_state_dir() -> Path:
    """Get the default state directory for daemon."""
    from tsugite.config import get_xdg_data_path

    return get_xdg_data_path("daemon")


class AutoCompactConfig(BaseModel):
    """Scheduled auto-compaction settings."""

    schedule: Optional[str] = None  # Cron expression, e.g. "0 0 * * *"
    min_turns: int = 1  # Skip if fewer turns since last compaction


class AgentConfig(BaseModel):
    """Configuration for a single agent."""

    workspace_dir: Path
    agent_file: str
    context_limit: Optional[int] = None  # Explicit override; auto-detected from model if unset
    model: Optional[str] = None
    compaction_model: Optional[str] = None
    max_turns: Optional[int] = None
    timezone: str = ""  # IANA timezone for display (e.g. "America/Chicago")
    auto_compact: Optional[AutoCompactConfig] = None


class DiscordBotConfig(BaseModel):
    """Configuration for a single Discord bot."""

    name: str
    agent: str  # References agents key
    token_secret: Optional[str] = None  # Resolved via tsugite.secrets.get_backend().get()
    token_file: Optional[Path] = None  # File path containing the token
    command_prefix: str = "!"
    guild_id: Optional[str] = None  # Sync app commands to this guild only (instant; good for dev)
    dm_policy: Literal["allowlist", "open"] = "allowlist"
    allow_from: List[str] = Field(default_factory=list)
    # DMs from a Discord user route to the latest non-finished session tagged with
    # metadata.session_name == this value (auto-creates one if absent). Channels and threads
    # keep their existing shared-team-session behavior. The name is preserved across compaction.
    # Set to "" to fall back to the user's default-interactive session.
    session_name: str = "discord"

    @model_validator(mode="after")
    def _validate_token_source(self):
        has_secret = self.token_secret is not None
        has_file = self.token_file is not None
        if has_secret == has_file:
            raise ValueError(f"DiscordBotConfig {self.name!r}: must set exactly one of token_secret, token_file")
        return self

    def resolve_token(self) -> str:
        """Resolve the bot token from its configured source.

        Called by the Discord adapter at bot start, after the secrets backend
        has been configured by `configure_from_daemon()`.
        """
        if self.token_secret is not None:
            from tsugite.secrets import get_backend

            value = get_backend().get(self.token_secret)
            if value is None:
                raise RuntimeError(
                    f"Discord bot {self.name!r}: secret {self.token_secret!r} not found in secrets backend"
                )
            return value
        if self.token_file is not None:
            return self.token_file.expanduser().read_text(encoding="utf-8").strip()
        raise RuntimeError(f"Discord bot {self.name!r}: no token source configured")


class NotificationChannelConfig(BaseModel):
    """Configuration for a notification channel (discord DM or webhook)."""

    type: Literal["discord", "webhook", "web-push"]
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
        # web-push: no required fields — subscriptions managed via API
        return self


class HTTPConfig(BaseModel):
    """Configuration for the HTTP API server."""

    enabled: bool = False
    host: str = "127.0.0.1"
    port: int = 8374
    max_workspace_file_size: int = 1024 * 1024  # 1MB


class DaemonConfig(BaseModel):
    """Main daemon configuration."""

    state_dir: Path = Field(default_factory=_get_default_state_dir)
    log_level: str = "info"
    log_file: Optional[Path] = None
    log_to_console: bool = True
    agents: Dict[str, AgentConfig]
    discord_bots: List[DiscordBotConfig] = Field(default_factory=list)
    http: Optional[HTTPConfig] = None
    notification_channels: Dict[str, NotificationChannelConfig] = Field(default_factory=dict)
    identity_links: Dict[str, List[str]] = Field(default_factory=dict)
    plugins: Dict[str, Dict[str, Any]] = Field(default_factory=dict)


def _expand_env_vars(data: dict, *keys: str) -> None:
    """Expand environment variables in the specified string-valued keys in-place."""
    for key in keys:
        if key in data and isinstance(data[key], str):
            data[key] = os.path.expandvars(data[key])


def _expand_paths(data: dict, *keys: str) -> None:
    """Expand user home directory in the specified path-valued keys in-place."""
    for key in keys:
        if key in data and data[key]:
            data[key] = Path(data[key]).expanduser()


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
        if "token" in bot:
            raise ValueError(
                f"Discord bot {bot.get('name', '?')!r}: plaintext 'token:' is no longer supported. "
                "Use 'token_secret: <name>' (resolved via tsugite secrets store) "
                "or 'token_file: <path>' instead. "
                "Migrate with: tsu secrets set <name> (then update daemon.yaml)."
            )
        _expand_env_vars(bot, "token_file")
        _expand_paths(bot, "token_file")

    for agent_data in data.get("agents", {}).values():
        if "workspace_dir" in agent_data:
            agent_data["workspace_dir"] = Path(agent_data["workspace_dir"]).expanduser()

    for channel in data.get("notification_channels", {}).values():
        _expand_env_vars(channel, "url", "body_template")
        if "headers" in channel:
            channel["headers"] = {k: os.path.expandvars(v) for k, v in channel["headers"].items()}

    _expand_paths(data, "state_dir", "log_file")

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
