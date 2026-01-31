"""Daemon management CLI commands."""

from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console
from rich.panel import Panel

from tsugite.config import get_xdg_write_path

daemon_app = typer.Typer(help="Daemon management commands")
console = Console()


def _get_questionary_style():
    """Lazy load questionary style."""
    from prompt_toolkit.styles import Style

    return Style(
        [
            ("qmark", "fg:ansicyan bold"),
            ("question", "bold"),
            ("answer", "fg:ansicyan bold"),
            ("pointer", "fg:ansicyan bold"),
            ("highlighted", "fg:ansiwhite bg:ansiblue"),
            ("selected", "fg:ansicyan"),
            ("separator", "fg:ansiblack"),
            ("instruction", ""),
            ("text", ""),
        ]
    )


def _prompt_bot_name(style) -> str:
    """Prompt for bot name."""
    import questionary

    return questionary.text(
        "Bot name (e.g., 'my-assistant'):",
        default="my-bot",
        style=style,
    ).ask()


def _prompt_token_env_var(style) -> str:
    """Prompt for token environment variable name."""
    import questionary

    return questionary.text(
        "Environment variable name for Discord token:",
        default="DISCORD_BOT_TOKEN",
        style=style,
    ).ask()


def _prompt_command_prefix(style) -> str:
    """Prompt for command prefix."""
    import questionary

    return questionary.text(
        "Command prefix:",
        default="!",
        style=style,
    ).ask()


def _prompt_dm_policy(style) -> str:
    """Prompt for DM policy."""
    import questionary

    console.print("\n[bold]DM Policy:[/bold]")
    console.print("  • [cyan]allowlist[/cyan]: Only respond to users in allow list (recommended)")
    console.print("  • [cyan]open[/cyan]: Respond to any DM (use with caution)")

    return questionary.select(
        "Select DM policy:",
        choices=["allowlist (Recommended)", "open"],
        style=style,
    ).ask().split()[0]  # Extract just "allowlist" or "open"


def _prompt_allowed_users(style) -> List[str]:
    """Prompt for allowed Discord user IDs."""
    import questionary

    console.print("\n[bold]Allowed Users:[/bold]")
    console.print("[dim]To get your Discord user ID:[/dim]")
    console.print("  1. Enable Developer Mode in Discord settings")
    console.print("  2. Right-click your username → Copy User ID")
    console.print()

    users_input = questionary.text(
        "Enter Discord user ID(s) (comma-separated):",
        style=style,
    ).ask()

    if not users_input:
        return []

    return [u.strip() for u in users_input.split(",") if u.strip()]


def _prompt_workspace_setup(style) -> tuple[str, Path, str]:
    """Prompt for workspace setup.

    Returns:
        Tuple of (workspace_name, workspace_path, agent_file)
    """
    import questionary

    from tsugite.workspace import WorkspaceManager

    manager = WorkspaceManager()
    workspaces = manager.list_workspaces()

    if workspaces:
        console.print("\n[bold]Workspace Setup:[/bold]")
        choices = [f"Use existing: {name}" for name in workspaces] + ["Create new workspace"]

        choice = questionary.select(
            "Select workspace:",
            choices=choices,
            style=style,
        ).ask()

        if choice and choice.startswith("Use existing:"):
            workspace_name = choice.split(": ")[1]
            workspace_path = manager.find_workspace_path(workspace_name)
            console.print(f"[dim]Using workspace at: {workspace_path}[/dim]")
        else:
            workspace_name, workspace_path = _create_new_workspace(style)
    else:
        console.print("\n[bold]Workspace Setup:[/bold]")
        console.print("[dim]No existing workspaces found. Creating new one...[/dim]")
        workspace_name, workspace_path = _create_new_workspace(style)

    agent_file = questionary.text(
        "Agent name (e.g., 'default', 'assistant', or path to .md file):",
        default="default",
        style=style,
    ).ask()

    return workspace_name, workspace_path, agent_file


def _create_new_workspace(style) -> tuple[str, Path]:
    """Create a new workspace interactively.

    Returns:
        Tuple of (workspace_name, workspace_path)
    """
    import questionary

    from tsugite.workspace import WorkspaceManager

    manager = WorkspaceManager()

    workspace_name = questionary.text(
        "Workspace name:",
        default="discord-bot",
        style=style,
    ).ask()

    default_path = Path.home() / ".tsugite" / "workspaces" / workspace_name
    path_input = questionary.text(
        "Workspace path:",
        default=str(default_path),
        style=style,
    ).ask()

    workspace_path = Path(path_input).expanduser()

    console.print(f"\n[dim]Creating workspace at: {workspace_path}[/dim]")

    workspace = manager.create_workspace(
        name=workspace_name,
        path=workspace_path,
        init_git=False,
    )

    console.print(f"[green]✓[/green] Created workspace: {workspace.name}")

    return workspace_name, workspace_path


def _config_to_dict(config) -> dict:
    """Convert DaemonConfig to serializable dict."""
    return {
        "state_dir": str(config.state_dir),
        "log_level": config.log_level,
        "agents": {
            name: {
                "workspace_dir": str(agent_cfg.workspace_dir),
                "agent_file": agent_cfg.agent_file,
                "memory_enabled": agent_cfg.memory_enabled,
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


def _show_generated_config(config_data: dict):
    """Display the generated config for review."""
    import yaml

    config_yaml = yaml.safe_dump(config_data, default_flow_style=False, sort_keys=False)
    console.print("\n[bold]Generated Configuration:[/bold]")
    console.print(Panel(config_yaml, border_style="cyan"))


def _show_next_steps(token_env_var: str, bot_name: str):
    """Display next steps for the user."""
    console.print("\n" + "=" * 60)
    console.print("[bold green]Setup Complete![/bold green]")
    console.print("=" * 60)

    console.print("\n[bold]Next Steps:[/bold]")
    console.print(f"  1. Set your Discord token: [cyan]export {token_env_var}=\"your-token\"[/cyan]")
    console.print("  2. Start the daemon: [cyan]tsugite daemon[/cyan]")
    console.print("  3. DM your bot with: [cyan]!hello[/cyan]")

    console.print("\n[bold]Create a Discord Bot:[/bold]")
    console.print("  1. Go to: [cyan]https://discord.com/developers/applications[/cyan]")
    console.print("  2. Click 'New Application' and give it a name")
    console.print("  3. Go to [bold]Installation[/bold] → Set Install Link to [cyan]None[/cyan] → Save")
    console.print("  4. Go to [bold]Bot[/bold] tab:")
    console.print("     • Click 'Reset Token' → Copy the token")
    console.print("     • Disable 'Public Bot' (optional, for private use)")
    console.print("     • Enable 'Message Content Intent' [yellow](required)[/yellow]")
    console.print("     • Save")
    console.print("  5. Go to [bold]OAuth2 → URL Generator[/bold]:")
    console.print("     • Scopes: [cyan]bot[/cyan]")
    console.print("     • Bot Permissions:")
    console.print("       - Send Messages")
    console.print("       - Read Message History")
    console.print("       - Manage Messages")
    console.print("       - Create Public Threads")
    console.print("       - Create Private Threads")
    console.print("       - Send Messages in Threads")
    console.print("  6. Copy the generated URL and open it to invite the bot")

    console.print()


@daemon_app.command("init")
def init_daemon(
    force: bool = typer.Option(False, "--force", "-f", help="Overwrite existing configuration"),
):
    """Interactive setup wizard for Discord bot."""
    import yaml

    from tsugite.daemon.config import load_daemon_config

    console.print(
        Panel(
            "[bold]Discord Bot Setup Wizard[/bold]\n\n"
            "This wizard will help you configure a Discord bot for tsugite daemon.\n"
            "You'll need a Discord bot token from the Discord Developer Portal.",
            border_style="blue",
            title="Daemon Setup",
        )
    )

    config_path = get_xdg_write_path("daemon.yaml")

    existing_config = None
    if config_path.exists():
        console.print(f"\n[yellow]Existing config found at: {config_path}[/yellow]")
        if not force:
            if not typer.confirm("Add a new bot to existing config?", default=True):
                console.print("[dim]Use --force to overwrite existing config[/dim]")
                raise typer.Exit(0)
            try:
                existing_config = load_daemon_config(config_path)
            except Exception as e:
                console.print(f"[yellow]Warning: Could not load existing config: {e}[/yellow]")

    style = _get_questionary_style()

    console.print("\n[bold cyan]Bot Configuration[/bold cyan]")

    bot_name = _prompt_bot_name(style)
    if not bot_name:
        raise typer.Exit(1)

    token_env_var = _prompt_token_env_var(style)
    if not token_env_var:
        raise typer.Exit(1)

    command_prefix = _prompt_command_prefix(style)
    if not command_prefix:
        raise typer.Exit(1)

    workspace_name, workspace_path, agent_file = _prompt_workspace_setup(style)
    if not workspace_name:
        raise typer.Exit(1)

    dm_policy = _prompt_dm_policy(style)
    if not dm_policy:
        raise typer.Exit(1)

    allowed_users = []
    if dm_policy == "allowlist":
        allowed_users = _prompt_allowed_users(style)

    # Use bot name as the agent config key for clarity
    agent_config_name = bot_name

    if existing_config and not force:
        config_data = _config_to_dict(existing_config)
    else:
        config_data = {
            "state_dir": str(Path.home() / ".tsugite-daemon"),
            "log_level": "info",
            "agents": {},
            "discord_bots": [],
        }

    config_data["agents"][agent_config_name] = {
        "workspace_dir": str(workspace_path),
        "agent_file": agent_file,
        "memory_enabled": True,
    }

    bot_config = {
        "name": bot_name,
        "token": f"${{{token_env_var}}}",
        "agent": agent_config_name,
        "command_prefix": command_prefix,
        "dm_policy": dm_policy,
    }
    if allowed_users:
        bot_config["allow_from"] = allowed_users

    config_data["discord_bots"].append(bot_config)

    _show_generated_config(config_data)

    if not typer.confirm("Save this configuration?", default=True):
        console.print("[yellow]Configuration not saved.[/yellow]")
        raise typer.Exit(0)

    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config_data, f, default_flow_style=False, sort_keys=False)

    console.print(f"[green]✓[/green] Configuration saved to: {config_path}")

    _show_next_steps(token_env_var, bot_name)


@daemon_app.callback(invoke_without_command=True)
def daemon_main(
    ctx: typer.Context,
    config: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Path to daemon config (default: ~/.config/tsugite/daemon.yaml)"
    ),
):
    """Start tsugite daemon for Discord/Telegram bots."""
    if ctx.invoked_subcommand is not None:
        return

    import asyncio

    from tsugite.daemon.gateway import run_daemon

    try:
        asyncio.run(run_daemon(config))
    except KeyboardInterrupt:
        console.print("\n[yellow]Daemon stopped[/yellow]")


__all__ = ["daemon_app"]
