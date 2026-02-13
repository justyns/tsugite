"""Daemon management CLI commands."""

import re
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel

from tsugite.config import get_xdg_data_path, get_xdg_write_path

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

    return (
        questionary.select(
            "Select DM policy:",
            choices=["allowlist (Recommended)", "open"],
            style=style,
        )
        .ask()
        .split()[0]
    )  # Extract just "allowlist" or "open"


def _prompt_allowed_users(style) -> list[str]:
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

    default_path = get_xdg_data_path("workspaces") / workspace_name
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
    console.print(f'  1. Set your Discord token: [cyan]export {token_env_var}="your-token"[/cyan]')
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
            "state_dir": str(get_xdg_data_path("daemon")),
            "log_level": "info",
            "agents": {},
            "discord_bots": [],
        }

    config_data["agents"][agent_config_name] = {
        "workspace_dir": str(workspace_path),
        "agent_file": agent_file,
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


def _daemon_request(
    method: str, host: str, port: int, path: str, token: Optional[str] = None, **kwargs
):
    """Make an HTTP request to the daemon, handling connection errors and non-2xx responses.

    Returns the parsed JSON response body on success, or exits on failure.
    """
    import httpx

    url = f"http://{host}:{port}{path}"
    headers = {"Authorization": f"Bearer {token}"} if token else {}

    try:
        resp = httpx.request(method, url, headers=headers, timeout=kwargs.pop("timeout", 10), **kwargs)
    except httpx.ConnectError:
        console.print(f"[red]Could not connect to daemon at {host}:{port}[/red]")
        raise typer.Exit(1)

    if resp.status_code >= 400:
        try:
            msg = resp.json().get("error", resp.text)
        except Exception:
            msg = resp.text
        console.print(f"[red]Error ({resp.status_code}):[/red] {msg}")
        raise typer.Exit(1)

    return resp.json()


@daemon_app.command("sessions")
def list_sessions(
    agent: str = typer.Argument(help="Agent name"),
    host: str = typer.Option("127.0.0.1", "--host", help="Daemon HTTP host"),
    port: int = typer.Option(8321, "--port", "-p", help="Daemon HTTP port"),
    token: Optional[str] = typer.Option(None, "--token", "-t", help="Auth token"),
):
    """List active sessions for a daemon agent."""
    data = _daemon_request("GET", host, port, f"/api/agents/{agent}/sessions", token)

    sessions = data.get("sessions", [])
    if not sessions:
        console.print(f"No sessions for [cyan]{agent}[/cyan]")
        return

    console.print(f"[bold]Sessions for {agent}:[/bold]\n")
    for s in sessions:
        label = s.get("label", s["user_id"])
        conv_id = s.get("conversation_id", "")
        created = s.get("created_at", "")
        console.print(f"  [cyan]{s['user_id']}[/cyan] ({label})")
        console.print(f"    conversation: {conv_id}")
        if created:
            console.print(f"    created: {created}")


@daemon_app.command("compact")
def compact_session(
    agent: str = typer.Argument(help="Agent name"),
    user_id: str = typer.Option("web-anonymous", "--user", "-u", help="User ID to compact"),
    host: str = typer.Option("127.0.0.1", "--host", help="Daemon HTTP host"),
    port: int = typer.Option(8321, "--port", "-p", help="Daemon HTTP port"),
    token: Optional[str] = typer.Option(None, "--token", "-t", help="Auth token"),
):
    """Force compact a daemon session."""
    data = _daemon_request(
        "POST", host, port, f"/api/agents/{agent}/compact", token,
        json={"user_id": user_id}, timeout=120,
    )
    console.print(f"[green]✓[/green] Session compacted for [cyan]{agent}[/cyan] user [cyan]{user_id}[/cyan]")
    console.print(f"  old: {data['old_conversation_id']}")
    console.print(f"  new: {data['new_conversation_id']}")


schedule_app = typer.Typer(help="Manage daemon schedules")
daemon_app.add_typer(schedule_app, name="schedule")


def _parse_every_to_cron(every: str) -> str:
    """Convert --every shorthand (e.g., '2h', '30m', '1d') to cron expression."""
    m = re.match(r"^(\d+)([mhd])$", every.strip())
    if not m:
        raise typer.BadParameter(f"Invalid --every format '{every}'. Use e.g. 30m, 2h, 1d")
    value, unit = int(m.group(1)), m.group(2)
    if unit == "m":
        if value < 1 or value > 59:
            raise typer.BadParameter("Minutes must be 1-59")
        return f"*/{value} * * * *"
    elif unit == "h":
        if value < 1 or value > 23:
            raise typer.BadParameter("Hours must be 1-23")
        return f"0 */{value} * * *"
    else:  # d
        if value != 1:
            raise typer.BadParameter("Only --every 1d is supported (use --cron for other day intervals)")
        return "0 0 * * *"


@schedule_app.command("list")
def schedule_list(
    host: str = typer.Option("127.0.0.1", "--host", help="Daemon HTTP host"),
    port: int = typer.Option(8321, "--port", "-p", help="Daemon HTTP port"),
    token: Optional[str] = typer.Option(None, "--token", "-t", help="Auth token"),
):
    """List all schedules."""
    from rich.table import Table

    data = _daemon_request("GET", host, port, "/api/schedules", token)
    schedules = data.get("schedules", [])
    if not schedules:
        console.print("No schedules configured")
        return

    table = Table(title="Schedules")
    table.add_column("ID", style="cyan")
    table.add_column("Agent")
    table.add_column("Type")
    table.add_column("Schedule")
    table.add_column("Enabled")
    table.add_column("Next Run")
    table.add_column("Last Status")

    for s in schedules:
        sched_str = s.get("cron_expr") or s.get("run_at") or ""
        enabled = "[green]yes[/green]" if s.get("enabled") else "[red]no[/red]"
        status = s.get("last_status") or "-"
        if status == "error":
            status = f"[red]{status}[/red]"
        elif status == "success":
            status = f"[green]{status}[/green]"
        next_run = s.get("next_run") or "-"
        table.add_row(s["id"], s["agent"], s["schedule_type"], sched_str, enabled, next_run, status)

    console.print(table)


@schedule_app.command("add")
def schedule_add(
    schedule_id: str = typer.Argument(help="Unique schedule name"),
    agent: str = typer.Option(..., "--agent", "-a", help="Agent name"),
    prompt: str = typer.Option(..., "--prompt", "-p", help="Prompt to send"),
    cron: Optional[str] = typer.Option(None, "--cron", help="Cron expression (5 fields)"),
    at: Optional[str] = typer.Option(None, "--at", help="ISO datetime for one-off task"),
    every: Optional[str] = typer.Option(None, "--every", help="Simple interval (e.g., 30m, 2h, 1d)"),
    tz: str = typer.Option("UTC", "--timezone", "--tz", help="IANA timezone"),
    host: str = typer.Option("127.0.0.1", "--host", help="Daemon HTTP host"),
    port: int = typer.Option(8321, "--port", help="Daemon HTTP port"),
    token: Optional[str] = typer.Option(None, "--token", "-t", help="Auth token"),
):
    """Add a new schedule."""
    spec_count = sum(1 for x in (cron, at, every) if x is not None)
    if spec_count != 1:
        console.print("[red]Specify exactly one of --cron, --at, or --every[/red]")
        raise typer.Exit(1)

    if every:
        cron = _parse_every_to_cron(every)

    body = {
        "id": schedule_id,
        "agent": agent,
        "prompt": prompt,
        "schedule_type": "once" if at else "cron",
        "timezone": tz,
    }
    if cron:
        body["cron_expr"] = cron
    if at:
        body["run_at"] = at

    data = _daemon_request("POST", host, port, "/api/schedules", token, json=body)
    console.print(f"[green]✓[/green] Schedule [cyan]{schedule_id}[/cyan] created")
    if data.get("next_run"):
        console.print(f"  next run: {data['next_run']}")


@schedule_app.command("remove")
def schedule_remove(
    schedule_id: str = typer.Argument(help="Schedule ID to remove"),
    host: str = typer.Option("127.0.0.1", "--host", help="Daemon HTTP host"),
    port: int = typer.Option(8321, "--port", "-p", help="Daemon HTTP port"),
    token: Optional[str] = typer.Option(None, "--token", "-t", help="Auth token"),
):
    """Remove a schedule."""
    _daemon_request("DELETE", host, port, f"/api/schedules/{schedule_id}", token)
    console.print(f"[green]✓[/green] Schedule [cyan]{schedule_id}[/cyan] removed")


@schedule_app.command("enable")
def schedule_enable(
    schedule_id: str = typer.Argument(help="Schedule ID"),
    host: str = typer.Option("127.0.0.1", "--host", help="Daemon HTTP host"),
    port: int = typer.Option(8321, "--port", "-p", help="Daemon HTTP port"),
    token: Optional[str] = typer.Option(None, "--token", "-t", help="Auth token"),
):
    """Enable a schedule."""
    _daemon_request("POST", host, port, f"/api/schedules/{schedule_id}/enable", token)
    console.print(f"[green]✓[/green] Schedule [cyan]{schedule_id}[/cyan] enabled")


@schedule_app.command("disable")
def schedule_disable(
    schedule_id: str = typer.Argument(help="Schedule ID"),
    host: str = typer.Option("127.0.0.1", "--host", help="Daemon HTTP host"),
    port: int = typer.Option(8321, "--port", "-p", help="Daemon HTTP port"),
    token: Optional[str] = typer.Option(None, "--token", "-t", help="Auth token"),
):
    """Disable a schedule."""
    _daemon_request("POST", host, port, f"/api/schedules/{schedule_id}/disable", token)
    console.print(f"[green]✓[/green] Schedule [cyan]{schedule_id}[/cyan] disabled")


@schedule_app.command("run")
def schedule_run(
    schedule_id: str = typer.Argument(help="Schedule ID to trigger now"),
    host: str = typer.Option("127.0.0.1", "--host", help="Daemon HTTP host"),
    port: int = typer.Option(8321, "--port", "-p", help="Daemon HTTP port"),
    token: Optional[str] = typer.Option(None, "--token", "-t", help="Auth token"),
):
    """Trigger a schedule immediately."""
    _daemon_request("POST", host, port, f"/api/schedules/{schedule_id}/run", token)
    console.print(f"[green]✓[/green] Schedule [cyan]{schedule_id}[/cyan] triggered")


@schedule_app.command("show")
def schedule_show(
    schedule_id: str = typer.Argument(help="Schedule ID"),
    host: str = typer.Option("127.0.0.1", "--host", help="Daemon HTTP host"),
    port: int = typer.Option(8321, "--port", "-p", help="Daemon HTTP port"),
    token: Optional[str] = typer.Option(None, "--token", "-t", help="Auth token"),
):
    """Show schedule details."""
    data = _daemon_request("GET", host, port, f"/api/schedules/{schedule_id}", token)

    console.print(f"[bold]{data['id']}[/bold]")
    console.print(f"  agent:    {data['agent']}")
    console.print(f"  prompt:   {data['prompt']}")
    console.print(f"  type:     {data['schedule_type']}")
    if data.get("cron_expr"):
        console.print(f"  cron:     {data['cron_expr']}")
    if data.get("run_at"):
        console.print(f"  run at:   {data['run_at']}")
    console.print(f"  timezone: {data['timezone']}")
    console.print(f"  enabled:  {'[green]yes[/green]' if data['enabled'] else '[red]no[/red]'}")
    console.print(f"  next run: {data.get('next_run') or '-'}")
    console.print(f"  last run: {data.get('last_run') or '-'}")
    if data.get("last_status"):
        console.print(f"  status:   {data['last_status']}")
    if data.get("last_error"):
        console.print(f"  error:    [red]{data['last_error']}[/red]")


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
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        console.print("\n[yellow]Daemon stopped[/yellow]")


__all__ = ["daemon_app"]
