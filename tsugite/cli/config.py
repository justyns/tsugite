"""Configuration management CLI commands."""

from typing import Optional

import typer
from rich.console import Console

console = Console()

config_app = typer.Typer(help="Manage Tsugite configuration")


@config_app.command("show")
def config_show():
    """Show current configuration."""
    from tsugite.config import get_config_path, load_config

    config_path = get_config_path()
    config = load_config()

    console.print(f"[cyan]Configuration file:[/cyan] [dim]{config_path}[/dim]\n")

    if config.default_model:
        console.print(f"[bold]Default Model:[/bold] {config.default_model}\n")
    else:
        console.print("[yellow]No default model set[/yellow]\n")

    if config.default_base_agent is not None:
        console.print(f"[bold]Default Base Agent:[/bold] {config.default_base_agent}\n")
    else:
        console.print("[bold]Default Base Agent:[/bold] default (fallback)\n")

    if config.default_workspace:
        console.print(f"[bold]Default Workspace:[/bold] {config.default_workspace}\n")
    else:
        console.print("[yellow]No default workspace set[/yellow]\n")

    console.print(f"[bold]Chat Theme:[/bold] {config.chat_theme}\n")

    # Auto-context settings
    status = "[green]enabled[/green]" if config.auto_context_enabled else "[red]disabled[/red]"
    console.print(f"[bold]Auto-Context:[/bold] {status}")
    if config.auto_context_files:
        console.print(f"  Files: {', '.join(config.auto_context_files)}")
    console.print(f"  Include Global: {config.auto_context_include_global}\n")

    if config.model_aliases:
        console.print(f"[bold]Model Aliases ({len(config.model_aliases)}):[/bold]")
        for alias, model in config.model_aliases.items():
            console.print(f"  {alias} → {model}")
    else:
        console.print("[dim]No model aliases defined[/dim]")


@config_app.command("set-default")
def config_set_default(
    model: Optional[str] = typer.Argument(None, help="Model string (optional, interactive if omitted)"),
):
    """Set the default model (interactive or direct).

    Examples:
        # Interactive mode with provider and model selection
        tsugite config set-default

        # Direct mode with model string
        tsugite config set-default "ollama:qwen2.5-coder:7b"
    """
    from tsugite.config import get_config_path, update_config

    # If no model provided, launch interactive selector
    if not model:
        from tsugite.cli.init import detect_available_providers, prompt_for_model

        console.print("\n[bold cyan]Select Default Model[/bold cyan]")
        console.print("[dim]Detecting available providers...[/dim]\n")

        providers = detect_available_providers()
        model = prompt_for_model(providers)

        if not model:
            console.print("[yellow]No model selected[/yellow]")
            raise typer.Exit(1)

    try:
        update_config(None, lambda cfg: setattr(cfg, "default_model", model))
        config_path = get_config_path()
        console.print(f"\n[green]✓ Default model set to:[/green] {model}")
        console.print(f"[dim]Saved to: {config_path}[/dim]")
    except Exception as e:
        console.print(f"[red]Failed to set default model: {e}[/red]")
        raise typer.Exit(1)


@config_app.command("alias")
def config_alias(
    name: str = typer.Argument(help="Alias name (e.g., 'cheap')"),
    model: str = typer.Argument(help="Model string (e.g., 'openai:gpt-4o-mini')"),
):
    """Create or update a model alias."""
    from tsugite.config import get_config_path, update_config

    try:
        update_config(None, lambda cfg: cfg.model_aliases.update({name: model}))
        config_path = get_config_path()
        console.print(f"[green]✓ Alias created:[/green] {name} → {model}")
        console.print(f"[dim]Saved to: {config_path}[/dim]")
    except Exception as e:
        console.print(f"[red]Failed to create alias: {e}[/red]")
        raise typer.Exit(1)


@config_app.command("alias-remove")
def config_alias_remove(
    name: str = typer.Argument(help="Alias name to remove"),
):
    """Remove a model alias."""
    from tsugite.config import get_config_path, remove_model_alias

    try:
        config_path = get_config_path()
        if remove_model_alias(name):
            console.print(f"[green]✓ Alias removed:[/green] {name}")
            console.print(f"[dim]Saved to: {config_path}[/dim]")
        else:
            console.print(f"[yellow]Alias '{name}' not found[/yellow]")
    except Exception as e:
        console.print(f"[red]Failed to remove alias: {e}[/red]")
        raise typer.Exit(1)


@config_app.command("list-aliases")
def config_list_aliases():
    """List all model aliases."""
    from tsugite.config import load_config

    config = load_config()

    if not config.model_aliases:
        console.print("[yellow]No model aliases defined[/yellow]")
        console.print("\nCreate an alias with: [cyan]tsugite config alias NAME MODEL[/cyan]")
        return

    console.print(f"[cyan]Model Aliases ({len(config.model_aliases)}):[/cyan]\n")
    for alias, model in config.model_aliases.items():
        console.print(f"  [bold]{alias}[/bold] → {model}")


@config_app.command("set-default-base")
def config_set_default_base(
    base_agent: str = typer.Argument(help="Base agent name (e.g., 'default') or 'none' to disable"),
):
    """Set the default base agent for inheritance."""
    from tsugite.config import get_config_path, update_config

    try:
        # Handle "none" as None
        agent_value = None if base_agent.lower() == "none" else base_agent

        update_config(None, lambda cfg: setattr(cfg, "default_base_agent", agent_value))
        config_path = get_config_path()

        if agent_value is None:
            console.print("[green]✓ Default base agent disabled[/green]")
        else:
            console.print(f"[green]✓ Default base agent set to:[/green] {base_agent}")

        console.print(f"[dim]Saved to: {config_path}[/dim]")
    except Exception as e:
        console.print(f"[red]Failed to set default base agent: {e}[/red]")
        raise typer.Exit(1)


@config_app.command("set-default-workspace")
def config_set_default_workspace(
    workspace: str = typer.Argument(help="Workspace name or 'none' to disable"),
):
    """Set the default workspace (auto-loaded unless --no-workspace is used).

    Examples:
        tsu config set-default-workspace justyn
        tsu config set-default-workspace none  # Disable default workspace
    """
    from tsugite.config import get_config_path, update_config

    try:
        workspace_value = None if workspace.lower() == "none" else workspace

        update_config(None, lambda cfg: setattr(cfg, "default_workspace", workspace_value))
        config_path = get_config_path()

        if workspace_value is None:
            console.print("[green]✓ Default workspace disabled[/green]")
        else:
            console.print(f"[green]✓ Default workspace set to:[/green] {workspace}")
            console.print("\n[dim]The workspace will be auto-loaded on every run.[/dim]")
            console.print("[dim]Use --no-workspace to disable for a specific run.[/dim]")

        console.print(f"\n[dim]Saved to: {config_path}[/dim]")
    except Exception as e:
        console.print(f"[red]Failed to set default workspace: {e}[/red]")
        raise typer.Exit(1)


def _get_available_themes() -> list[str]:
    """Get list of available themes from Textual."""
    from textual.app import App

    app = App()
    return sorted(app.available_themes.keys())


@config_app.command("set-theme")
def config_set_theme(
    theme: str = typer.Argument(help="Theme name (e.g., 'gruvbox', 'nord', 'tokyo-night')"),
):
    """Set the chat UI theme."""
    from tsugite.config import get_config_path, update_config

    # Get available themes from Textual
    available_themes = _get_available_themes()

    # Validate theme name
    if theme not in available_themes:
        console.print(f"[red]Unknown theme: {theme}[/red]")
        console.print("\nAvailable themes:")
        for t in available_themes:
            console.print(f"  • {t}")
        console.print("\nUse [cyan]tsugite config list-themes[/cyan] to see all themes")
        raise typer.Exit(1)

    try:
        update_config(None, lambda cfg: setattr(cfg, "chat_theme", theme))
        config_path = get_config_path()
        console.print(f"[green]✓ Chat theme set to:[/green] {theme}")
        console.print(f"[dim]Saved to: {config_path}[/dim]")
    except Exception as e:
        console.print(f"[red]Failed to set theme: {e}[/red]")
        raise typer.Exit(1)


@config_app.command("list-themes")
def config_list_themes():
    """List all available chat UI themes."""
    from tsugite.config import load_config

    config = load_config()
    current_theme = config.chat_theme

    # Get available themes from Textual
    available_themes = _get_available_themes()

    console.print("[cyan]Available Textual Themes:[/cyan]\n")

    for theme in available_themes:
        if theme == current_theme:
            console.print(f"  [bold green]• {theme}[/bold green] [dim](current)[/dim]")
        else:
            console.print(f"  • {theme}")

    console.print(f"\n[dim]Current theme: {current_theme}[/dim]")
    console.print("[dim]Change with: tsugite config set-theme <theme>[/dim]")


@config_app.command("set")
def config_set(
    key: str = typer.Argument(help="Configuration key (e.g., 'auto-context-enabled')"),
    value: str = typer.Argument(help="Configuration value"),
):
    """Set a configuration value.

    Supported keys:
        auto-context-enabled: Enable/disable auto-context (true/false)

    Examples:
        tsugite config set auto-context-enabled true
        tsugite config set auto-context-enabled false
    """
    from tsugite.config import get_config_path, update_config

    # Normalize key (convert dashes to underscores)
    normalized_key = key.replace("-", "_")

    if normalized_key == "auto_context_enabled":
        # Parse boolean value
        value_lower = value.lower()
        if value_lower in ("true", "yes", "1", "on"):
            bool_value = True
        elif value_lower in ("false", "no", "0", "off"):
            bool_value = False
        else:
            console.print(f"[red]Invalid boolean value: {value}[/red]")
            console.print("[yellow]Use: true, false, yes, no, 1, 0, on, or off[/yellow]")
            raise typer.Exit(1)

        try:
            update_config(None, lambda cfg: setattr(cfg, "auto_context_enabled", bool_value))
            config_path = get_config_path()
            status = "enabled" if bool_value else "disabled"
            console.print(f"[green]✓ Auto-context {status}[/green]")
            console.print(f"[dim]Saved to: {config_path}[/dim]")
        except Exception as e:
            console.print(f"[red]Failed to set auto-context-enabled: {e}[/red]")
            raise typer.Exit(1)
    else:
        console.print(f"[red]Unknown configuration key: {key}[/red]")
        console.print("\n[yellow]Supported keys:[/yellow]")
        console.print("  • auto-context-enabled")
        raise typer.Exit(1)
