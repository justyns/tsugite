"""Plugin management CLI commands."""

from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

console = Console()

plugin_app = typer.Typer(help="Manage and inspect installed plugins")


@plugin_app.command("list")
def plugin_list(
    group: Optional[str] = typer.Option(None, "--group", "-g", help="Filter by group (tools, adapters)"),
):
    """List all discovered plugins."""
    from tsugite.plugins import discover_plugins

    plugins = discover_plugins()

    if group:
        full_group = f"tsugite.{group}" if not group.startswith("tsugite.") else group
        plugins = [p for p in plugins if p.group == full_group]

    if not plugins:
        console.print("[dim]No plugins discovered.[/dim]")
        console.print("\nInstall a plugin package (e.g. [cyan]pip install tsugite-weather[/cyan]) to get started.")
        return

    table = Table(show_header=True, header_style="bold cyan", border_style="dim")
    table.add_column("Name", style="bold")
    table.add_column("Type")
    table.add_column("Entry Point", style="dim")
    table.add_column("Status")

    for p in sorted(plugins, key=lambda x: (x.group, x.name)):
        plugin_type = p.group.removeprefix("tsugite.")
        if not p.enabled:
            status = "[yellow]disabled[/yellow]"
        elif p.error:
            status = f"[red]error: {p.error}[/red]"
        elif p.loaded:
            status = "[green]loaded[/green]"
        else:
            status = "[dim]discovered[/dim]"
        table.add_row(p.name, plugin_type, p.entry_point, status)

    console.print(table)
    console.print(f"\n[dim]{len(plugins)} plugin(s)[/dim]")


@plugin_app.command("info")
def plugin_info(
    name: str = typer.Argument(help="Plugin name to inspect"),
):
    """Show detailed information about a plugin."""
    from tsugite.plugins import discover_plugins

    plugins = discover_plugins()
    matches = [p for p in plugins if p.name == name]

    if not matches:
        console.print(f"[red]Plugin '{name}' not found[/red]")
        available = [p.name for p in plugins]
        if available:
            console.print(f"\nAvailable: {', '.join(sorted(available))}")
        raise typer.Exit(1)

    for p in matches:
        console.print(f"\n[bold cyan]{p.name}[/bold cyan]")
        console.print(f"  Group:       {p.group}")
        console.print(f"  Entry point: {p.entry_point}")
        console.print(f"  Enabled:     {'yes' if p.enabled else 'no'}")
        if p.error:
            console.print(f"  Error:       [red]{p.error}[/red]")
