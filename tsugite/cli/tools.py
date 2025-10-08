"""Tools management CLI commands."""

from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

console = Console()

tools_app = typer.Typer(help="Manage and inspect available tools")


@tools_app.command("list")
def tools_list(
    category: Optional[str] = typer.Option(None, "--category", help="Filter by category (fs, shell, http, etc.)"),
):
    """List all available tools."""
    from tsugite.tools import _tools

    if not _tools:
        console.print("[yellow]No tools registered[/yellow]")
        return

    # Group tools by module
    categorized = {}
    for tool_name, tool_info in _tools.items():
        module = tool_info.func.__module__.split(".")[-1]  # Get last part (e.g., 'fs' from 'tsugite.tools.fs')
        if module not in categorized:
            categorized[module] = []
        categorized[module].append((tool_name, tool_info.description))

    # Filter by category if specified
    if category:
        if category not in categorized:
            console.print(f"[red]Category '{category}' not found[/red]")
            console.print(f"\nAvailable categories: {', '.join(sorted(categorized.keys()))}")
            raise typer.Exit(1)
        categorized = {category: categorized[category]}

    # Display tools by category
    total_count = 0
    for cat in sorted(categorized.keys()):
        tools = sorted(categorized[cat])
        total_count += len(tools)

        console.print(f"\n[cyan]{cat.upper()}[/cyan] ({len(tools)} tool{'s' if len(tools) != 1 else ''})")
        console.print("[dim]" + "─" * 60 + "[/dim]")

        for tool_name, description in tools:
            console.print(f"  [bold]{tool_name}[/bold]")
            console.print(f"    {description}")

    console.print(f"\n[dim]Total: {total_count} tool{'s' if total_count != 1 else ''}[/dim]")


@tools_app.command("show")
def tools_show(
    tool_name: str = typer.Argument(help="Tool name to inspect"),
):
    """Show detailed information about a specific tool."""
    from tsugite.tools import _tools

    if tool_name not in _tools:
        console.print(f"[red]Tool '{tool_name}' not found[/red]")
        console.print("\nUse [cyan]tsugite tools list[/cyan] to see all available tools")
        raise typer.Exit(1)

    tool_info = _tools[tool_name]

    # Header
    console.print(f"\n[bold cyan]{tool_info.name}[/bold cyan]")
    console.print(f"[dim]{tool_info.description}[/dim]\n")

    # Parameters table
    if tool_info.parameters:
        table = Table(show_header=True, header_style="bold cyan", border_style="dim")
        table.add_column("Parameter", style="bold")
        table.add_column("Type")
        table.add_column("Required")
        table.add_column("Default")

        for param_name, param_info in tool_info.parameters.items():
            param_type = (
                param_info["type"].__name__ if hasattr(param_info["type"], "__name__") else str(param_info["type"])
            )
            required = "✓" if param_info["required"] else ""
            default = "" if param_info["default"] is None else str(param_info["default"])

            table.add_row(param_name, param_type, required, default)

        console.print(table)
    else:
        console.print("[dim]No parameters[/dim]")

    # Module info
    module = tool_info.func.__module__
    console.print(f"\n[dim]Module: {module}[/dim]")
