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


@tools_app.command("add")
def tools_add(
    name: str = typer.Argument(help="Tool name"),
    command: str = typer.Option(..., "--command", "-c", help="Shell command template (e.g., 'rg {pattern} {path}')"),
    description: str = typer.Option("", "--description", "-d", help="Tool description"),
    param: Optional[list[str]] = typer.Option(
        None,
        "--param",
        "-p",
        help="Parameter: name[:type][:required|default]. Type defaults to 'str' if omitted.",
    ),
    timeout: int = typer.Option(30, "--timeout", "-t", help="Command timeout in seconds"),
):
    """Add a custom shell tool.

    Parameter formats (type defaults to str):
        -p pattern              # str, optional
        -p pattern:required     # str, required
        -p path:.               # str with default value "."
        -p count:int            # int, optional
        -p count:int:required   # int, required
        -p count:int:10         # int with default value 10

    Examples:
        tsugite tools add file_search -c "rg {pattern} {path}" -p pattern:required -p path:.
        tsugite tools add git_search -c "git log --all --grep={query}" -p query:required
        tsugite tools add count_lines -c "wc -l {file}" -p file:required
    """
    from tsugite.shell_tool_config import load_custom_tools_config, save_custom_tools_config
    from tsugite.tools.shell_tools import ShellToolDefinition, ShellToolParameter

    # Parse parameters
    parameters = {}
    if param:
        for param_spec in param:
            parts = param_spec.split(":")
            if len(parts) < 1:
                console.print(f"[red]Invalid parameter spec: {param_spec}[/red]")
                console.print("Format: name[:type][:required|default]")
                raise typer.Exit(1)

            param_name = parts[0]

            # Default to string type
            param_type = "str"
            required = False
            default = None

            if len(parts) >= 2:
                # Second part is either type or default/required
                second = parts[1]
                if second in ("str", "int", "bool", "float"):
                    # It's a type
                    param_type = second

                    # Check for third part (required or default)
                    if len(parts) >= 3:
                        if parts[2] == "required":
                            required = True
                        else:
                            default = parts[2]
                elif second == "required":
                    # No type specified, but marked as required
                    required = True
                else:
                    # It's a default value (type defaults to str)
                    default = second

            parameters[param_name] = ShellToolParameter(
                name=param_name,
                type=param_type,
                description="",
                required=required,
                default=default,
            )

    # Create tool definition
    tool_def = ShellToolDefinition(
        name=name,
        description=description or f"Custom shell tool: {name}",
        command=command,
        parameters=parameters,
        timeout=timeout,
    )

    # Load existing tools
    try:
        existing_tools = load_custom_tools_config()

        # Check for duplicates
        if any(t.name == name for t in existing_tools):
            console.print(f"[red]Tool '{name}' already exists[/red]")
            console.print("Use [cyan]tsugite tools remove[/cyan] first or choose a different name")
            raise typer.Exit(1)

        # Add new tool
        existing_tools.append(tool_def)

        # Save
        save_custom_tools_config(existing_tools)

        console.print(f"[green]✓[/green] Added custom tool: [bold]{name}[/bold]")
        console.print(f"\nCommand: {command}")
        if parameters:
            console.print("\nParameters:")
            for param_name, param_def in parameters.items():
                req_str = " (required)" if param_def.required else f" (default: {param_def.default})"
                console.print(f"  • {param_name}: {param_def.type}{req_str}")

        console.print(f'\n[dim]Try it: tsugite run +assistant "use {name} to..."[/dim]')

    except Exception as e:
        console.print(f"[red]Failed to add tool: {e}[/red]")
        raise typer.Exit(1)


@tools_app.command("remove")
def tools_remove(
    name: str = typer.Argument(help="Tool name to remove"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation"),
):
    """Remove a custom shell tool."""
    from tsugite.shell_tool_config import load_custom_tools_config, save_custom_tools_config

    try:
        existing_tools = load_custom_tools_config()

        # Find tool
        tool_def = next((t for t in existing_tools if t.name == name), None)
        if not tool_def:
            console.print(f"[red]Custom tool '{name}' not found[/red]")
            console.print("\nUse [cyan]tsugite tools list --custom[/cyan] to see custom tools")
            raise typer.Exit(1)

        # Confirm
        if not yes:
            console.print(f"\nRemove custom tool [bold]{name}[/bold]?")
            console.print(f"Command: {tool_def.command}")
            confirm = typer.confirm("Continue?")
            if not confirm:
                console.print("Cancelled")
                return

        # Remove
        existing_tools = [t for t in existing_tools if t.name != name]
        save_custom_tools_config(existing_tools)

        console.print(f"[green]✓[/green] Removed custom tool: [bold]{name}[/bold]")

    except Exception as e:
        console.print(f"[red]Failed to remove tool: {e}[/red]")
        raise typer.Exit(1)


@tools_app.command("edit")
def tools_edit(
    name: str = typer.Argument(help="Tool name to edit"),
):
    """Edit a custom shell tool in your editor."""
    import os
    import subprocess

    from tsugite.shell_tool_config import get_custom_tools_config_path

    try:
        config_path = get_custom_tools_config_path()

        if not config_path.exists():
            console.print("[yellow]No custom tools config found[/yellow]")
            console.print("Add a tool first with [cyan]tsugite tools add[/cyan]")
            return

        # Open in editor
        editor = os.environ.get("EDITOR", "nano")
        subprocess.run([editor, str(config_path)], check=False)

        console.print("[green]✓[/green] Config updated")
        console.print("\n[dim]Restart tsugite to load changes[/dim]")

    except Exception as e:
        console.print(f"[red]Failed to edit tools: {e}[/red]")
        raise typer.Exit(1)


@tools_app.command("validate")
def tools_validate():
    """Validate custom_tools.yaml configuration."""
    from tsugite.shell_tool_config import get_custom_tools_config_path, load_custom_tools_config

    try:
        config_path = get_custom_tools_config_path()

        if not config_path.exists():
            console.print("[yellow]No custom tools configuration found[/yellow]")
            console.print(f"\nExpected location: {config_path}")
            console.print("\nUse [cyan]tsugite tools add[/cyan] to create custom tools")
            return

        console.print(f"Validating: {config_path}\n")

        # Try to load the configuration
        definitions = load_custom_tools_config()

        if not definitions:
            console.print("[yellow]⚠[/yellow] Configuration is valid but no tools are defined")
            return

        # Validate each tool definition
        console.print(f"[green]✓[/green] Found {len(definitions)} tool definition(s)\n")

        table = Table(show_header=True, header_style="bold cyan", border_style="dim")
        table.add_column("Tool Name", style="bold")
        table.add_column("Command")
        table.add_column("Parameters")
        table.add_column("Status")

        for tool_def in definitions:
            param_count = len(tool_def.parameters)
            param_str = f"{param_count} parameter(s)"

            # Check for potential issues
            issues = []
            if not tool_def.description:
                issues.append("no description")
            if not tool_def.parameters:
                issues.append("no parameters")

            status = "[green]✓[/green]" if not issues else f"[yellow]⚠ {', '.join(issues)}[/yellow]"

            cmd_display = tool_def.command[:40] + "..." if len(tool_def.command) > 40 else tool_def.command
            table.add_row(tool_def.name, cmd_display, param_str, status)

        console.print(table)
        console.print("\n[green]✓[/green] Configuration is valid")

    except Exception as e:
        console.print(f"[red]✗ Validation failed:[/red] {e}")
        console.print(f"\nConfig file: {config_path}")
        raise typer.Exit(1)


@tools_app.command("check")
def tools_check(
    agent_path: str = typer.Argument(help="Path to agent markdown file"),
):
    """Check if an agent's tools are available."""
    from pathlib import Path

    from tsugite.md_agents import parse_agent_file
    from tsugite.tools import _tools

    try:
        agent_file = Path(agent_path)

        if not agent_file.exists():
            console.print(f"[red]Agent file not found:[/red] {agent_path}")
            raise typer.Exit(1)

        # Parse agent
        agent = parse_agent_file(agent_file)

        console.print(f"Checking tools for: [bold]{agent.config.name}[/bold]\n")

        if not agent.config.tools:
            console.print("[yellow]No tools configured in agent[/yellow]")
            return

        # Expand tool specifications
        from tsugite.tools import expand_tool_specs

        try:
            expanded_tools = expand_tool_specs(agent.config.tools)
        except Exception as e:
            console.print(f"[red]Error expanding tool specifications:[/red] {e}")
            raise typer.Exit(1)

        # Check each tool
        available = []
        missing = []

        for tool_name in expanded_tools:
            if tool_name in _tools:
                available.append(tool_name)
            else:
                missing.append(tool_name)

        # Display results
        if available:
            console.print(f"[green]✓ Available ({len(available)}):[/green]")
            for tool in available:
                tool_info = _tools[tool]
                console.print(f"  • {tool} - {tool_info.description}")

        if missing:
            console.print(f"\n[red]✗ Missing ({len(missing)}):[/red]")
            for tool in missing:
                console.print(f"  • {tool}")

            console.print("\n[yellow]Suggestions:[/yellow]")
            from tsugite.shell_tool_config import get_custom_tools_config_path

            config_path = get_custom_tools_config_path()
            if config_path.exists():
                console.print(f"  • Check custom tools in: {config_path}")
            else:
                console.print(f"  • Create custom tools at: {config_path}")

            console.print("  • Run [cyan]tsugite tools list[/cyan] to see all tools")
            console.print("  • Run [cyan]tsugite tools add <name> ...[/cyan] to add missing tools")

            raise typer.Exit(1)
        else:
            console.print("\n[green]✓ All tools are available[/green]")

    except Exception as e:
        if not isinstance(e, typer.Exit):
            console.print(f"[red]Error:[/red] {e}")
            raise typer.Exit(1)
        raise
