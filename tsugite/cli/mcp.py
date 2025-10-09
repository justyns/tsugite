"""MCP server management CLI commands."""

from typing import Optional

import typer
from rich.console import Console

console = Console()

mcp_app = typer.Typer(help="Manage MCP server configurations")


@mcp_app.command("list")
def mcp_list():
    """List all configured MCP servers."""
    from tsugite.mcp_config import get_default_config_path, load_mcp_config

    config_path = get_default_config_path()
    servers = load_mcp_config()

    if not servers:
        console.print("[yellow]No MCP servers configured[/yellow]")
        console.print(f"\nConfig file will be created at: {config_path}")
        return

    console.print(f"[cyan]Found {len(servers)} MCP server(s)[/cyan] in [dim]{config_path}[/dim]\n")

    for name, config in servers.items():
        server_type = "stdio" if config.is_stdio() else "HTTP"
        console.print(f"[bold]{name}[/bold] ({server_type})")

        if config.is_stdio():
            console.print(f"  Command: {config.command}")
            if config.args:
                console.print(f"  Args: {' '.join(config.args)}")
            if config.env:
                env_keys = list(config.env.keys())
                console.print(f"  Env vars: {', '.join(env_keys)}")
        else:
            console.print(f"  URL: {config.url}")

        console.print()


@mcp_app.command("show")
def mcp_show(server_name: str = typer.Argument(help="Name of the MCP server to show")):
    """Show detailed configuration for a specific MCP server."""
    from rich.panel import Panel

    from tsugite.mcp_config import load_mcp_config

    servers = load_mcp_config()

    if server_name not in servers:
        console.print(f"[red]MCP server '{server_name}' not found[/red]")
        console.print(f"\nAvailable servers: {', '.join(servers.keys())}")
        raise typer.Exit(1)

    config = servers[server_name]
    server_type = "stdio" if config.is_stdio() else "HTTP"

    console.print(Panel(f"[cyan]Server:[/cyan] {server_name}\n[cyan]Type:[/cyan] {server_type}", border_style="blue"))

    if config.is_stdio():
        console.print(f"\n[bold]Command:[/bold] {config.command}")
        if config.args:
            console.print("[bold]Arguments:[/bold]")
            for arg in config.args:
                console.print(f"  - {arg}")
        if config.env:
            console.print("\n[bold]Environment Variables:[/bold]")
            for key, value in config.env.items():
                if "token" in key.lower() or "key" in key.lower() or "secret" in key.lower():
                    console.print(f"  {key}: [dim]<redacted>[/dim]")
                else:
                    console.print(f"  {key}: {value}")
    else:
        console.print(f"\n[bold]URL:[/bold] {config.url}")


@mcp_app.command("test")
def mcp_test(
    server_name: str = typer.Argument(help="Name of the MCP server to test"),
):
    """Test connection to an MCP server and list available tools."""
    import asyncio

    from tsugite.mcp_client import load_mcp_tools
    from tsugite.mcp_config import load_mcp_config

    servers = load_mcp_config()

    if server_name not in servers:
        console.print(f"[red]MCP server '{server_name}' not found[/red]")
        console.print(f"\nAvailable servers: {', '.join(servers.keys())}")
        raise typer.Exit(1)

    config = servers[server_name]

    console.print(f"[cyan]Testing connection to '{server_name}'...[/cyan]")

    try:
        # Use new async mcp_client implementation
        tools = asyncio.run(load_mcp_tools(config, allowed_tools=None))

        console.print(f"[green]✓ Successfully connected to '{server_name}'[/green]")
        console.print(f"\n[bold]Available tools ({len(tools)}):[/bold]")

        for tool in tools:
            console.print(f"  - {tool.name}: {tool.description}")

    except Exception as e:
        console.print(f"[red]✗ Connection failed: {e}[/red]")
        raise typer.Exit(1)


@mcp_app.command("add")
def mcp_add(
    name: str = typer.Argument(help="Name for the MCP server"),
    url: Optional[str] = typer.Option(None, "--url", help="URL for HTTP server"),
    command: Optional[str] = typer.Option(None, "--command", help="Command for stdio server"),
    args: Optional[list[str]] = typer.Option(None, "--args", help="Argument for stdio server (repeatable)"),
    env: Optional[list[str]] = typer.Option(None, "--env", help="Environment variable as KEY=value (repeatable)"),
    server_type: Optional[str] = typer.Option(None, "--type", help="Server type: stdio or http"),
    force: bool = typer.Option(False, "--force", help="Overwrite if server already exists"),
):
    """Add a new MCP server to the configuration."""
    from tsugite.mcp_config import MCPServerConfig, add_server_to_config

    # Validate that either url or command is provided
    if not url and not command:
        console.print("[red]Error: Must specify either --url (for HTTP) or --command (for stdio)[/red]")
        raise typer.Exit(1)

    if url and command:
        console.print("[red]Error: Cannot specify both --url and --command[/red]")
        raise typer.Exit(1)

    # Parse environment variables
    env_dict = None
    if env:
        env_dict = {}
        for env_var in env:
            if "=" not in env_var:
                console.print(f"[red]Error: Invalid env var format '{env_var}'. Must be KEY=value[/red]")
                raise typer.Exit(1)
            key, value = env_var.split("=", 1)
            env_dict[key] = value

    # Create server config
    try:
        server = MCPServerConfig(
            name=name,
            url=url,
            command=command,
            args=args if args else None,
            env=env_dict,
            type=server_type,
        )
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)

    # Add to config
    try:
        from tsugite.mcp_config import get_config_path_for_write

        config_path = get_config_path_for_write()
        add_server_to_config(server, overwrite=force)

        action = "Updated" if force else "Added"
        server_type_name = "HTTP" if server.is_http() else "stdio"

        console.print(f"[green]✓ {action} MCP server '{name}' ({server_type_name})[/green]")

        # Show config summary
        if server.is_http():
            console.print(f"  URL: {server.url}")
        else:
            console.print(f"  Command: {server.command}")
            if server.args:
                console.print(f"  Args: {' '.join(server.args)}")
            if server.env:
                console.print(f"  Env vars: {', '.join(server.env.keys())}")

        console.print(f"\nServer saved to: {config_path}")

    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Failed to add server: {e}[/red]")
        raise typer.Exit(1)
