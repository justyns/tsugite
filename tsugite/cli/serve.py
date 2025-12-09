"""CLI commands for running Tsugite servers."""

import json
import sys

import typer
from rich.console import Console

from tsugite.mcp_server import EXPOSED_TOOLS

serve_app = typer.Typer(help="Run Tsugite servers")
console = Console()


def show_mcp_info():
    """Display MCP server configuration info."""
    console.print("\n[bold]Tsugite MCP Server[/bold]")
    console.print("=" * 40)

    console.print("\n[bold]For Claude Code[/bold] (~/.claude/mcp_servers.json):\n")
    stdio_config = {"tsugite": {"command": "tsu", "args": ["serve", "mcp"]}}
    console.print(f"  {json.dumps(stdio_config, indent=2).replace(chr(10), chr(10) + '  ')}")

    console.print(f"\n[bold]Available tools ({len(EXPOSED_TOOLS)}):[/bold]")
    console.print(f"  {', '.join(EXPOSED_TOOLS)}")
    console.print()


@serve_app.command("mcp")
def serve_mcp(
    info: bool = typer.Option(False, "--info", help="Show client configuration"),
):
    """Run MCP server over stdio (for Claude Code integration)."""
    if info:
        show_mcp_info()
        return

    from tsugite.mcp_server import run_stdio

    print("Tsugite MCP server (stdio)", file=sys.stderr)
    run_stdio()
