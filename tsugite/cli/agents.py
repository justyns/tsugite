"""Agent management CLI commands."""

from pathlib import Path

import typer
from rich.console import Console

console = Console()

agents_app = typer.Typer(help="Manage agents and agent inheritance")


@agents_app.command("list")
def agents_list(
    global_only: bool = typer.Option(False, "--global", help="List only global agents"),
    local_only: bool = typer.Option(False, "--local", help="List only local agents"),
):
    """List available agents."""
    from tsugite.agent_inheritance import get_global_agents_paths
    from tsugite.agent_utils import list_local_agents

    # Show global agents
    if global_only or not local_only:
        console.print("[cyan]Global Agents:[/cyan]\n")

        found_any = False
        for global_path in get_global_agents_paths():
            if not global_path.exists():
                continue

            agent_files = sorted(global_path.glob("*.md"))
            if agent_files:
                found_any = True
                console.print(f"[dim]{global_path}[/dim]")
                for agent_file in agent_files:
                    console.print(f"  • {agent_file.stem}")
                console.print()

        if not found_any:
            console.print("[yellow]No global agents found[/yellow]")
            console.print("\nGlobal agent locations:")
            for path in get_global_agents_paths():
                console.print(f"  {path}")
            console.print()

    # Show local agents
    if local_only or not global_only:
        console.print("[cyan]Local Agents:[/cyan]\n")

        local_agents = list_local_agents()

        if not local_agents:
            console.print("[yellow]No local agents found[/yellow]")
            console.print("\nLocal agent locations:")
            console.print("  • Current directory (*.md)")
            console.print("  • .tsugite/")
            console.print("  • ./agents/")
        else:
            for location, agent_files in local_agents.items():
                console.print(f"[dim]{location}[/dim]")
                for agent_file in agent_files:
                    console.print(f"  • {agent_file.stem}")
                console.print()


@agents_app.command("show")
def agents_show(
    agent_path: str = typer.Argument(help="Agent name or path to agent file"),
    show_inheritance: bool = typer.Option(False, "--inheritance", help="Show inheritance chain"),
):
    """Show agent information.

    Can be either a file path (e.g., 'examples/my_agent.md') or
    an agent name to search globally (e.g., 'default', 'builtin-default').
    """
    from tsugite.agent_inheritance import find_agent_file
    from tsugite.builtin_agents import is_builtin_agent_path
    from tsugite.md_agents import parse_agent_file

    try:
        agent_file = Path(agent_path)

        # If path doesn't exist, try to find it as an agent name
        if not agent_file.exists():
            found_path = find_agent_file(agent_path, Path.cwd())
            if found_path:
                agent_file = found_path

                # Show if it's a package-provided agent
                if is_builtin_agent_path(agent_file):
                    console.print(f"[dim]Package agent: {agent_file.stem}[/dim]\n")
                else:
                    console.print(f"[dim]Found: {agent_file}[/dim]\n")
            else:
                console.print(f"[red]Agent not found: {agent_path}[/red]")
                console.print("\nSearched in:")
                console.print("  • Package agents")
                console.print("  • Current directory")
                console.print("  • .tsugite/")
                console.print("  • ./agents/")
                console.print("  • Global locations (use 'agents list --global' to see)")
                raise typer.Exit(1)

        agent = parse_agent_file(agent_file)
        config = agent.config

        console.print(f"[cyan]Agent:[/cyan] [bold]{config.name}[/bold]\n")

        if config.description:
            console.print(f"[bold]Description:[/bold] {config.description}\n")

        if config.model:
            console.print(f"[bold]Model:[/bold] {config.model}")
        else:
            console.print("[bold]Model:[/bold] [dim](uses default)[/dim]")

        console.print(f"[bold]Max Steps:[/bold] {config.max_turns}")

        if config.tools:
            console.print(f"\n[bold]Tools ({len(config.tools)}):[/bold]")
            for tool in config.tools:
                console.print(f"  • {tool}")

        if config.extends:
            console.print(f"\n[bold]Extends:[/bold] {config.extends}")

        if show_inheritance:
            from tsugite.agent_utils import build_inheritance_chain

            try:
                chain = build_inheritance_chain(agent_file)
                console.print("\n[bold cyan]Inheritance Chain:[/bold cyan]")

                if len(chain) == 1:
                    console.print("  [dim](no inheritance)[/dim]")
                else:
                    for i, (agent_name, agent_path) in enumerate(chain):
                        is_current = i == len(chain) - 1
                        prefix = "  └─" if is_current else "  ├─"

                        if is_current:
                            console.print(f"{prefix} [bold]{agent_name}[/bold] [dim](current)[/dim]")
                        else:
                            console.print(f"{prefix} {agent_name} [dim]({agent_path.name})[/dim]")
                            if i < len(chain) - 1:
                                console.print("  │")
            except Exception as e:
                console.print(f"\n[yellow]Could not build inheritance chain: {e}[/yellow]")

        console.print()

    except Exception as e:
        console.print(f"[red]Failed to load agent: {e}[/red]")
        raise typer.Exit(1)
