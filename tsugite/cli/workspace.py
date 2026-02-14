"""Workspace management CLI commands."""

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from tsugite.workspace import Workspace, WorkspaceManager, WorkspaceNotFoundError, WorkspaceSession
from tsugite.workspace.templates import list_persona_templates

workspace_app = typer.Typer(help="Manage workspaces")
console = Console()


def _load_workspace_or_exit(manager: WorkspaceManager, name: str) -> Workspace:
    """Load workspace or exit with error message.

    Args:
        manager: WorkspaceManager instance
        name: Workspace name

    Returns:
        Loaded workspace

    Raises:
        typer.Exit: If workspace not found
    """
    try:
        return manager.load_workspace(name)
    except WorkspaceNotFoundError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@workspace_app.command("list")
def list_workspaces():
    """List available workspaces."""
    manager = WorkspaceManager()
    workspaces = manager.list_workspaces()

    if not workspaces:
        console.print("[yellow]No workspaces found.[/yellow]")
        console.print("\nCreate one with: tsu workspace init <name>")
        return

    table = Table(title="Available Workspaces")
    table.add_column("Name", style="cyan")
    table.add_column("Path", style="dim")

    for name in workspaces:
        path = manager.find_workspace_path(name)
        table.add_row(name, str(path))

    console.print(table)


@workspace_app.command("init")
def init_workspace(
    name: str = typer.Argument(..., help="Workspace name"),
    persona: Optional[str] = typer.Option(None, "--persona", "-p", help="Persona template name"),
    user: Optional[str] = typer.Option(None, "--user", "-u", help="Your name"),
    path: Optional[str] = typer.Option(None, "--path", help="Custom workspace path"),
    git: Optional[bool] = typer.Option(None, "--git/--no-git", help="Initialize git repository"),
    onboard: bool = typer.Option(False, "--onboard", help="Run interactive onboarding after creation"),
):
    """Create a new workspace."""
    manager = WorkspaceManager()

    # Validate persona template if provided
    if persona:
        available_templates = list_persona_templates()
        if persona not in available_templates:
            console.print(f"[red]Persona template '{persona}' not found.[/red]")
            console.print(f"Available templates: {', '.join(available_templates)}")
            raise typer.Exit(1)

    # Ask about git if not specified
    init_git = git
    if git is None:
        init_git = typer.confirm("Initialize git repository?", default=True)

    workspace_path = Path(path).expanduser() if path else None

    try:
        workspace = manager.create_workspace(
            name=name,
            path=workspace_path,
            persona_template=persona,
            user_name=user,
            init_git=init_git,
        )
        console.print(f"[green]✓[/green] Created workspace: {workspace.name}")
        console.print(f"  Path: {workspace.path}")

        if persona:
            console.print(f"  Persona: {persona}")
        if user:
            console.print(f"  User: {user}")
        if init_git:
            console.print("  Git: initialized")

        console.print(f'\nRun with: tsu run <agent> --workspace {name} "<prompt>"')

        # Run onboarding if requested
        if onboard:
            console.print()
            onboard_workspace(name, model=None)

    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@workspace_app.command("onboard")
def onboard_workspace(
    name: str = typer.Argument(..., help="Workspace name"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Override model"),
):
    """Run interactive onboarding for a workspace."""
    from tsugite.cli.helpers import PathContext, load_and_validate_agent
    from tsugite.options import ExecutionOptions, HistoryOptions
    from tsugite.ui.repl_chat import run_repl_chat
    from tsugite.workspace.context import build_workspace_attachments

    manager = WorkspaceManager()
    workspace = _load_workspace_or_exit(manager, name)

    console.print(f"[dim]Starting onboarding for workspace '{name}'...[/dim]\n")

    # Load the onboard builtin agent
    _, agent_path, _ = load_and_validate_agent("onboard", console)

    # Build workspace attachments
    workspace_attachments = [str(att.source) for att in build_workspace_attachments(workspace)]

    # Run as REPL chat session (multi-turn conversation)
    exec_opts = ExecutionOptions(model_override=model, stream=True)
    history_opts = HistoryOptions(enabled=False)

    path_context = PathContext(
        invoked_from=Path.cwd(),
        workspace_dir=workspace.path,
        effective_cwd=workspace.path,
    )

    run_repl_chat(
        agent_path=agent_path,
        exec_options=exec_opts,
        history_options=history_opts,
        resume_turns=None,
        path_context=path_context,
        workspace_attachments=workspace_attachments,
    )

    console.print(f"\n[green]✓[/green] Workspace '{name}' onboarding complete")


@workspace_app.command("info")
def show_info(
    name: str = typer.Argument(..., help="Workspace name"),
):
    """Show workspace information."""
    manager = WorkspaceManager()
    workspace = _load_workspace_or_exit(manager, name)

    console.print(f"[bold]Workspace: {workspace.name}[/bold]")
    console.print(f"Path: {workspace.path}")

    # Check for workspace files
    console.print("\n[bold]Workspace Files:[/bold]")
    files = workspace.get_workspace_files()
    if files:
        for file in files:
            console.print(f"  ✓ {file.name}")
    else:
        console.print("  [dim]None found[/dim]")

    # Check for memory files
    console.print("\n[bold]Recent Memory:[/bold]")
    memory_files = workspace.get_memory_files()
    if memory_files:
        for file in memory_files:
            console.print(f"  ✓ {file.name}")
    else:
        console.print("  [dim]None found[/dim]")

    # Check for workspace-specific agents
    console.print("\n[bold]Workspace Agents:[/bold]")
    if workspace.agents_dir.exists():
        agent_files = list(workspace.agents_dir.glob("*.md"))
        if agent_files:
            for file in agent_files:
                console.print(f"  ✓ {file.stem}")
        else:
            console.print("  [dim]None found[/dim]")
    else:
        console.print("  [dim]Directory not created[/dim]")

    # Check for workspace-specific skills
    console.print("\n[bold]Workspace Skills:[/bold]")
    if workspace.skills_dir.exists():
        skill_files = list(workspace.skills_dir.glob("**/*.md"))
        if skill_files:
            for file in skill_files:
                console.print(f"  ✓ {file.relative_to(workspace.skills_dir)}")
        else:
            console.print("  [dim]None found[/dim]")
    else:
        console.print("  [dim]Directory not created[/dim]")


@workspace_app.command("session")
def manage_session(
    name: str = typer.Argument(..., help="Workspace name"),
    compact: bool = typer.Option(False, "--compact", help="Force session compaction"),
    new: bool = typer.Option(False, "--new", help="Start fresh session"),
    history: bool = typer.Option(False, "--history", help="List archived sessions"),
):
    """Manage workspace session."""
    manager = WorkspaceManager()
    workspace = _load_workspace_or_exit(manager, name)

    session = WorkspaceSession(workspace)

    if history:
        console.print(f"[bold]Archived Sessions for {workspace.name}:[/bold]\n")
        archived = session.list_archived()
        if archived:
            for archive_path in archived:
                console.print(f"  {archive_path.name}")
        else:
            console.print("  [dim]No archived sessions[/dim]")
        return

    if new:
        session_id = session.start_new()
        console.print(f"[green]✓[/green] Started new session: {session_id}")
        return

    if compact:
        if not workspace.session_path.exists():
            console.print("[yellow]No active session to compact[/yellow]")
            return

        session_id = session.compact()
        console.print(f"[green]✓[/green] Compacted session, new ID: {session_id}")
        return

    # Show session info
    info = session.get_info()

    if not info.conversation_id:
        console.print("[yellow]No active session[/yellow]")
        console.print(f'Run with: tsu run <agent> --workspace {name} "<prompt>"')
        return

    console.print(f"[bold]Session for {workspace.name}:[/bold]")
    console.print(f"  ID: {info.conversation_id}")
    console.print(f"  Messages: {info.message_count}")
    console.print(f"  Tokens: ~{info.token_estimate:,}")

    if info.last_agent:
        console.print(f"  Last Agent: {info.last_agent}")

    if info.last_updated:
        console.print(f"  Last Updated: {info.last_updated.strftime('%Y-%m-%d %H:%M:%S')}")

    if session.should_compact():
        console.print("\n[yellow]⚠ Session approaching context limit[/yellow]")
        console.print("Consider compacting: tsu workspace session --compact")


@workspace_app.command("templates")
def list_templates():
    """List available persona templates."""
    templates = list_persona_templates()

    if not templates:
        console.print("[yellow]No persona templates found.[/yellow]")
        return

    console.print("[bold]Available Persona Templates:[/bold]\n")
    for template in templates:
        console.print(f"  • {template}")

    console.print("\nUse with: tsu workspace init <name> --persona <template>")


__all__ = ["workspace_app"]
