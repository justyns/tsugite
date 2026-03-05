"""CLI chat command - interactive chat sessions."""

from typing import Optional

import typer

from tsugite.options import ExecutionOptions, HistoryOptions

from .helpers import load_and_validate_agent, workspace_directory_context
from .run import (
    _build_workspace_attachments,
    _check_and_run_onboarding,
    _resolve_effective_workspace,
)

# Chat history limit - keeps last N turns to balance context retention vs memory usage
DEFAULT_MAX_CHAT_HISTORY = 50


def chat(
    agent: Optional[str] = typer.Argument(None, help="Agent name or path (optional, uses default if not provided)"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Override agent model"),
    max_history: int = typer.Option(DEFAULT_MAX_CHAT_HISTORY, "--max-history", help="Maximum turns to keep in context"),
    stream: bool = typer.Option(False, "--stream", help="Stream LLM responses in real-time"),
    no_history: bool = typer.Option(False, "--no-history", help="Disable conversation history persistence"),
    continue_: Optional[str] = typer.Option(
        None, "--continue", "-c", help="Resume conversation by ID, or latest if no ID given"
    ),
    root: Optional[str] = typer.Option(None, "--root", help="Working directory"),
    workspace: Optional[str] = typer.Option(
        None, "--workspace", "-w", help="Workspace directory (auto-loads PERSONA.md, USER.md, MEMORY.md)"
    ),
    no_workspace: bool = typer.Option(False, "--no-workspace", help="Disable workspace (ignore default workspace)"),
):
    """Start an interactive chat session with an agent."""
    from . import console

    # Validate flag combinations
    if workspace and no_workspace:
        console.print("[red]Error: Cannot use --workspace with --no-workspace[/red]")
        raise typer.Exit(1)

    # Build option dataclasses
    exec_opts = ExecutionOptions(model_override=model, stream=stream)
    history_opts = HistoryOptions(enabled=not no_history, max_turns=max_history)

    workspace_to_use, resolved_workspace = _resolve_effective_workspace(workspace, no_workspace)
    if workspace_to_use and not resolved_workspace:
        console.print(f"[yellow]Warning: Workspace '{workspace_to_use}' not found[/yellow]")

    # Build workspace attachments
    workspace_attachments = []
    if resolved_workspace:
        resolved_workspace = _check_and_run_onboarding(resolved_workspace, workspace_to_use, model)
        workspace_attachments = _build_workspace_attachments(resolved_workspace)

    with workspace_directory_context(resolved_workspace, root, console) as path_context:
        # Handle conversation resume
        resume_turns = None

        if continue_ is not None:
            from tsugite.agent_runner.history_integration import get_latest_conversation
            from tsugite.history import get_history_dir, get_turns

            if continue_ == "" or continue_.lower() == "latest":
                history_opts.continue_id = get_latest_conversation()
                if not history_opts.continue_id:
                    console.print("[red]No conversations found to resume[/red]")
                    raise typer.Exit(1)
                console.print(f"[cyan]Resuming latest conversation: {history_opts.continue_id}[/cyan]")
            else:
                history_opts.continue_id = continue_
                console.print(f"[cyan]Resuming conversation: {history_opts.continue_id}[/cyan]")

            try:
                session_path = get_history_dir() / f"{history_opts.continue_id}.jsonl"
                resume_turns = get_turns(session_path)
                console.print(f"[cyan]Loaded {len(resume_turns)} previous turns[/cyan]")
            except FileNotFoundError:
                console.print(f"[red]Conversation not found: {history_opts.continue_id}[/red]")
                raise typer.Exit(1)
            except Exception as e:
                console.print(f"[red]Failed to load conversation: {e}[/red]")
                raise typer.Exit(1)

        agent_to_load = agent if agent else "default"
        _, primary_agent_path, _ = load_and_validate_agent(agent_to_load, console)

        from tsugite.ui.repl_chat import run_repl_chat

        # REPL defaults to streaming for better UX
        exec_opts.stream = True

        run_repl_chat(
            agent_path=primary_agent_path,
            exec_options=exec_opts,
            history_options=history_opts,
            resume_turns=resume_turns,
            path_context=path_context,
            workspace_attachments=workspace_attachments,
        )
