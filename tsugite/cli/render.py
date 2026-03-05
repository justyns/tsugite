"""CLI render command - preview agent templates."""

from pathlib import Path
from typing import List, Optional

import typer
from rich.panel import Panel

from .helpers import (
    assemble_prompt_with_attachments,
    change_to_root_directory,
    inject_auto_context_if_enabled,
    load_and_validate_agent,
)


def render(
    agent_path: Optional[str] = typer.Argument(
        None, help="Path to agent markdown file or builtin agent name (optional when using --continue)"
    ),
    prompt: Optional[str] = typer.Argument(default="", help="Prompt/task for the agent (optional)"),
    root: Optional[str] = typer.Option(None, "--root", help="Working directory"),
    no_color: bool = typer.Option(False, "--no-color", help="Disable ANSI colors"),
    verbose: bool = typer.Option(False, "--verbose", help="Show full attachment content (default: truncated)"),
    raw: bool = typer.Option(False, "--raw", help="Show raw Jinja templates in instructions without rendering"),
    attachment: Optional[List[str]] = typer.Option(
        None, "-f", "--attachment", help="Attachment(s) to include (repeatable)"
    ),
    refresh_cache: bool = typer.Option(False, "--refresh-cache", help="Force refresh cached attachment content"),
    auto_context: Optional[bool] = typer.Option(
        None,
        "--auto-context/--no-auto-context",
        help="Enable/disable auto-context attachments (overrides config/agent)",
    ),
    continue_conversation: bool = typer.Option(
        False, "--continue", "-c", help="Show prompt for continuing conversation"
    ),
    conversation_id: Optional[str] = typer.Option(
        None, "--conversation-id", help="Specific conversation ID (use with --continue)"
    ),
):
    """Render an agent template without executing it.

    Examples:
        tsu render agent.md "prompt"
        tsu render +builtin-default "prompt"
        tsu render builtin-default "prompt"
        tsu render --continue "prompt"  # Auto-detects agent
        tsu render agent.md "prompt" --continue
        tsu render --continue --conversation-id CONV_ID "prompt"
    """
    from tsugite.agent_preparation import AgentPreparer

    from . import console

    if no_color:
        console.no_color = True

    # Handle conversation continuation
    continue_conversation_id = None
    if continue_conversation:
        from tsugite.agent_runner.history_integration import get_latest_conversation

        if conversation_id:
            continue_conversation_id = conversation_id
            console.print(f"[cyan]Rendering for conversation: {continue_conversation_id}[/cyan]")
        else:
            continue_conversation_id = get_latest_conversation()
            if not continue_conversation_id:
                console.print("[red]No conversations found[/red]")
                raise typer.Exit(1)
            console.print(f"[cyan]Rendering for latest conversation: {continue_conversation_id}[/cyan]")

    # Auto-detect agent from conversation if not specified
    if continue_conversation_id and not agent_path:
        from tsugite.history import SessionStorage, get_history_dir

        session_path = get_history_dir() / f"{continue_conversation_id}.jsonl"
        try:
            storage = SessionStorage.load(session_path)
            agent_name = storage.agent
        except Exception:
            console.print(f"[red]Could not load metadata for conversation: {continue_conversation_id}[/red]")
            raise typer.Exit(1)

        agent_path = f"+{agent_name}"
        console.print(f"[cyan]Auto-detected agent from conversation: {agent_name}[/cyan]")

    # Validate agent_path is provided
    if not agent_path:
        console.print("[red]Error: AGENT_PATH is required (or use --continue to auto-detect)[/red]")
        raise typer.Exit(1)

    with change_to_root_directory(root, console):
        try:
            agent, agent_file_path, agent_display_name = load_and_validate_agent(agent_path, console)

            base_dir = Path.cwd()

            # Inject auto-context if enabled
            agent_attachments = inject_auto_context_if_enabled(
                agent.config.attachments,
                agent.config.auto_context,
                cli_override=auto_context,
            )

            prompt_updated, resolved_attachments = assemble_prompt_with_attachments(
                prompt=prompt,
                agent_attachments=agent_attachments,
                cli_attachments=attachment,
                base_dir=base_dir,
                refresh_cache=refresh_cache,
                console=console,
            )

            context = {}

            preparer = AgentPreparer()
            prepared = preparer.prepare(
                agent=agent,
                prompt=prompt_updated,
                skip_tool_directives=True,
                context=context,
                attachments=resolved_attachments,
            )

            # Display what will be sent to LLM
            console.print(
                Panel(
                    f"[cyan]Agent:[/cyan] {agent_display_name}\n"
                    f"[cyan]Prompt:[/cyan] {prompt}\n"
                    f"[cyan]Directory:[/cyan] {Path.cwd()}",
                    title="Tsugite Template Renderer",
                    border_style="green",
                )
            )

            # Show message structure
            console.print()
            console.rule(
                "[bold yellow]MESSAGE STRUCTURE[/bold yellow] [dim](sent to LLM as separate content blocks)[/dim]",
                style="yellow",
            )

            # Message 1: System (role: system)
            console.print()
            console.rule("[bold cyan]Message 1: System Role[/bold cyan]", style="cyan", align="left")

            # Content Block 1: System Instructions
            console.print()
            console.rule("[dim]Content Block 1: System Instructions[/dim]", style="dim", align="left")
            console.print(prepared.system_message)

            # Display attachments if they exist (as additional content blocks in system message)
            if prepared.attachments:
                from tsugite.attachments.base import AttachmentContentType

                for idx, att in enumerate(prepared.attachments, start=2):
                    console.print()
                    console.rule(
                        f"[dim]Content Block {idx}: Attachment - {att.name}[/dim]",
                        style="dim",
                        align="left",
                    )
                    console.print(f"[yellow]<Attachment: {att.name}>[/yellow]")

                    if att.content_type == AttachmentContentType.TEXT and att.content:
                        if verbose:
                            console.print(att.content)
                        else:
                            lines = att.content.split("\n")
                            if len(lines) > 20:
                                preview = "\n".join(lines[:10])
                                preview += f"\n[dim]... ({len(lines) - 15} lines truncated, use --verbose to see all) ...[/dim]\n"
                                preview += "\n".join(lines[-5:])
                                console.print(preview)
                            else:
                                console.print(att.content)
                    elif att.source_url:
                        console.print(f"[dim][{att.content_type.value}: {att.source_url}][/dim]")
                    else:
                        console.print(f"[dim][{att.content_type.value} file: {att.mime_type}][/dim]")

                    console.print(f"[yellow]</Attachment: {att.name}>[/yellow]")

            # Display loaded skills (as additional content blocks in system message)
            if prepared.skills:
                next_idx = len(prepared.attachments) + 2 if prepared.attachments else 2
                for idx, skill in enumerate(prepared.skills, start=next_idx):
                    console.print()
                    console.rule(
                        f"[dim]Content Block {idx}: Skill - {skill.name}[/dim]",
                        style="dim",
                        align="left",
                    )
                    console.print(f"[magenta]<Skill: {skill.name}>[/magenta]")
                    if verbose:
                        console.print(skill.content)
                    else:
                        lines = skill.content.split("\n")
                        if len(lines) > 20:
                            preview = "\n".join(lines[:10])
                            preview += (
                                f"\n[dim]... ({len(lines) - 15} lines truncated, use --verbose to see all) ...[/dim]\n"
                            )
                            preview += "\n".join(lines[-5:])
                            console.print(preview)
                        else:
                            console.print(skill.content)
                    console.print(f"[magenta]</Skill: {skill.name}>[/magenta]")

            # Message 2: User (role: user)
            console.print()
            console.rule("[bold cyan]Message 2: User Role[/bold cyan]", style="cyan", align="left")
            console.print()
            console.rule("[dim]Content: User Task/Prompt[/dim]", style="dim", align="left")
            console.print(prepared.user_message)
            console.print()
            console.rule(style="dim")

        except Exception as e:
            console.print(f"[red]Render error: {e}[/red]")
            raise typer.Exit(1)
