"""History management CLI commands."""

import json
from datetime import datetime
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from tsugite.history import (
    delete_conversation,
    get_history_dir,
    load_conversation,
    prune_conversations,
    query_index,
    rebuild_index,
    remove_from_index,
)
from tsugite.ui.chat_history import format_conversation_for_display

console = Console()

history_app = typer.Typer(help="Manage conversation history")


@history_app.command("list")
def history_list(
    machine: str = typer.Option(None, "--machine", help="Filter by machine name"),
    agent: str = typer.Option(None, "--agent", help="Filter by agent name"),
    limit: int = typer.Option(20, "--limit", "-n", help="Maximum number of results"),
):
    """List conversations with optional filters.

    Examples:
        tsugite history list
        tsugite history list --machine laptop
        tsugite history list --agent chat_assistant --limit 10
    """
    try:
        conversations = query_index(machine=machine, agent=agent, limit=limit)

        if not conversations:
            console.print("[yellow]No conversations found[/yellow]")

            history_dir = get_history_dir()
            if not history_dir.exists():
                console.print(f"\nHistory directory doesn't exist yet: {history_dir}")
                console.print("Conversations will be saved automatically when you use chat mode.")
            return

        table = Table(title=f"Conversation History ({len(conversations)} found)")
        table.add_column("ID", style="cyan", no_wrap=True)
        table.add_column("Agent", style="green")
        table.add_column("Machine", style="dim")
        table.add_column("Turns", justify="right")
        table.add_column("Tokens", justify="right", style="dim")
        table.add_column("Updated", style="dim")

        for conv in conversations:
            conv_id = conv.get("conversation_id", "unknown")
            agent_name = conv.get("agent", "unknown")
            machine_name = conv.get("machine", "unknown")
            turn_count = conv.get("turn_count", 0)
            total_tokens = conv.get("total_tokens", 0)
            updated_at = conv.get("updated_at", "unknown")

            # Format date
            try:
                dt = datetime.fromisoformat(updated_at)
                updated_str = dt.strftime("%Y-%m-%d %H:%M")
            except (ValueError, TypeError):
                updated_str = updated_at[:16] if len(updated_at) > 16 else updated_at

            table.add_row(
                conv_id,
                agent_name,
                machine_name,
                str(turn_count),
                f"{total_tokens:,}" if total_tokens > 0 else "-",
                updated_str,
            )

        console.print(table)

        # Show usage hint
        console.print("\n[dim]Use 'tsugite history show CONVERSATION_ID' to view details[/dim]")

    except Exception as e:
        console.print(f"[red]Failed to list conversations: {e}[/red]")
        raise typer.Exit(1)


@history_app.command("show")
def history_show(
    conversation_id: str = typer.Argument(help="Conversation ID to show"),
    format: str = typer.Option("plain", "--format", "-f", help="Output format (plain, json, markdown)"),
):
    """Show full conversation details.

    Examples:
        tsugite history show 20251024_103000_chat_abc123
        tsugite history show 20251024_103000_chat_abc123 --format json
        tsugite history show 20251024_103000_chat_abc123 --format markdown
    """
    try:
        turns = load_conversation(conversation_id)

        if not turns:
            console.print(f"[yellow]Conversation '{conversation_id}' is empty[/yellow]")
            return

        if format == "json":
            # JSON output
            output = json.dumps(turns, indent=2, ensure_ascii=False)
            console.print(output)

        elif format == "markdown":
            # Markdown output
            metadata = next((t for t in turns if t.get("type") == "metadata"), {})
            console.print(f"# Conversation: {conversation_id}\n")
            console.print(f"- **Agent**: {metadata.get('agent', 'unknown')}")
            console.print(f"- **Model**: {metadata.get('model', 'unknown')}")
            console.print(f"- **Machine**: {metadata.get('machine', 'unknown')}")
            console.print(f"- **Created**: {metadata.get('created_at', 'unknown')}\n")
            console.print("---\n")

            for turn in turns:
                if turn.get("type") == "turn":
                    console.print(f"## Turn ({turn.get('timestamp', '')})\n")
                    console.print(f"**User**: {turn.get('user', '')}\n")
                    console.print(f"**Assistant**: {turn.get('assistant', '')}\n")

                    if turn.get("tools"):
                        console.print(f"*Tools*: {', '.join(turn.get('tools', []))}\n")

                    console.print(f"*Tokens*: {turn.get('tokens', 0)} | *Cost*: ${turn.get('cost', 0.0):.4f}\n")
                    console.print("---\n")

        else:
            # Plain text output (default)
            output = format_conversation_for_display(turns)
            console.print(output)

    except FileNotFoundError:
        console.print(f"[red]Conversation '{conversation_id}' not found[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Failed to load conversation: {e}[/red]")
        raise typer.Exit(1)


@history_app.command("export")
def history_export(
    conversation_id: str = typer.Argument(help="Conversation ID to export"),
    output_file: Path = typer.Argument(help="Output file path"),
    format: str = typer.Option("json", "--format", "-f", help="Export format (json, markdown, plain)"),
):
    """Export conversation to a file.

    Examples:
        tsugite history export 20251024_103000_chat_abc123 conversation.json
        tsugite history export 20251024_103000_chat_abc123 conversation.md --format markdown
    """
    try:
        turns = load_conversation(conversation_id)

        if not turns:
            console.print(f"[yellow]Conversation '{conversation_id}' is empty[/yellow]")
            return

        # Generate content based on format
        if format == "json":
            content = json.dumps(turns, indent=2, ensure_ascii=False)
        elif format == "markdown":
            # Generate markdown
            lines = []
            metadata = next((t for t in turns if t.get("type") == "metadata"), {})
            lines.append(f"# Conversation: {conversation_id}\n")
            lines.append(f"- **Agent**: {metadata.get('agent', 'unknown')}")
            lines.append(f"- **Model**: {metadata.get('model', 'unknown')}")
            lines.append(f"- **Machine**: {metadata.get('machine', 'unknown')}")
            lines.append(f"- **Created**: {metadata.get('created_at', 'unknown')}\n")
            lines.append("---\n")

            for turn in turns:
                if turn.get("type") == "turn":
                    lines.append(f"## Turn ({turn.get('timestamp', '')})\n")
                    lines.append(f"**User**: {turn.get('user', '')}\n")
                    lines.append(f"**Assistant**: {turn.get('assistant', '')}\n")

                    if turn.get("tools"):
                        lines.append(f"*Tools*: {', '.join(turn.get('tools', []))}\n")

                    lines.append(f"*Tokens*: {turn.get('tokens', 0)} | *Cost*: ${turn.get('cost', 0.0):.4f}\n")
                    lines.append("---\n")

            content = "\n".join(lines)
        else:
            # Plain text
            content = format_conversation_for_display(turns)

        # Write to file
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(content, encoding="utf-8")

        console.print(f"[green]✓ Exported to {output_file}[/green]")
        console.print(f"  Format: {format}")
        console.print(f"  Size: {len(content):,} bytes")

    except FileNotFoundError:
        console.print(f"[red]Conversation '{conversation_id}' not found[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Failed to export conversation: {e}[/red]")
        raise typer.Exit(1)


@history_app.command("prune")
def history_prune(
    days: int = typer.Option(None, "--days", help="Delete conversations older than N days"),
    keep: int = typer.Option(None, "--keep", help="Keep only N most recent conversations"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be deleted without deleting"),
):
    """Delete old conversations based on retention policy.

    Examples:
        tsugite history prune --days 30
        tsugite history prune --keep 100
        tsugite history prune --days 90 --dry-run
    """
    if days is None and keep is None:
        console.print("[red]Must specify either --days or --keep[/red]")
        raise typer.Exit(1)

    try:
        if dry_run:
            console.print("[yellow]DRY RUN - No files will be deleted[/yellow]\n")

            # Show what would be deleted
            from tsugite.history import list_conversation_files

            files = list_conversation_files()

            if keep is not None and len(files) > keep:
                to_delete = files[keep:]
                console.print(f"Would delete {len(to_delete)} conversations (keeping {keep} most recent):")
                for f in to_delete:
                    console.print(f"  - {f.stem}")

            if days is not None:
                from datetime import timedelta, timezone

                cutoff = datetime.now(timezone.utc) - timedelta(days=days)
                cutoff_timestamp = cutoff.timestamp()

                to_delete = [f for f in files if f.stat().st_mtime < cutoff_timestamp]
                console.print(f"\nWould delete {len(to_delete)} conversations older than {days} days:")
                for f in to_delete:
                    console.print(f"  - {f.stem}")

        else:
            deleted = prune_conversations(keep_count=keep, older_than_days=days)
            console.print(f"[green]✓ Deleted {deleted} conversations[/green]")

            if deleted > 0:
                console.print("\nRebuilding index...")
                rebuild_index()
                console.print("[green]✓ Index rebuilt[/green]")

    except Exception as e:
        console.print(f"[red]Failed to prune conversations: {e}[/red]")
        raise typer.Exit(1)


@history_app.command("rebuild-index")
def history_rebuild_index():
    """Rebuild conversation index from JSONL files.

    Scans all conversation files and rebuilds the index.
    Useful after manual file changes or index corruption.
    """
    try:
        console.print("Rebuilding conversation index...")
        count = rebuild_index()
        console.print(f"[green]✓ Indexed {count} conversations[/green]")

    except Exception as e:
        console.print(f"[red]Failed to rebuild index: {e}[/red]")
        raise typer.Exit(1)


@history_app.command("delete")
def history_delete(
    conversation_id: str = typer.Argument(help="Conversation ID to delete"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt"),
):
    """Delete a specific conversation.

    Example:
        tsugite history delete 20251024_103000_chat_abc123
    """
    if not yes:
        confirm = typer.confirm(f"Are you sure you want to delete conversation '{conversation_id}'?")
        if not confirm:
            console.print("[yellow]Cancelled[/yellow]")
            return

    try:
        if delete_conversation(conversation_id):
            # Also remove from index
            remove_from_index(conversation_id)
            console.print(f"[green]✓ Deleted conversation '{conversation_id}'[/green]")
        else:
            console.print(f"[yellow]Conversation '{conversation_id}' not found[/yellow]")

    except Exception as e:
        console.print(f"[red]Failed to delete conversation: {e}[/red]")
        raise typer.Exit(1)
