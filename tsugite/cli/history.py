"""History management CLI commands."""

import json
from datetime import datetime

import typer
from rich.console import Console
from rich.table import Table

from tsugite.history import (
    ConversationMetadata,
    Turn,
    get_history_dir,
    load_conversation,
    query_index,
    rebuild_index,
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
            # JSON output - convert Pydantic models to dicts
            turns_as_dicts = [t.model_dump(mode="json") for t in turns]
            output = json.dumps(turns_as_dicts, indent=2, ensure_ascii=False)
            # Use plain print to avoid Rich wrapping that breaks JSON
            print(output)

        elif format == "markdown":
            # Markdown output
            metadata = next((t for t in turns if isinstance(t, ConversationMetadata)), None)

            console.print(f"# Conversation: {conversation_id}\n")
            if metadata:
                console.print(f"- **Agent**: {metadata.agent}")
                console.print(f"- **Model**: {metadata.model}")
                console.print(f"- **Machine**: {metadata.machine}")
                console.print(f"- **Created**: {metadata.created_at}\n")
            console.print("---\n")

            for turn in turns:
                if isinstance(turn, Turn):
                    console.print(f"## Turn ({turn.timestamp})\n")
                    console.print(f"**User**: {turn.user}\n")
                    console.print(f"**Assistant**: {turn.assistant}\n")

                    if turn.tools:
                        console.print(f"*Tools*: {', '.join(turn.tools)}\n")

                    # Display execution steps if available
                    if turn.steps:
                        console.print("### Execution Steps\n")
                        for step in turn.steps:
                            step_num = step.get("step_number", "?")
                            thought = step.get("thought", "").strip()
                            code = step.get("code", "").strip()
                            output = step.get("output", "").strip()
                            error = step.get("error")
                            tools_called = step.get("tools_called", [])

                            console.print(f"**Step {step_num}**\n")
                            if thought:
                                console.print(f"*Thought*: {thought}\n")
                            if tools_called:
                                console.print(f"*Tools used*: {', '.join(tools_called)}\n")
                            if code:
                                console.print("*Code*:")
                                console.print(f"```python\n{code}\n```\n")
                            if output:
                                console.print(f"*Output*: {output}\n")
                            if error:
                                console.print(f"*Error*: {error}\n")

                    console.print(f"*Tokens*: {turn.tokens or 0} | *Cost*: ${turn.cost or 0.0:.4f}\n")
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
