"""History management CLI commands."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.text import Text

console = Console()

history_app = typer.Typer(help="Manage conversation history")


def _format_duration(ms: Optional[int]) -> str:
    if not ms:
        return "-"
    if ms < 1000:
        return f"{ms}ms"
    secs = ms / 1000
    if secs < 60:
        return f"{secs:.1f}s"
    mins = secs / 60
    return f"{mins:.1f}m"


def _format_cost(cost: Optional[float]) -> str:
    if not cost:
        return "-"
    return f"${cost:.4f}"


def _status_style(status: Optional[str]) -> str:
    return {"success": "green", "error": "red", "interrupted": "yellow"}.get(status or "", "dim")


def _format_created(dt: Optional[datetime]) -> str:
    try:
        return dt.strftime("%Y-%m-%d %H:%M") if dt else "unknown"
    except (ValueError, TypeError):
        return "unknown"


def _parse_date(date_str: Optional[str]) -> Optional[datetime]:
    if not date_str:
        return None
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except ValueError:
        raise typer.BadParameter(f"Invalid date format: {date_str}. Use YYYY-MM-DD")


def _filter_by_date(
    session_files: List[Path],
    since_dt: Optional[datetime] = None,
    before_dt: Optional[datetime] = None,
) -> List[Path]:
    """Pre-filter session files by date extracted from filename (YYYYMMDD_HHMMSS_...)."""
    if not since_dt and not before_dt:
        return session_files
    filtered = []
    for sf in session_files:
        try:
            file_dt = datetime.strptime(sf.stem[:15], "%Y%m%d_%H%M%S").replace(tzinfo=timezone.utc)
            if since_dt and file_dt < since_dt:
                continue
            if before_dt and file_dt >= before_dt:
                continue
        except ValueError:
            pass
        filtered.append(sf)
    return filtered


def _search_events(events, query_lower: str) -> Optional[str]:
    """Search events for a query match, returning a display snippet or None."""
    for event in events:
        data = event.data
        if event.type == "user_input":
            text = data.get("text", "")
            if text and query_lower in text.lower():
                return _make_snippet(text, query_lower, prefix="User: ")
        elif event.type == "model_response":
            text = data.get("raw_content", "")
            if text and query_lower in text.lower():
                return _make_snippet(text, query_lower, prefix="Output: ")
        elif event.type == "code_execution":
            for fn in data.get("tools_called") or []:
                if fn and query_lower in fn.lower():
                    return f"Tool: {fn}"
        elif event.type == "tool_invocation":
            fn = data.get("name")
            if fn and query_lower in fn.lower():
                return f"Tool: {fn}"
    return None


def _make_snippet(text: str, query_lower: str, prefix: str = "", max_len: int = 80) -> str:
    """Extract a snippet around the first match of query in text."""
    idx = text.lower().find(query_lower)
    if idx == -1:
        return prefix + text[:max_len]
    start = max(0, idx - 20)
    end = min(len(text), idx + len(query_lower) + 40)
    snippet = text[start:end]
    if start > 0:
        snippet = "..." + snippet
    if end < len(text):
        snippet = snippet + "..."
    return prefix + snippet


@history_app.command("list")
def history_list(
    machine: str = typer.Option(None, "--machine", help="Filter by machine name"),
    agent: str = typer.Option(None, "--agent", help="Filter by agent name"),
    status: str = typer.Option(None, "--status", help="Filter by status (success, error, interrupted)"),
    tool: str = typer.Option(None, "--tool", help="Filter by tool/function name"),
    since: str = typer.Option(None, "--since", help="Only sessions after this date (YYYY-MM-DD)"),
    before: str = typer.Option(None, "--before", help="Only sessions before this date (YYYY-MM-DD)"),
    limit: int = typer.Option(20, "--limit", "-n", help="Maximum number of results"),
):
    """List conversations with optional filters.

    Examples:
        tsugite history list
        tsugite history list --agent chat_assistant --status error
        tsugite history list --since 2026-01-01 --tool shell_exec
    """
    from tsugite.history import SessionStorage, get_history_dir, list_session_files

    since_dt = _parse_date(since)
    before_dt = _parse_date(before)
    try:
        session_files = list_session_files()

        if not session_files:
            console.print("[yellow]No conversations found[/yellow]")

            history_dir = get_history_dir()
            if not history_dir.exists():
                console.print(f"\nHistory directory doesn't exist yet: {history_dir}")
                console.print("Conversations will be saved automatically when you use chat mode.")
            return

        session_files = _filter_by_date(session_files, since_dt, before_dt)

        table = Table(title=f"Conversation History ({len(session_files)} total)")
        table.add_column("ID", style="cyan", no_wrap=True)
        table.add_column("Agent", style="green")
        table.add_column("Status")
        table.add_column("Turns", justify="right")
        table.add_column("Tokens", justify="right", style="dim")
        table.add_column("Cost", justify="right", style="dim")
        table.add_column("Duration", justify="right", style="dim")
        table.add_column("Created", style="dim")

        count = 0
        for session_file in session_files:
            if count >= limit:
                break

            try:
                if agent or machine:
                    meta = SessionStorage.load_meta_fast(session_file)
                    if meta:
                        if agent and meta.data.get("agent") != agent:
                            continue
                        if machine and meta.data.get("machine") != machine:
                            continue

                storage = SessionStorage.load(session_file)
                summary = storage.summary()

                if status and summary.status != status:
                    continue
                if tool and tool not in summary.functions_called:
                    continue

                status_text = Text(summary.status or "unknown", style=_status_style(summary.status))

                table.add_row(
                    storage.session_id,
                    summary.agent or "unknown",
                    status_text,
                    str(summary.turn_count),
                    f"{summary.total_tokens:,}" if summary.total_tokens > 0 else "-",
                    _format_cost(summary.total_cost),
                    _format_duration(summary.total_duration_ms),
                    _format_created(summary.created_at),
                )
                count += 1

            except Exception:
                continue

        console.print(table)

        console.print(
            "\n[dim]Use 'tsugite history show ID' to view details, 'tsugite history search QUERY' to search[/dim]"
        )

    except Exception as e:
        console.print(f"[red]Failed to list conversations: {e}[/red]")
        raise typer.Exit(1)


@history_app.command("search")
def history_search(
    query: str = typer.Argument(help="Text to search for in prompts, outputs, and tool names"),
    agent: str = typer.Option(None, "--agent", help="Filter by agent name"),
    status: str = typer.Option(None, "--status", help="Filter by status"),
    since: str = typer.Option(None, "--since", help="Only sessions after this date (YYYY-MM-DD)"),
    limit: int = typer.Option(20, "--limit", "-n", help="Maximum results"),
):
    """Search conversation history by text content.

    Searches across user prompts, agent outputs, and tool/function names.

    Examples:
        tsugite history search "deploy"
        tsugite history search "error" --status error --since 2026-01-01
    """
    from tsugite.history import SessionStorage, SessionSummary, list_session_files

    since_dt = _parse_date(since)
    query_lower = query.lower()

    try:
        session_files = list_session_files()

        if not session_files:
            console.print("[yellow]No conversations found[/yellow]")
            return

        session_files = _filter_by_date(session_files, since_dt)

        table = Table(title=f"Search Results for '{query}'")
        table.add_column("ID", style="cyan", no_wrap=True)
        table.add_column("Agent", style="green")
        table.add_column("Match", max_width=60)
        table.add_column("Status")
        table.add_column("Created", style="dim")

        count = 0
        for session_file in session_files:
            if count >= limit:
                break

            try:
                if agent:
                    meta = SessionStorage.load_meta_fast(session_file)
                    if meta and meta.data.get("agent") != agent:
                        continue

                storage = SessionStorage.load(session_file)
                events = storage.load_events()
                summary = SessionSummary.from_events(events)

                if status and summary.status != status:
                    continue

                match_snippet = _search_events(events, query_lower)
                if not match_snippet:
                    continue

                status_text = Text(summary.status or "unknown", style=_status_style(summary.status))

                table.add_row(
                    storage.session_id,
                    summary.agent or "unknown",
                    match_snippet,
                    status_text,
                    _format_created(summary.created_at),
                )
                count += 1

            except Exception:
                continue

        if count == 0:
            console.print(f"[yellow]No matches found for '{query}'[/yellow]")
        else:
            console.print(table)

    except Exception as e:
        console.print(f"[red]Search failed: {e}[/red]")
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
    from tsugite.history import SessionStorage, SessionSummary, get_history_dir

    try:
        session_path = get_history_dir() / f"{conversation_id}.jsonl"
        storage = SessionStorage.load(session_path)
        events = storage.load_events()

        if not events:
            console.print(f"[yellow]Conversation '{conversation_id}' is empty[/yellow]")
            return

        summary = SessionSummary.from_events(events)

        if format == "json":
            output = json.dumps([e.model_dump(mode="json") for e in events], indent=2, ensure_ascii=False)
            print(output)
            return

        if format == "markdown":
            console.print(f"# Conversation: {conversation_id}\n")
            for label, value in [
                ("Agent", summary.agent),
                ("Model", summary.model),
                ("Machine", summary.machine),
                ("Created", summary.created_at),
                ("Status", summary.status),
            ]:
                if value:
                    console.print(f"- **{label}**: {value}")
            console.print()
            console.print("---\n")

            current_user: Optional[str] = None
            for event in events:
                if event.type == "user_input":
                    current_user = event.data.get("text", "")
                    console.print(f"## Turn ({event.ts})\n")
                    console.print(f"**User**: {current_user}\n")
                elif event.type == "model_response":
                    text = event.data.get("raw_content", "")
                    console.print(f"**Assistant**: {text}\n")
                elif event.type == "code_execution":
                    code = event.data.get("code", "")
                    console.print(f"```python\n{code}\n```\n")
                    console.print("---\n")
            return

        # plain
        console.print("=" * 60)
        console.print(f"Conversation: {conversation_id}")
        console.print(f"Agent: {summary.agent or 'unknown'}")
        console.print(f"Model: {summary.model or 'unknown'}")
        console.print(f"Machine: {summary.machine or 'unknown'}")
        console.print(f"Created: {summary.created_at}")
        console.print(f"Status: {summary.status or 'unknown'}")
        console.print("=" * 60)
        console.print()

        for event in events:
            if event.type == "user_input":
                console.print(f"[{event.ts}]")
                console.print(f"User: {event.data.get('text', '')}")
            elif event.type == "model_response":
                console.print(f"Assistant: {event.data.get('raw_content', '')}")
                console.print()
                console.print("-" * 60)
                console.print()
            elif event.type == "code_execution":
                fn_list = event.data.get("tools_called") or []
                if fn_list:
                    console.print(f"  Functions: {', '.join(fn_list)}")

    except FileNotFoundError:
        console.print(f"[red]Conversation '{conversation_id}' not found[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Failed to load conversation: {e}[/red]")
        raise typer.Exit(1)


@history_app.command("rebuild-index")
def history_rebuild_index():
    """Rebuild conversation index from JSONL files.

    Note: V2 storage format no longer uses a separate index file.
    This command scans all session files to verify they are valid.
    """
    from tsugite.history import SessionStorage, list_session_files

    try:
        console.print("Scanning session files...")
        session_files = list_session_files()
        valid_count = 0
        for session_file in session_files:
            try:
                SessionStorage.load(session_file)
                valid_count += 1
            except Exception as e:
                console.print(f"[yellow]Skipping invalid session {session_file.stem}: {e}[/yellow]")

        console.print(f"[green]Found {valid_count} valid sessions[/green]")

    except Exception as e:
        console.print(f"[red]Failed to scan sessions: {e}[/red]")
        raise typer.Exit(1)
