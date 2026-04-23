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


def _convert_old_records(records: list[dict]) -> list[dict]:
    """Translate pre-event-log records (session_meta/turn/etc.) into new events.

    Preserves user/assistant text and compaction summaries. Drops context,
    context_update, and hook_execution records — the new format isn't a
    superset and the user opted into losing fine-grained tool detail.
    """
    events: list[dict] = []
    first_ts = records[0].get("created_at") if records else None
    for r in records:
        rt = r.get("type")
        if rt == "session_meta":
            events.append({
                "type": "session_start",
                "ts": r.get("created_at") or first_ts,
                "data": {
                    k: v
                    for k, v in {
                        "agent": r.get("agent") or "unknown",
                        "model": r.get("model") or "unknown",
                        "machine": r.get("machine") or "unknown",
                        "workspace": r.get("workspace"),
                        "parent_session": r.get("compacted_from"),
                    }.items()
                    if v is not None
                },
            })
        elif rt == "turn":
            ts = r.get("timestamp") or first_ts
            user_text = ""
            for msg in r.get("messages") or []:
                if msg.get("role") == "user":
                    content = msg.get("content", "")
                    if isinstance(content, str):
                        user_text = content
                    break
            if not user_text:
                user_text = r.get("user_summary") or ""
            assistant_text = r.get("final_answer") or ""
            if not assistant_text:
                for msg in reversed(r.get("messages") or []):
                    if msg.get("role") == "assistant":
                        content = msg.get("content", "")
                        if isinstance(content, str):
                            assistant_text = content
                        break

            events.append({"type": "user_input", "ts": ts, "data": {"text": user_text}})
            tokens = r.get("tokens")
            response_data = {"raw_content": assistant_text}
            if r.get("model"):
                response_data["model"] = r["model"]
            if tokens:
                response_data["usage"] = {"total_tokens": tokens}
            if r.get("cost"):
                response_data["cost"] = r["cost"]
            events.append({"type": "model_response", "ts": ts, "data": response_data})
        elif rt == "compaction_summary":
            events.append({
                "type": "compaction",
                "ts": r.get("timestamp") or first_ts,
                "data": {
                    k: v
                    for k, v in {
                        "summary": r.get("summary", ""),
                        "replaced_count": r.get("previous_turns", 0),
                        "retained_count": r.get("retained_turns", 0),
                        "reason": r.get("compaction_reason"),
                    }.items()
                    if v is not None
                },
            })
        elif rt == "session_status":
            events.append({
                "type": "session_end",
                "ts": r.get("timestamp") or first_ts,
                "data": {
                    k: v
                    for k, v in {
                        "status": r.get("status", "success"),
                        "error_message": r.get("error_message"),
                    }.items()
                    if v is not None
                },
            })
        # context / context_update / hook_execution are dropped intentionally.
    return events


def migrate_session_file(path: Path, *, backup: bool, dry_run: bool) -> str:
    """Migrate one old-format session file in place.

    Returns a short status string ('migrated ...', 'skipped:...', or 'would-migrate ...').
    Raises on write errors; the original file is untouched until the rewrite succeeds.
    """
    if not path.exists() or path.stat().st_size == 0:
        return "skipped:empty"

    try:
        with open(path, "r", encoding="utf-8") as f:
            first_line = f.readline().strip()
    except OSError:
        return "skipped:read_error"

    if not first_line:
        return "skipped:empty"

    try:
        first = json.loads(first_line)
    except json.JSONDecodeError:
        return "skipped:invalid_json"

    ftype = first.get("type")
    if ftype == "session_start":
        return "skipped:already_new"
    if ftype != "session_meta":
        return f"skipped:unknown_format({ftype})"

    records: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if not records:
        return "skipped:no_records"

    events = _convert_old_records(records)
    if dry_run:
        return f"would-migrate ({len(records)} records -> {len(events)} events)"

    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for e in events:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    if backup:
        bak = path.with_suffix(path.suffix + ".bak")
        path.rename(bak)
    tmp.replace(path)

    return f"migrated ({len(records)} records -> {len(events)} events)"


def migrate_path(root: Path, *, backup: bool, dry_run: bool, recursive: bool) -> tuple[int, int, int]:
    """Migrate all session JSONL files under ``root``.

    Returns (migrated, skipped, failed).
    """
    if not root.exists():
        return (0, 0, 0)
    pattern = "**/*.jsonl" if recursive else "*.jsonl"
    migrated = skipped = failed = 0
    for path in sorted(root.glob(pattern)):
        try:
            status = migrate_session_file(path, backup=backup, dry_run=dry_run)
        except Exception as e:
            console.print(f"[red]{path}: {e}[/red]")
            failed += 1
            continue
        if status.startswith("migrated") or status.startswith("would-migrate"):
            console.print(f"[green]{path}[/green] {status}")
            migrated += 1
        else:
            skipped += 1
    return (migrated, skipped, failed)


def migrate_daemon_sessions(
    daemon_dir: Path, history_dir: Path, *, backup: bool, dry_run: bool
) -> tuple[int, int]:
    """Merge legacy ``daemon/sessions/{id}.jsonl`` UI events into the matching
    ``history/{id}.jsonl`` file.

    The daemon used to keep a parallel event log with reactions, prompt_snapshots
    and other UI telemetry. Now those events live in the same per-session JSONL
    as conversation events, so the daemon log is redundant. This function moves
    each daemon file's events into the matching history file (sorted by
    timestamp), then deletes (or backs up) the daemon source file.

    Returns (merged, skipped).
    """
    if not daemon_dir.exists():
        return (0, 0)

    merged = skipped = 0
    for daemon_file in sorted(daemon_dir.glob("*.jsonl")):
        sid = daemon_file.stem
        history_file = history_dir / f"{sid}.jsonl"

        try:
            daemon_events = []
            with open(daemon_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        daemon_events.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        except OSError:
            skipped += 1
            continue

        if not daemon_events:
            skipped += 1
            continue

        # Convert to new Event shape, sorted by timestamp
        new_events = []
        for e in daemon_events:
            new_events.append({
                "type": e.get("type", "unknown"),
                "ts": e.get("timestamp"),
                "data": {k: v for k, v in e.items() if k not in ("type", "timestamp")},
            })
        new_events.sort(key=lambda e: e.get("ts") or "")

        if dry_run:
            console.print(f"[green]{daemon_file}[/green] would merge {len(new_events)} events into {history_file.name}")
            merged += 1
            continue

        history_file.parent.mkdir(parents=True, exist_ok=True)
        with open(history_file, "a", encoding="utf-8") as f:
            for e in new_events:
                clean_data = {k: v for k, v in e["data"].items() if v is not None}
                f.write(json.dumps({"type": e["type"], "ts": e["ts"], "data": clean_data}, ensure_ascii=False) + "\n")

        if backup:
            daemon_file.rename(daemon_file.with_suffix(daemon_file.suffix + ".bak"))
        else:
            daemon_file.unlink()
        console.print(f"[green]{daemon_file}[/green] merged {len(new_events)} events into {history_file.name}")
        merged += 1

    return (merged, skipped)


@history_app.command("migrate")
def history_migrate(
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would change without writing"),
    backup: bool = typer.Option(True, "--backup/--no-backup", help="Keep .bak copies of originals"),
):
    """Convert pre-event-log session JSONLs to the per-event format.

    Scans:
      - the main history dir
      - each workspace's archived sessions
      - the legacy daemon UI event log (merged into matching history files)

    Runs idempotently — already-new files are skipped.
    """
    from tsugite.config import get_xdg_data_path
    from tsugite.history import get_history_dir

    history_dir = get_history_dir()
    data_root = get_xdg_data_path("")  # ~/.local/share/tsugite/

    total_migrated = total_skipped = total_failed = 0

    console.print(f"[bold]Scanning history dir:[/bold] {history_dir}")
    m, s, f = migrate_path(history_dir, backup=backup, dry_run=dry_run, recursive=False)
    total_migrated += m
    total_skipped += s
    total_failed += f

    workspaces_root = data_root / "workspaces"
    if workspaces_root.exists():
        console.print(f"\n[bold]Scanning workspace archives:[/bold] {workspaces_root}")
        for ws in sorted(p for p in workspaces_root.iterdir() if p.is_dir()):
            sessions_dir = ws / "sessions"
            if sessions_dir.exists():
                m, s, f = migrate_path(sessions_dir, backup=backup, dry_run=dry_run, recursive=False)
                total_migrated += m
                total_skipped += s
                total_failed += f

    daemon_dir = data_root / "daemon" / "sessions"
    if daemon_dir.exists():
        console.print(f"\n[bold]Merging legacy daemon event log:[/bold] {daemon_dir}")
        merged, skipped = migrate_daemon_sessions(daemon_dir, history_dir, backup=backup, dry_run=dry_run)
        total_migrated += merged
        total_skipped += skipped

    verb = "Would migrate" if dry_run else "Migrated"
    console.print(
        f"\n[bold]{verb}:[/bold] {total_migrated}  [dim]skipped:[/dim] {total_skipped}  [dim]failed:[/dim] {total_failed}"
    )
    if not dry_run and backup and total_migrated:
        console.print("[dim]Originals saved as *.bak — delete when satisfied.[/dim]")


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
