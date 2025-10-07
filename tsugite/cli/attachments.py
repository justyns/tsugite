"""Attachment management CLI commands."""

from pathlib import Path

import typer
from rich.console import Console

console = Console()

attachments_app = typer.Typer(help="Manage reusable text attachments")


@attachments_app.command("add")
def attachments_add(
    alias: str = typer.Argument(help="Unique name for the attachment"),
    source: str = typer.Argument(help="File path, URL, or '-' for stdin"),
):
    """Add or update an attachment.

    For stdin input ('-'), content is stored inline in attachments.json.
    For files and URLs, only the reference is stored (content fetched on demand).
    """
    from tsugite.attachments import add_attachment

    try:
        if source == "-":
            # Read from stdin - store inline
            import sys

            content = sys.stdin.read()
            add_attachment(alias, source="inline", content=content)

            console.print(f"[green]✓ Inline attachment '{alias}' saved[/green]")
            console.print("  Type: Inline text")
            console.print(f"  Size: {len(content)} characters")
        else:
            # File or URL reference - validate but don't fetch
            if source.startswith("http://") or source.startswith("https://"):
                # URL reference
                add_attachment(alias, source=source)
                console.print(f"[green]✓ URL attachment '{alias}' saved[/green]")
                console.print(f"  Source: {source}")
                console.print("  Type: URL (fetched on demand)")
            else:
                # File reference - validate it exists
                file_path = Path(source).expanduser()
                if not file_path.exists():
                    console.print(f"[red]File not found: {source}[/red]")
                    raise typer.Exit(1)

                # Store absolute path for reliability
                absolute_path = str(file_path.resolve())
                add_attachment(alias, source=absolute_path)

                console.print(f"[green]✓ File attachment '{alias}' saved[/green]")
                console.print(f"  Source: {absolute_path}")
                console.print("  Type: File (read on demand)")

    except Exception as e:
        console.print(f"[red]Failed to add attachment: {e}[/red]")
        raise typer.Exit(1)


@attachments_app.command("list")
def attachments_list():
    """List all attachments."""
    from rich.table import Table

    from tsugite.attachments import list_attachments

    attachments = list_attachments()

    if not attachments:
        console.print("[yellow]No attachments found[/yellow]")
        console.print("\nAdd an attachment with: [cyan]tsugite attachments add NAME SOURCE[/cyan]")
        return

    table = Table(title=f"Attachments ({len(attachments)} total)")
    table.add_column("Alias", style="cyan")
    table.add_column("Source", style="dim")
    table.add_column("Size", justify="right")
    table.add_column("Updated", style="dim")

    for alias, data in sorted(attachments.items()):
        size = len(data.get("content", ""))
        updated = data.get("updated", "unknown")[:10]  # Just date part
        table.add_row(alias, data.get("source", "unknown"), f"{size:,}", updated)

    console.print(table)


@attachments_app.command("show")
def attachments_show(
    alias: str = typer.Argument(help="Attachment alias to show"),
    content: bool = typer.Option(False, "--content", help="Show full content"),
):
    """Show details of an attachment."""
    from rich.panel import Panel

    from tsugite.attachments import get_attachment
    from tsugite.cache import get_cache_key, list_cache

    result = get_attachment(alias)

    if result is None:
        console.print(f"[red]Attachment '{alias}' not found[/red]")
        raise typer.Exit(1)

    source, stored_content = result

    # Determine attachment type
    is_inline = source.lower() in ("inline", "text")

    # Build panel content
    panel_content = f"[cyan]Alias:[/cyan] {alias}\n"
    panel_content += f"[cyan]Type:[/cyan] {'Inline' if is_inline else 'Reference'}\n"
    panel_content += f"[cyan]Source:[/cyan] {source}\n"

    # Check cache status for references
    if not is_inline:
        cache_entries = list_cache()
        cache_key = get_cache_key(source)
        if cache_key in cache_entries:
            cache_info = cache_entries[cache_key]
            panel_content += f"[cyan]Cached:[/cyan] Yes (size: {cache_info['size']:,} bytes)\n"
            panel_content += f"[cyan]Cached at:[/cyan] {cache_info['cached_at']}\n"
        else:
            panel_content += "[cyan]Cached:[/cyan] No\n"

    # Show content or preview
    if is_inline and stored_content:
        panel_content += f"[cyan]Size:[/cyan] {len(stored_content):,} characters\n"
        if content:
            panel_content += f"\n[cyan]Content:[/cyan]\n{stored_content}"
        else:
            preview = stored_content[:200]
            if len(stored_content) > 200:
                preview += "..."
            panel_content += f"\n[cyan]Preview:[/cyan]\n{preview}"
            panel_content += "\n\n[dim]Use --content to show full content[/dim]"
    elif not is_inline and content:
        # For references, fetch and show content if requested
        from tsugite.utils import resolve_attachments

        resolved = resolve_attachments([alias], Path.cwd())
        if resolved:
            _, resolved_content = resolved[0]
            panel_content += f"[cyan]Size:[/cyan] {len(resolved_content):,} characters\n"
            panel_content += f"\n[cyan]Content:[/cyan]\n{resolved_content}"
    elif not is_inline:
        panel_content += "\n[dim]Use --content to fetch and show full content[/dim]"

    console.print(Panel(panel_content, title=f"Attachment: {alias}", border_style="blue"))


@attachments_app.command("remove")
def attachments_remove(
    alias: str = typer.Argument(help="Attachment alias to remove"),
):
    """Remove an attachment."""
    from tsugite.attachments import remove_attachment

    if remove_attachment(alias):
        console.print(f"[green]✓ Attachment '{alias}' removed[/green]")
    else:
        console.print(f"[yellow]Attachment '{alias}' not found[/yellow]")


@attachments_app.command("search")
def attachments_search(
    query: str = typer.Argument(help="Search term"),
):
    """Search attachments by alias or source."""
    from rich.table import Table

    from tsugite.attachments import search_attachments

    results = search_attachments(query)

    if not results:
        console.print(f"[yellow]No attachments found matching '{query}'[/yellow]")
        return

    table = Table(title=f"Search Results for '{query}' ({len(results)} found)")
    table.add_column("Alias", style="cyan")
    table.add_column("Source", style="dim")
    table.add_column("Size", justify="right")

    for alias, data in sorted(results.items()):
        size = len(data.get("content", ""))
        table.add_row(alias, data.get("source", "unknown"), f"{size:,}")

    console.print(table)
