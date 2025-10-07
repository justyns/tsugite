"""Cache management CLI commands."""

from typing import Optional

import typer
from rich.console import Console

console = Console()

cache_app = typer.Typer(help="Manage attachment cache")


@cache_app.command("clear")
def cache_clear(
    alias: Optional[str] = typer.Argument(None, help="Attachment alias to clear cache for (or all if omitted)"),
):
    """Clear cache for an attachment or entire cache."""
    from tsugite.attachments import get_attachment
    from tsugite.cache import clear_cache

    try:
        if alias:
            # Get attachment source
            result = get_attachment(alias)
            if result is None:
                console.print(f"[red]Attachment '{alias}' not found[/red]")
                raise typer.Exit(1)

            source, content = result
            if content is not None:
                console.print(f"[yellow]Attachment '{alias}' is inline (no cache)[/yellow]")
                return

            # Clear cache for this source
            count = clear_cache(source)
            if count > 0:
                console.print(f"[green]✓ Cache cleared for '{alias}'[/green]")
            else:
                console.print(f"[yellow]No cache found for '{alias}'[/yellow]")
        else:
            # Clear all cache
            count = clear_cache()
            console.print(f"[green]✓ Cleared {count} cache entries[/green]")

    except Exception as e:
        console.print(f"[red]Failed to clear cache: {e}[/red]")
        raise typer.Exit(1)


@cache_app.command("list")
def cache_list():
    """List all cached attachments."""
    from rich.table import Table

    from tsugite.cache import list_cache

    cache_entries = list_cache()

    if not cache_entries:
        console.print("[yellow]No cached entries found[/yellow]")
        return

    table = Table(title=f"Cached Attachments ({len(cache_entries)} total)")
    table.add_column("Source", style="cyan")
    table.add_column("Cached At", style="dim")
    table.add_column("Size", justify="right")

    for key, metadata in sorted(cache_entries.items(), key=lambda x: x[1].get("cached_at", "")):
        source = metadata.get("source", "unknown")
        cached_at = metadata.get("cached_at", "unknown")[:19]  # Just date + time
        size = metadata.get("size", 0)
        table.add_row(source, cached_at, f"{size:,} bytes")

    console.print(table)


@cache_app.command("info")
def cache_info(
    alias: str = typer.Argument(help="Attachment alias"),
):
    """Show cache info for an attachment."""
    from rich.panel import Panel

    from tsugite.attachments import get_attachment
    from tsugite.cache import get_cache_info

    result = get_attachment(alias)
    if result is None:
        console.print(f"[red]Attachment '{alias}' not found[/red]")
        raise typer.Exit(1)

    source, content = result

    if content is not None:
        console.print(f"[yellow]Attachment '{alias}' is inline (no cache)[/yellow]")
        return

    cache_metadata = get_cache_info(source)

    if cache_metadata:
        cached_at = cache_metadata.get("cached_at", "unknown")
        size = cache_metadata.get("size", 0)

        panel_content = (
            f"[cyan]Source:[/cyan] {source}\n"
            f"[cyan]Cached:[/cyan] Yes\n"
            f"[cyan]Cached At:[/cyan] {cached_at}\n"
            f"[cyan]Size:[/cyan] {size:,} bytes"
        )
        console.print(Panel(panel_content, title=f"Cache Info: {alias}", border_style="green"))
    else:
        panel_content = f"[cyan]Source:[/cyan] {source}\n[yellow]Cached:[/yellow] No (will be fetched on first use)"
        console.print(Panel(panel_content, title=f"Cache Info: {alias}", border_style="yellow"))
