"""CLI commands for model management."""

from __future__ import annotations

from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

models_app = typer.Typer(help="List and manage available models.")
console = Console()


def _fmt_tokens(n: int | None) -> str:
    if n is None:
        return "-"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    return f"{n // 1000}K"


def _fmt_cost(c: float | None) -> str:
    if c is None:
        return "-"
    return f"${c:.2f}"


def _fmt_flag(v: bool) -> str:
    return "y" if v else ""


@models_app.command("list")
def list_models(
    provider: Optional[str] = typer.Option(None, "--provider", "-p", help="Filter to a specific provider"),
    refresh: bool = typer.Option(False, "--refresh", "-r", help="Refresh cached model list"),
) -> None:
    """List available models from all providers."""
    import asyncio

    from tsugite.providers.model_cache import get_all_models, get_provider_models

    if provider:
        models_by_provider = asyncio.run(
            get_all_models(providers=[provider], refresh=refresh)
        )
    else:
        models_by_provider = asyncio.run(get_all_models(refresh=refresh))

    table = Table(show_header=True, header_style="bold cyan", padding=(0, 1))
    table.add_column("Provider")
    table.add_column("Model")
    table.add_column("Context", justify="right")
    table.add_column("Vision")
    table.add_column("Reasoning")
    table.add_column("In $/1M", justify="right")
    table.add_column("Out $/1M", justify="right")

    total = 0
    for prov_name in sorted(models_by_provider.keys()):
        models = models_by_provider[prov_name]
        if not models:
            continue
        for m in sorted(models, key=lambda x: x["name"]):
            info = m.get("info")
            total += 1
            table.add_row(
                prov_name,
                m["name"],
                _fmt_tokens(info.max_input_tokens if info else None),
                _fmt_flag(info.supports_vision if info else False),
                _fmt_flag(info.supports_reasoning if info else False),
                _fmt_cost(info.input_cost_per_million if info else None),
                _fmt_cost(info.output_cost_per_million if info else None),
            )

    if total == 0:
        console.print("[yellow]No models found. Check your API keys and provider configuration.[/yellow]")
        return

    console.print(table)
    console.print(f"\n[dim]{total} models across {len([v for v in models_by_provider.values() if v])} providers[/dim]")
