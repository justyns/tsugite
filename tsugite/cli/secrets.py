"""CLI commands for secret management."""

import typer
from rich.console import Console

secrets_app = typer.Typer(help="Manage secrets")
console = Console()


@secrets_app.command("list")
def secrets_list():
    """List available secret names (not values)."""
    from tsugite.secrets import get_backend

    names = get_backend().list_names()
    if not names:
        console.print("[dim]No secrets found (or backend does not support listing)[/dim]")
        return
    for name in sorted(names):
        console.print(f"  {name}")


@secrets_app.command("set")
def secrets_set(
    name: str = typer.Argument(help="Secret name"),
    value: str = typer.Option(None, prompt="Secret value", hide_input=True, help="Secret value (prompted if omitted)"),
):
    """Set a secret value (only for backends that support writing)."""
    from tsugite.secrets import get_backend

    try:
        get_backend().set(name, value)
        console.print(f"[green]Set secret '{name}'[/green]")
    except NotImplementedError:
        console.print("[red]This secrets backend does not support writing[/red]")
        raise typer.Exit(1)


@secrets_app.command("delete")
def secrets_delete(name: str = typer.Argument(help="Secret name")):
    """Delete a secret."""
    from tsugite.secrets import get_backend

    try:
        if get_backend().delete(name):
            console.print(f"[green]Deleted secret '{name}'[/green]")
        else:
            console.print(f"[yellow]Secret '{name}' not found[/yellow]")
    except NotImplementedError:
        console.print("[red]This secrets backend does not support deletion[/red]")
        raise typer.Exit(1)
