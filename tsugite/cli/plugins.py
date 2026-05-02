"""Plugin management CLI commands."""

import re
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

console = Console()

plugin_app = typer.Typer(help="Manage and inspect installed plugins")

_PLUGIN_NAME_RE = re.compile(r"^[a-z][a-z0-9_-]*$")


@plugin_app.command("list")
def plugin_list(
    group: Optional[str] = typer.Option(None, "--group", "-g", help="Filter by group (tools, adapters)"),
):
    """List all discovered plugins."""
    from tsugite.plugins import discover_plugins

    plugins = discover_plugins()

    if group:
        full_group = f"tsugite.{group}" if not group.startswith("tsugite.") else group
        plugins = [p for p in plugins if p.group == full_group]

    if not plugins:
        console.print("[dim]No plugins discovered.[/dim]")
        console.print("\nInstall a plugin package (e.g. [cyan]pip install tsugite-weather[/cyan]) to get started.")
        return

    table = Table(show_header=True, header_style="bold cyan", border_style="dim")
    table.add_column("Name", style="bold")
    table.add_column("Type")
    table.add_column("Entry Point", style="dim")
    table.add_column("Status")

    for p in sorted(plugins, key=lambda x: (x.group, x.name)):
        plugin_type = p.group.removeprefix("tsugite.")
        if not p.enabled:
            status = "[yellow]disabled[/yellow]"
        elif p.error:
            status = f"[red]error: {p.error}[/red]"
        elif p.loaded:
            status = "[green]loaded[/green]"
        else:
            status = "[dim]discovered[/dim]"
        table.add_row(p.name, plugin_type, p.entry_point, status)

    console.print(table)
    console.print(f"\n[dim]{len(plugins)} plugin(s)[/dim]")


@plugin_app.command("info")
def plugin_info(
    name: str = typer.Argument(help="Plugin name to inspect"),
):
    """Show detailed information about a plugin."""
    from tsugite.plugins import discover_plugins

    plugins = discover_plugins()
    matches = [p for p in plugins if p.name == name]

    if not matches:
        console.print(f"[red]Plugin '{name}' not found[/red]")
        available = [p.name for p in plugins]
        if available:
            console.print(f"\nAvailable: {', '.join(sorted(available))}")
        raise typer.Exit(1)

    for p in matches:
        console.print(f"\n[bold cyan]{p.name}[/bold cyan]")
        console.print(f"  Group:       {p.group}")
        console.print(f"  Entry point: {p.entry_point}")
        console.print(f"  Enabled:     {'yes' if p.enabled else 'no'}")
        if p.error:
            console.print(f"  Error:       [red]{p.error}[/red]")


def _repo_root() -> Path:
    """Locate the tsugite workspace root by walking up from cwd looking for plugins/."""
    cwd = Path.cwd().resolve()
    for d in (cwd, *cwd.parents):
        if (d / "plugins").is_dir() and (d / "pyproject.toml").is_file():
            return d
    raise typer.BadParameter(
        "Could not find a tsugite workspace root (looking for plugins/ + pyproject.toml). Run from inside the repo."
    )


def _read_lockstep_version(root_pyproject: Path) -> str:
    text = root_pyproject.read_text()
    match = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
    if not match:
        raise typer.BadParameter(f"Could not read version from {root_pyproject}")
    return match.group(1)


_PLUGIN_PYPROJECT_TEMPLATE = """\
[project]
name = "tsugite-{name}"
version = "{version}"
description = "Tsugite plugin: {name}"
requires-python = ">=3.11"
dependencies = ["tsugite-cli=={version}"]

[project.entry-points."tsugite.plugins"]
{name} = "tsugite_{module}"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.uv.sources]
tsugite-cli = {{ workspace = true }}
"""

_PLUGIN_MODULE_TEMPLATE = '''\
"""Tsugite plugin: {name}.

Tools defined here auto-register via the @tool decorator at import time;
the entry point in pyproject.toml is module-only (no :function suffix).
"""

from tsugite.tools import tool


@tool(category="{name}")
def {module}_hello(message: str = "world") -> str:
    """Example tool. Replace with your real tools.

    Args:
        message: Greeting target.
    """
    return f"Hello, {{message}}, from the {name} plugin!"
'''

_PLUGIN_TEST_TEMPLATE = '''\
"""Tests for the tsugite-{name} plugin."""

from tsugite_{module} import {module}_hello


def test_hello_default():
    assert {module}_hello() == "Hello, world, from the {name} plugin!"


def test_hello_custom():
    assert {module}_hello(message="tsugite") == "Hello, tsugite, from the {name} plugin!"
'''


def _patch_root_pyproject(root_pyproject: Path, dist_name: str) -> None:
    text = root_pyproject.read_text()
    sources_line = f'{dist_name} = {{ workspace = true }}\n'
    dev_dep_line = f'  "{dist_name}",\n'

    if dist_name in text:
        return  # idempotent

    new_text = re.sub(
        r'(\[tool\.uv\.sources\][^\[]*?tsugite-cli = \{ workspace = true \}\n)',
        rf'\1{sources_line}',
        text,
        count=1,
    )
    if new_text == text:
        raise typer.BadParameter("Could not locate [tool.uv.sources] block in root pyproject.toml")

    new_text2 = re.sub(
        r'(\[dependency-groups\]\s*\ndev\s*=\s*\[(?:[^\]]|\n)*?)\n\]',
        rf'\1\n{dev_dep_line.rstrip()}\n]',
        new_text,
        count=1,
    )
    if new_text2 == new_text:
        raise typer.BadParameter("Could not locate [dependency-groups] dev list in root pyproject.toml")

    root_pyproject.write_text(new_text2)


@plugin_app.command("create")
def plugin_create(
    name: str = typer.Argument(help="Plugin short name (lowercase, alphanumeric + hyphens). E.g. 'discord' creates tsugite-discord."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print planned changes without writing"),
):
    """Scaffold a new workspace plugin under plugins/tsugite-<name>/."""
    if not _PLUGIN_NAME_RE.match(name):
        raise typer.BadParameter(
            f"'{name}' is not a valid plugin name. Use lowercase letters, digits, hyphens, underscores; must start with a letter."
        )

    root = _repo_root()
    plugin_dir = root / "plugins" / f"tsugite-{name}"
    if plugin_dir.exists():
        raise typer.BadParameter(f"{plugin_dir} already exists")

    module = name.replace("-", "_")
    version = _read_lockstep_version(root / "pyproject.toml")

    pyproject_body = _PLUGIN_PYPROJECT_TEMPLATE.format(name=name, module=module, version=version)
    module_body = _PLUGIN_MODULE_TEMPLATE.format(name=name, module=module)
    test_body = _PLUGIN_TEST_TEMPLATE.format(name=name, module=module)

    pyproject_path = plugin_dir / "pyproject.toml"
    module_path = plugin_dir / f"tsugite_{module}" / "__init__.py"
    test_path = plugin_dir / "tests" / f"test_{module}.py"

    plan = [
        (pyproject_path, pyproject_body),
        (module_path, module_body),
        (test_path, test_body),
    ]

    console.print(f"[bold]Creating tsugite-{name}{' (dry run)' if dry_run else ''}[/bold]")
    for path, _ in plan:
        rel = path.relative_to(root)
        console.print(f"  {'would write' if dry_run else 'write'}: {rel}")
    console.print(f"  {'would patch' if dry_run else 'patch'}: pyproject.toml ([tool.uv.sources] + dev group)")

    if dry_run:
        return

    for path, body in plan:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body)
    _patch_root_pyproject(root / "pyproject.toml", f"tsugite-{name}")

    console.print(f"\n[green]Created {plugin_dir.relative_to(root)}[/green]")
    console.print("\nNext steps:")
    console.print("  uv sync --all-extras                 # install the new plugin into the workspace venv")
    console.print(f"  uv run pytest plugins/tsugite-{name}/tests/    # confirm the example tool passes")
    console.print(f"  uv run tsu tools list | grep {module}_hello  # verify it surfaces in the registry")
