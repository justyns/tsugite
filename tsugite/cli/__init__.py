"""Tsugite CLI application - main entry point."""

import importlib
from typing import Any, Optional

import typer
from typer.core import TyperCommand, TyperGroup

# name -> (module, attribute, help line). The help text lives here so the root
# help renders without importing any command module.
_LAZY_COMMANDS: dict[str, tuple[str, str, str]] = {
    "run": ("tsugite.cli.run", "run", "Run an agent with the given prompt."),
    "render": ("tsugite.cli.render", "render", "Render an agent template without executing it."),
    "chat": ("tsugite.cli.chat", "chat", "Start an interactive chat session with an agent."),
    "exec": ("tsugite.cli.exec", "exec_cmd", "Run a Python snippet in tsugite's tool namespace."),
    "init": ("tsugite.cli.init", "init", "Initialize tsugite with global configuration."),
    "validate": (
        "tsugite.cli.validate",
        "validate_command",
        "Validate agent frontmatter configuration against the schema.",
    ),
    "agents": ("tsugite.cli.agents", "agents_app", "Manage agents and agent inheritance"),
    "config": ("tsugite.cli.config", "config_app", "Manage Tsugite configuration"),
    "daemon": ("tsugite.cli.daemon", "daemon_app", "Daemon management commands"),
    "attachments": ("tsugite.cli.attachments", "attachments_app", "Manage reusable text attachments"),
    "cache": ("tsugite.cli.cache", "cache_app", "Manage attachment cache"),
    "tools": ("tsugite.cli.tools", "tools_app", "Manage and inspect available tools"),
    "plugin": ("tsugite.cli.plugins", "plugin_app", "Manage and inspect installed plugins"),
    "history": ("tsugite.cli.history", "history_app", "Manage conversation history"),
    "workspace": ("tsugite.cli.workspace", "workspace_app", "Manage workspaces"),
    "models": ("tsugite.cli.models", "models_app", "List and manage available models."),
    "usage": ("tsugite.cli.usage", "usage_app", "View token usage and cost analytics."),
    "secrets": ("tsugite.cli.secrets", "secrets_app", "Manage secrets"),
    "skill": ("tsugite.cli.skills", "skills_app", "Inspect and validate skills"),
}


def _import_command(name: str) -> TyperGroup | TyperCommand:
    module, attribute, _ = _LAZY_COMMANDS[name]
    target = getattr(importlib.import_module(module), attribute)
    if isinstance(target, typer.Typer):
        return typer.main.get_group(target)
    holder = typer.Typer(add_completion=False)
    holder.command(name)(target)
    return typer.main.get_command(holder)


class _LazyCommand(TyperCommand):
    """Stand-in with just enough to render a help row.

    Subclasses typer's command, not click's: typer 0.26+ vendors its own click,
    so a real-click subclass is foreign to the group typer builds.

    `make_context` returns the real command's context, so the group's `invoke`
    reads `sub_ctx.command` and dispatches to it.
    """

    def make_context(
        self,
        info_name: Optional[str],
        args: list[str],
        parent: Any = None,
        **extra: Any,
    ) -> Any:
        real = _import_command(self.name or "")
        return real.make_context(info_name, args, parent=parent, **extra)


class LazyCommandGroup(TyperGroup):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        for name, (_module, _attribute, help_text) in _LAZY_COMMANDS.items():
            self.commands[name] = _LazyCommand(name, help=help_text)

    def invoke(self, ctx: Any) -> Any:
        # Only a command run needs the rich excepthook, and installing it pulls in rich.console.
        from rich.traceback import install

        install(show_locals=False, width=None, word_wrap=True)
        return super().invoke(ctx)


app = typer.Typer(
    name="tsugite",
    help="Micro-agent runner for task automation using markdown definitions",
    no_args_is_help=True,
    cls=LazyCommandGroup,
)


def __getattr__(name: str) -> Any:
    if name != "console":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from rich.console import Console

    globals()["console"] = Console()
    return globals()["console"]


def _print_version() -> None:
    from tsugite import __version__

    typer.echo(f"Tsugite version {__version__}")


def _version_callback(value: bool) -> None:
    if not value:
        return
    _print_version()
    raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        "-V",
        callback=_version_callback,
        is_eager=True,
        help="Show version information.",
    ),
) -> None:
    pass


@app.command()
def version():
    """Show version information."""
    _print_version()
