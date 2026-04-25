"""Tsugite CLI application - main entry point."""

import typer
from rich.console import Console
from rich.traceback import install

# Install rich traceback handler for better error messages
install(show_locals=False, width=None, word_wrap=True)

app = typer.Typer(
    name="tsugite",
    help="Micro-agent runner for task automation using markdown definitions",
    no_args_is_help=True,
)

# Global console for CLI messages (version, help, errors) - uses stdout
console = Console()


@app.command()
def version():
    """Show version information."""
    from tsugite import __version__

    console.print(f"Tsugite version {__version__}")


# Register main commands from split modules
from .chat import chat  # noqa: E402
from .render import render  # noqa: E402
from .run import run  # noqa: E402

app.command()(run)
app.command()(render)
app.command()(chat)

# Backward-compatible re-exports + subcommand registration
from .agents import agents_app  # noqa: E402
from .attachments import attachments_app  # noqa: E402
from .cache import cache_app  # noqa: E402
from .chat import DEFAULT_MAX_CHAT_HISTORY  # noqa: E402, F401
from .config import config_app  # noqa: E402
from .daemon import daemon_app  # noqa: E402
from .history import history_app  # noqa: E402
from .init import init  # noqa: E402
from .models import models_app  # noqa: E402
from .plugins import plugin_app  # noqa: E402
from .run import _unpack_execution_result  # noqa: E402, F401
from .secrets import secrets_app  # noqa: E402
from .skills import skills_app  # noqa: E402
from .tools import tools_app  # noqa: E402
from .usage import usage_app  # noqa: E402
from .validate import validate_command  # noqa: E402
from .workspace import workspace_app  # noqa: E402

app.add_typer(agents_app, name="agents")
app.add_typer(config_app, name="config")
app.add_typer(daemon_app, name="daemon")
app.add_typer(attachments_app, name="attachments")
app.add_typer(cache_app, name="cache")
app.add_typer(tools_app, name="tools")
app.add_typer(plugin_app, name="plugin")
app.add_typer(history_app, name="history")
app.add_typer(workspace_app, name="workspace")
app.add_typer(models_app, name="models")
app.add_typer(usage_app, name="usage")
app.add_typer(secrets_app, name="secrets")
app.add_typer(skills_app, name="skill")
app.command("init")(init)
app.command("validate")(validate_command)
