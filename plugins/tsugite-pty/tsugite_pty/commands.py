"""Daemon slash command contributed by the PTY plugin.

Registered via this package's `tsugite.commands` entry point (see pyproject).
Importing the module runs the `@adapter_command` decorator, landing `/terminals`
in the daemon's shared command registry alongside the built-ins. Handlers get a
daemon adapter as their first arg, so this command reads the terminal store the
daemon wired onto the adapter - the same store `/run` spawns into.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from tsugite_daemon.commands import CommandParam, adapter_command

if TYPE_CHECKING:
    from tsugite_daemon.adapters.base import BaseAdapter


@adapter_command(
    name="terminals",
    description="List the daemon's terminal sessions and their state",
    params=[
        CommandParam(
            "state",
            str,
            "Filter by state (starting, running, succeeded, failed, cancelled, stream_lost)",
            required=False,
        )
    ],
)
async def cmd_terminals(adapter: BaseAdapter, state: str | None = None) -> str:
    """List the PTY terminals the daemon owns (id, command, state)."""
    terminal_store = getattr(adapter, "terminal_store", None)
    if terminal_store is None:
        return "Terminal sessions require the daemon terminal runtime to be enabled."

    terminals = terminal_store.list_all()
    if state:
        terminals = [t for t in terminals if t.state == state]
    if not terminals:
        return "No terminals found."

    lines = [f"[{t.state}] {t.id}: {t.cmd}" for t in terminals[:10]]
    if len(terminals) > 10:
        lines.append(f"... and {len(terminals) - 10} more")
    return "\n".join(lines)
