"""Example tsugite plugin: tools, an event subscriber, and a stub adapter.

Demonstrates the unified `tsugite.plugins` entry point: one declaration in
pyproject.toml imports this module and the @tool / @subscribe decorators
register everything. The adapter still uses the dedicated `tsugite.adapters`
group because adapter factories take constructor kwargs (config, agents_config,
session_store, identity_map) rather than running on import.
"""

import random

from tsugite.events.bus import subscribe
from tsugite.tools import tool


@tool(category="example")
def dice_roll(sides: int = 6) -> str:
    """Roll a die with the specified number of sides."""
    result = random.randint(1, sides)
    return f"Rolled a d{sides}: {result}"


@tool(category="example")
def coin_flip() -> str:
    """Flip a coin and return heads or tails."""
    return random.choice(["Heads", "Tails"])


@subscribe(event_name="tool_call")
def on_tool_call(event):
    """Print every tool invocation."""
    print(f"[example-plugin] tool_call: {event.tool_name}")


def create_adapter(config, agents_config, session_store, identity_map):
    """Entry point for tsugite.adapters group. Returns a BaseAdapter instance.

    This is a stub - a real adapter would subclass BaseAdapter and implement
    start()/stop() for a chat platform like Slack, Matrix, etc.
    """
    raise NotImplementedError("This is a stub adapter for demonstration only")
