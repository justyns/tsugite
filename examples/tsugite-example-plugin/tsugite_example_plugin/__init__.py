"""Example tsugite plugin — provides tools and a stub adapter.

Demonstrates how a single package can register multiple plugin types
by declaring entry points in different groups.
"""

import random


def dice_roll(sides: int = 6) -> str:
    """Roll a die with the specified number of sides."""
    result = random.randint(1, sides)
    return f"Rolled a d{sides}: {result}"


def coin_flip() -> str:
    """Flip a coin and return heads or tails."""
    return random.choice(["Heads", "Tails"])


def register_tools(config: dict | None = None) -> list:
    """Entry point for tsugite.tools group. Returns tool functions."""
    return [dice_roll, coin_flip]


def create_adapter(config, agents_config, session_store, identity_map):
    """Entry point for tsugite.adapters group. Returns a BaseAdapter instance.

    This is a stub — a real adapter would subclass BaseAdapter and implement
    start()/stop() for a chat platform like Slack, Matrix, etc.
    """
    raise NotImplementedError("This is a stub adapter for demonstration only")
