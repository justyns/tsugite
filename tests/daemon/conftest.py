"""Shared fixtures for daemon tests."""

import asyncio

# Shared Jobs fixtures (FakeRunner/store/orchestrator) live in
# test_jobs_orchestrator.py; re-export so sibling job test modules can use them
# without fixture-shadowing imports.
from .test_jobs_orchestrator import event_bus, orchestrator, runner, store  # noqa: F401, E402


async def _wait_until(predicate, timeout: float = 2.0) -> bool:
    """Returns False if `predicate` is still false at the deadline."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not predicate():
        if loop.time() > deadline:
            return False
        await asyncio.sleep(0.01)
    return True
