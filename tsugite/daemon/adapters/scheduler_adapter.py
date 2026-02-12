"""Scheduler adapter — bridges the Scheduler into the daemon Gateway."""

import logging
from pathlib import Path

from tsugite.daemon.adapters.base import BaseAdapter, ChannelContext
from tsugite.daemon.scheduler import Scheduler

logger = logging.getLogger(__name__)


class SchedulerAdapter:
    """Integrates the Scheduler with the daemon, executing agents via existing adapters."""

    def __init__(self, adapters: dict[str, BaseAdapter], schedules_path: Path):
        self._adapters = adapters
        self.scheduler = Scheduler(schedules_path, self._run_agent)

    async def start(self):
        await self.scheduler.start()

    async def stop(self):
        await self.scheduler.stop()

    async def _run_agent(self, agent_name: str, prompt: str) -> str:
        adapter = self._adapters.get(agent_name)
        if not adapter:
            raise ValueError(f"No adapter found for agent '{agent_name}'")

        user_id = f"scheduler:{agent_name}"
        channel_context = ChannelContext(
            source="scheduler",
            channel_id=None,
            user_id=user_id,
            reply_to=user_id,
        )

        return await adapter.handle_message(
            user_id=user_id,
            message=prompt,
            channel_context=channel_context,
        )
