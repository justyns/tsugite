"""Scheduler adapter — bridges the Scheduler into the daemon Gateway."""

import asyncio
import logging
from contextlib import nullcontext
from pathlib import Path

from tsugite.daemon.adapters.base import BaseAdapter, ChannelContext
from tsugite.daemon.config import NotificationChannelConfig
from tsugite.daemon.scheduler import ScheduleEntry, Scheduler
from tsugite.tools.notify import notify_context, send_notification

logger = logging.getLogger(__name__)


class SchedulerAdapter:
    """Integrates the Scheduler with the daemon, executing agents via existing adapters."""

    def __init__(
        self,
        adapters: dict[str, BaseAdapter],
        schedules_path: Path,
        notification_channels: dict[str, NotificationChannelConfig] | None = None,
    ):
        self._adapters = adapters
        self._notification_channels = notification_channels or {}
        self.scheduler = Scheduler(schedules_path, self._run_agent)

    async def start(self):
        await self.scheduler.start()

    async def stop(self):
        await self.scheduler.stop()

    def _resolve_channels(self, channel_names: list[str]) -> list[tuple[str, NotificationChannelConfig]]:
        """Resolve channel names to (name, config) tuples."""
        resolved = []
        for name in channel_names:
            config = self._notification_channels.get(name)
            if config:
                resolved.append((name, config))
            else:
                logger.warning("Notification channel '%s' not found in config, skipping", name)
        return resolved

    async def _run_agent(self, entry: ScheduleEntry) -> str:
        adapter = self._adapters.get(entry.agent)
        if not adapter:
            raise ValueError(f"No adapter found for agent '{entry.agent}'")
        logger.info("Schedule '%s' executing agent '%s': %s", entry.id, entry.agent, entry.prompt[:100])

        user_id = f"scheduler:{entry.agent}"
        metadata = {"schedule_id": entry.id}
        if entry.notify_tool:
            metadata["notify_tool"] = True

        channel_context = ChannelContext(
            source="scheduler",
            channel_id=None,
            user_id=user_id,
            reply_to=user_id,
            metadata=metadata,
        )

        resolved_channels = self._resolve_channels(entry.notify) if entry.notify else []

        from tsugite.interaction import NonInteractiveBackend, set_interaction_backend

        set_interaction_backend(NonInteractiveBackend())

        ctx = notify_context(resolved_channels) if (entry.notify_tool and resolved_channels) else nullcontext()
        with ctx:
            result = await adapter.handle_message(
                user_id=user_id,
                message=entry.prompt,
                channel_context=channel_context,
            )

        logger.info("Schedule '%s' agent '%s' completed", entry.id, entry.agent)

        if resolved_channels:
            try:
                notification = f"**Schedule `{entry.id}` completed:**\n\n{result[:4000]}"
                await asyncio.to_thread(send_notification, notification, resolved_channels)
            except Exception as e:
                logger.error("Auto-notify for schedule '%s' failed: %s", entry.id, e)

        return result
