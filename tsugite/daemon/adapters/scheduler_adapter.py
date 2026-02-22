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

_MAX_RESULT_CHARS = 4000


class SchedulerAdapter:
    """Integrates the Scheduler with the daemon, executing agents via existing adapters."""

    def __init__(
        self,
        adapters: dict[str, BaseAdapter],
        schedules_path: Path,
        notification_channels: dict[str, NotificationChannelConfig] | None = None,
        identity_map: dict[str, str] | None = None,
    ):
        self._adapters = adapters
        self._notification_channels = notification_channels or {}
        self._identity_map = identity_map or {}
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
            truncated = result[:_MAX_RESULT_CHARS]
            try:
                notification = f"**Schedule `{entry.id}` completed:**\n\n{truncated}"
                await asyncio.to_thread(send_notification, notification, resolved_channels)
            except Exception as e:
                logger.error("Auto-notify for schedule '%s' failed: %s", entry.id, e)

            if entry.inject_history:
                await self._inject_into_user_sessions(adapter, entry, truncated, resolved_channels)

        return result

    async def _inject_into_user_sessions(
        self,
        adapter: BaseAdapter,
        entry: ScheduleEntry,
        truncated_result: str,
        resolved_channels: list[tuple[str, NotificationChannelConfig]],
    ) -> None:
        """Inject a synthetic turn into each notified user's main chat session."""
        for _name, config in resolved_channels:
            if config.type != "discord":
                continue

            canonical = self._identity_map.get(f"discord:{config.user_id}", config.user_id)

            try:
                await asyncio.to_thread(self._record_synthetic_turn, adapter, canonical, entry, truncated_result)
            except Exception as e:
                logger.error("Failed to inject history for schedule '%s' user '%s': %s", entry.id, canonical, e)

    @staticmethod
    def _record_synthetic_turn(adapter: BaseAdapter, user_id: str, entry: ScheduleEntry, result: str) -> None:
        """Write a synthetic turn into the user's session JSONL."""
        from tsugite.history import SessionStorage, get_history_dir

        session_id = adapter.session_manager.get_or_create_session(user_id)
        session_path = get_history_dir() / f"{session_id}.jsonl"
        storage = SessionStorage.get_or_create(
            session_id, adapter.agent_name, adapter.resolve_model(), session_path=session_path
        )

        messages = [
            {
                "role": "user",
                "content": (
                    f'<scheduled_task id="{entry.id}">\n'
                    "This task ran in the background and the result "
                    "was sent as a notification to the user.\n"
                    "</scheduled_task>"
                ),
            },
            {"role": "assistant", "content": result},
        ]

        storage.record_turn(
            messages=messages,
            final_answer=result,
            metadata={"synthetic": True, "schedule_id": entry.id},
        )
