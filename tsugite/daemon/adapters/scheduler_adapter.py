"""Scheduler adapter — bridges the Scheduler into the daemon Gateway."""

import asyncio
import logging
import subprocess
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path

from tsugite.agent_runner.models import AgentSkippedError
from tsugite.daemon.adapters.base import BaseAdapter, ChannelContext
from tsugite.daemon.auth import TokenStore
from tsugite.daemon.config import NotificationChannelConfig
from tsugite.daemon.scheduler import RunResult, ScheduleEntry, Scheduler
from tsugite.daemon.session_store import Session, SessionSource, SessionStatus
from tsugite.exceptions import AgentExecutionError
from tsugite.tools.notify import notify_context, send_notification

logger = logging.getLogger(__name__)

_MAX_RESULT_CHARS = 4000


MAX_CHAIN_DEPTH = 5


class SchedulerAdapter:
    """Integrates the Scheduler with the daemon, executing agents via existing adapters."""

    def __init__(
        self,
        adapters: dict[str, BaseAdapter],
        schedules_path: Path,
        notification_channels: dict[str, NotificationChannelConfig] | None = None,
        identity_map: dict[str, str] | None = None,
        token_store: TokenStore | None = None,
        tsugite_api_url: str = "",
    ):
        self._adapters = adapters
        self._notification_channels = notification_channels or {}
        self._identity_map = identity_map or {}
        self._token_store = token_store
        self._tsugite_api_url = tsugite_api_url
        self._session_runner = None
        self.scheduler = Scheduler(schedules_path, self._run_agent, script_callback=self._run_script)

    def set_session_runner(self, session_runner) -> None:
        """Set the SessionRunner reference (called after both are constructed)."""
        self._session_runner = session_runner

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

    def _resolve_canonical_user(self, config: NotificationChannelConfig) -> str:
        """Resolve a notification channel's user to their canonical identity."""
        return self._identity_map.get(f"discord:{config.user_id}", config.user_id)

    def _create_run_session(self, entry: ScheduleEntry) -> str:
        """Create a Session record for a schedule run. Returns the conv_id."""
        if entry.session_id:
            conv_id = f"sched_{entry.session_id}"
        else:
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            safe_id = entry.id.replace(":", "_")
            conv_id = f"sched_{safe_id}_{ts}"

        adapter = self._adapters.get(entry.agent) or next(iter(self._adapters.values()), None)
        if adapter:
            sched_session = Session(
                id=conv_id,
                agent=entry.agent,
                source=SessionSource.SCHEDULE.value,
                status=SessionStatus.RUNNING.value,
                parent_id=entry.id,
                prompt=entry.prompt or entry.command or "",
                title=entry.id,
            )
            try:
                adapter.session_store.create_session(sched_session)
            except ValueError:
                pass
        return conv_id

    def _update_run_session(self, conv_id: str, entry: ScheduleEntry, **fields):
        """Update a schedule run session's status."""
        adapter = self._adapters.get(entry.agent) or next(iter(self._adapters.values()), None)
        if adapter:
            try:
                adapter.session_store.update_session(conv_id, **fields)
            except ValueError:
                pass

    async def _run_script(self, entry: ScheduleEntry) -> RunResult:
        """Run a shell command directly (no LLM)."""
        logger.info("Schedule '%s' executing script: %s", entry.id, entry.command[:100])
        conv_id = self._create_run_session(entry)

        try:
            proc = await asyncio.to_thread(
                subprocess.run,
                entry.command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=entry.script_timeout,
            )
        except subprocess.TimeoutExpired as e:
            self._update_run_session(conv_id, entry, status=SessionStatus.FAILED.value, error=str(e))
            raise RuntimeError(f"Script timed out after {entry.script_timeout}s") from e

        if proc.returncode != 0:
            output = (proc.stderr or proc.stdout or "")[:2000]
            self._update_run_session(conv_id, entry, status=SessionStatus.FAILED.value, error=output)
            raise RuntimeError(f"Script exited with code {proc.returncode}: {output}")

        result = proc.stdout[:_MAX_RESULT_CHARS]
        self._update_run_session(conv_id, entry, status=SessionStatus.COMPLETED.value, result=result[:2000])
        logger.info("Schedule '%s' script completed (exit 0)", entry.id)

        resolved_channels = self._resolve_channels(entry.notify) if entry.notify else []
        if resolved_channels:
            try:
                notification = f"**Schedule `{entry.id}` (script) completed:**\n\n```\n{result}\n```"
                await asyncio.to_thread(send_notification, notification, resolved_channels)
            except Exception as e:
                logger.error("Notification for script schedule '%s' failed: %s", entry.id, e)

            if entry.inject_history:
                adapter = next(iter(self._adapters.values()), None)
                if adapter:
                    await self._inject_into_user_sessions(adapter, entry, result, resolved_channels)

        return RunResult(output=result)

    async def _run_agent(self, entry: ScheduleEntry) -> RunResult:
        adapter = self._adapters.get(entry.agent)
        if not adapter:
            raise ValueError(f"No adapter found for agent '{entry.agent}'")
        logger.info("Schedule '%s' executing agent '%s': %s", entry.id, entry.agent, entry.prompt[:100])
        conv_id = self._create_run_session(entry)
        user_id = f"scheduler:{entry.agent}"
        metadata = {
            "schedule_id": entry.id,
            "running_tasks": self.scheduler.get_running_ids(),
            "conv_id_override": conv_id,
        }
        if entry.notify_tool:
            metadata["notify_tool"] = True
        if entry.model:
            metadata["model_override"] = entry.model
        if entry.max_turns is not None:
            metadata["max_turns_override"] = entry.max_turns

        if entry.agent_file:
            resolved = adapter._resolve_agent_path(entry.agent_file)
            if not resolved:
                raise FileNotFoundError(f"Agent file not found: {entry.agent_file}")
            metadata["agent_file_override"] = str(resolved)

        # Issue a temporary token for this scheduled task
        temp_token = ""
        if self._token_store:
            temp_token = self._token_store.issue(agent=entry.agent, schedule_id=entry.id)
        metadata["tsugite_url"] = self._tsugite_api_url
        metadata["tsugite_token"] = temp_token

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
        try:
            with ctx:
                result = await adapter.handle_message(
                    user_id=user_id,
                    message=entry.prompt,
                    channel_context=channel_context,
                )
        except AgentSkippedError:
            raise
        except AgentExecutionError as e:
            self._update_run_session(conv_id, entry, status=SessionStatus.FAILED.value, error=str(e))
            if resolved_channels:
                try:
                    notification = f"**Background task `{entry.id}` failed:**\n\n{e}"
                    if e.partial_output:
                        notification += f"\n\n**Partial output:**\n{e.partial_output[:2000]}"
                    await asyncio.to_thread(send_notification, notification, resolved_channels)
                except Exception as notify_err:
                    logger.error("Failure notification for schedule '%s' failed: %s", entry.id, notify_err)
            raise
        finally:
            if temp_token and self._token_store:
                self._token_store.revoke(temp_token)

        self._update_run_session(conv_id, entry, status=SessionStatus.COMPLETED.value, result=result[:2000])
        logger.info("Schedule '%s' agent '%s' completed", entry.id, entry.agent)

        if resolved_channels:
            truncated = result[:_MAX_RESULT_CHARS]

            if entry.auto_reply:
                await self._auto_reply(adapter, entry, truncated, resolved_channels)
            else:
                try:
                    notification = f"**Schedule `{entry.id}` completed:**\n\n{truncated}"
                    await asyncio.to_thread(send_notification, notification, resolved_channels)
                except Exception as e:
                    logger.error("Auto-notify for schedule '%s' failed: %s", entry.id, e)

                if entry.inject_history:
                    await self._inject_into_user_sessions(adapter, entry, truncated, resolved_channels)

        await self._handle_on_complete(entry, result)

        return RunResult(output=result, session_id=conv_id)

    async def _auto_reply(
        self,
        adapter: BaseAdapter,
        entry: ScheduleEntry,
        truncated_result: str,
        resolved_channels: list[tuple[str, NotificationChannelConfig]],
    ) -> None:
        """Process background task result on the user's session and send a response."""
        for _name, config in resolved_channels:
            if config.type != "discord":
                continue

            canonical = self._resolve_canonical_user(config)

            synthetic_message = (
                f'<background_task id="{entry.id}">\n'
                "This task ran in the background. Process the result and provide a "
                "concise, human-friendly summary to the user.\n\n"
                f"Original prompt: {entry.prompt}\n\n"
                f"Result:\n{truncated_result}\n"
                "</background_task>"
            )

            channel_context = ChannelContext(
                source="background_task",
                channel_id=None,
                user_id=canonical,
                reply_to=canonical,
                metadata={"schedule_id": entry.id, "background_task": True},
            )

            try:
                response = await adapter.handle_message(
                    user_id=canonical,
                    message=synthetic_message,
                    channel_context=channel_context,
                )
                notification = f"**Background task `{entry.id}` result:**\n\n{response[:_MAX_RESULT_CHARS]}"
                await asyncio.to_thread(send_notification, notification, [(_name, config)])
            except Exception as e:
                logger.error("Auto-reply for schedule '%s' user '%s' failed: %s", entry.id, canonical, e)
                # Fall back to raw notification
                try:
                    notification = f"**Background task `{entry.id}` completed:**\n\n{truncated_result}"
                    await asyncio.to_thread(send_notification, notification, [(_name, config)])
                except Exception as e2:
                    logger.error("Fallback notification for '%s' also failed: %s", entry.id, e2)

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

            canonical = self._resolve_canonical_user(config)

            try:
                await asyncio.to_thread(self._record_synthetic_turn, adapter, canonical, entry, truncated_result)
            except Exception as e:
                logger.error("Failed to inject history for schedule '%s' user '%s': %s", entry.id, canonical, e)

    @staticmethod
    def _get_session_storage(session_id: str, adapter: BaseAdapter) -> "SessionStorage":
        from tsugite.history import SessionStorage, get_history_dir

        session_path = get_history_dir() / f"{session_id}.jsonl"
        return SessionStorage.get_or_create(
            session_id, adapter.agent_name, adapter.resolve_model(), session_path=session_path,
        )

    @staticmethod
    def _record_synthetic_turn(adapter: BaseAdapter, user_id: str, entry: ScheduleEntry, result: str) -> None:
        """Write a synthetic turn into the user's session JSONL."""
        session = adapter.session_store.get_or_create_interactive(user_id, adapter.agent_name)
        storage = SchedulerAdapter._get_session_storage(session.id, adapter)

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


    async def _handle_on_complete(self, entry: ScheduleEntry, result: str) -> None:
        """Handle on_complete callback after a background task finishes."""
        if not entry.on_complete or entry.on_complete.get("action") != "reply":
            return

        session_id = entry.originating_session_id
        if not session_id or not self._session_runner:
            logger.warning("on_complete for '%s' skipped: no session runner or originating session", entry.id)
            return

        if entry.chain_depth >= MAX_CHAIN_DEPTH:
            logger.warning(
                "Chain depth %d reached max %d for task '%s', skipping auto-reply",
                entry.chain_depth, MAX_CHAIN_DEPTH, entry.id,
            )
            return

        truncated = result[:_MAX_RESULT_CHARS]
        prompt_summary = entry.prompt[:200] + ("…" if len(entry.prompt) > 200 else "")
        message = (
            f'<background_task_complete id="{entry.id}" chain_depth="{entry.chain_depth}">\n'
            f"  <prompt>{prompt_summary}</prompt>\n"
            f"  <result>\n{truncated}\n  </result>\n"
            "</background_task_complete>"
        )

        # If the session is mid-turn, inject into history so the agent sees it next turn.
        # Otherwise, reply directly to wake the agent.
        if self._session_runner.is_session_running(session_id):
            try:
                await self._inject_completion_into_history(session_id, entry, message)
            except Exception as e:
                logger.error("Failed to inject completion history for task '%s': %s", entry.id, e)
            return

        from tsugite.daemon.session_runner import set_current_chain_depth

        set_current_chain_depth(entry.chain_depth + 1)
        try:
            await self._session_runner.reply_to_session(
                session_id, message,
                source="completion_callback",
                metadata={"schedule_id": entry.id, "completion_callback": True},
            )
        except Exception as e:
            logger.error("on_complete reply to session '%s' failed: %s", session_id, e)
        finally:
            set_current_chain_depth(0)

    async def _inject_completion_into_history(
        self, session_id: str, entry: ScheduleEntry, message: str,
    ) -> None:
        """Write a completion result as a synthetic turn into the session's JSONL."""

        def _write():
            adapter = self._adapters.get(entry.agent) or next(iter(self._adapters.values()), None)
            if not adapter:
                return
            storage = self._get_session_storage(session_id, adapter)
            messages = [
                {"role": "user", "content": message},
                {"role": "assistant", "content": f"Background task {entry.id} completed. Result noted."},
            ]
            storage.record_turn(
                messages=messages,
                final_answer=message,
                metadata={"synthetic": True, "schedule_id": entry.id, "completion_callback": True},
            )

        await asyncio.to_thread(_write)
