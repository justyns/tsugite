"""Scheduler adapter — bridges the Scheduler into the daemon Gateway."""

import asyncio
import logging
import subprocess
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path

from tsugite.agent_runner.models import AgentSkippedError
from tsugite.exceptions import AgentExecutionError
from tsugite.tools.notify import notify_context, send_notification
from tsugite_daemon.adapters.base import BaseAdapter, ChannelContext
from tsugite_daemon.auth import TokenStore
from tsugite_daemon.config import NotificationChannelConfig
from tsugite_daemon.scheduler import (
    DELIVERY_MODE_AUTO,
    DELIVERY_MODE_NEW,
    DELIVERY_MODE_PARENT,
    TARGET_SESSION_NAME_PREFIX,
    TARGET_SESSION_NONE,
    TARGET_SESSION_ORIGINATING,
    TARGET_SESSION_PRIMARY,
    RunResult,
    ScheduleEntry,
    Scheduler,
)
from tsugite_daemon.session_runner import DELIVERY_KIND_NEEDS_ACK, MAX_CHAIN_DEPTH
from tsugite_daemon.session_store import (
    METADATA_INCIDENT_KEY,
    Session,
    SessionSource,
    SessionStatus,
    create_interactive_session,
)

logger = logging.getLogger(__name__)

_MAX_RESULT_CHARS = 4000


def _recorded_run_outcome(conv_id: str | None) -> tuple[str, str | None]:
    """An agent that answered with unexecuted tool-call markup, or hit max_turns,
    returns text and records a non-success end.
    """
    from tsugite.history import get_history_backend

    if not conv_id:
        return "success", None
    backend = get_history_backend()
    if not backend.exists(conv_id):
        return "success", None
    summary = backend.load(conv_id).summary()
    return (summary.status or "success"), summary.error_message


def _resolve_originating(entry: ScheduleEntry, store) -> Session | None:
    sid = entry.originating_session_id
    return store.resolve_live(sid) if sid else None


def resolve_target_session(entry: ScheduleEntry, user_id: str | None, store) -> Session | None:
    """Resolve `entry.target_session` to a concrete Session, or None to skip injection.

    See ScheduleEntry.target_session for the legal value forms.
    """
    spec = entry.target_session
    if spec == TARGET_SESSION_NONE:
        return None
    if spec is None:
        return store.find_primary_session(user_id) or _resolve_originating(entry, store)
    if spec == TARGET_SESSION_PRIMARY:
        return store.find_primary_session(user_id)
    if spec == TARGET_SESSION_ORIGINATING:
        return _resolve_originating(entry, store)
    if spec.startswith(TARGET_SESSION_NAME_PREFIX):
        return store.find_named_session(user_id, spec[len(TARGET_SESSION_NAME_PREFIX) :])
    return store.resolve_live(spec)


def _incident_session(entry: ScheduleEntry, user_id: str, store, event_bus=None) -> Session:
    key = entry.incident_key or entry.id
    existing = store.find_incident_session(user_id, key)
    if existing:
        return existing

    title = entry.incident_title or f"Incident: {entry.id}"
    session_id = create_interactive_session(
        store,
        user_id,
        title=title,
        event_bus=event_bus,
        metadata={"type": "ops", "topic": title, "schedule_id": entry.id, METADATA_INCIDENT_KEY: key},
        source=SessionSource.SCHEDULE.value,
    )
    return store.get_session(session_id)


def resolve_delivery_sessions(entry: ScheduleEntry, user_ids: list[str], store, event_bus=None) -> list[Session]:
    mode = entry.delivery_mode
    if mode == DELIVERY_MODE_PARENT:
        candidates = [_resolve_originating(entry, store)]
    elif mode == DELIVERY_MODE_NEW or (mode == DELIVERY_MODE_AUTO and entry.delivery_kind == DELIVERY_KIND_NEEDS_ACK):
        candidates = [_incident_session(entry, user_ids[0], store, event_bus)]
    else:
        candidates = [resolve_target_session(entry, user_id, store) for user_id in user_ids]

    return list({s.id: s for s in candidates if s is not None}.values())


class SchedulerAdapter:
    """Integrates the Scheduler with the daemon, executing runs via the daemon adapter."""

    def __init__(
        self,
        adapter: BaseAdapter,
        schedules_path: Path,
        notification_channels: dict[str, NotificationChannelConfig] | None = None,
        identity_map: dict[str, str] | None = None,
        token_store: TokenStore | None = None,
        tsugite_api_url: str = "",
    ):
        self._adapter = adapter
        self._notification_channels = notification_channels or {}
        self._identity_map = identity_map or {}
        self._token_store = token_store
        self._tsugite_api_url = tsugite_api_url
        self._session_runner = None
        self._failure_tasks: set[asyncio.Task] = set()
        self.scheduler = Scheduler(
            schedules_path,
            self._run_agent,
            script_callback=self._run_script,
            on_repeated_failure=self._on_repeated_failure,
        )

    def set_session_runner(self, session_runner) -> None:
        """Set the SessionRunner reference (called after both are constructed)."""
        self._session_runner = session_runner

    async def start(self):
        await self.scheduler.start()

    async def stop(self):
        await self.scheduler.stop()

    def _on_repeated_failure(self, entry: ScheduleEntry) -> None:
        adapter = self._run_adapter(entry)
        if not adapter:
            return
        message = (
            f"Schedule `{entry.id}` has failed {entry.consecutive_failures} runs in a row.\n\n"
            f"{entry.last_error or 'No error was recorded.'}"
        )
        task = asyncio.create_task(
            self._deliver_result(
                adapter,
                entry,
                message,
                self._resolve_channels(entry.notify) if entry.notify else [],
                source="schedule_failure",
                kind=DELIVERY_KIND_NEEDS_ACK,
                title=f"Schedule failing: {entry.id}",
            )
        )
        self._failure_tasks.add(task)
        task.add_done_callback(self._failure_tasks.discard)
        task.add_done_callback(self._log_failure_delivery)

    @staticmethod
    def _log_failure_delivery(task: asyncio.Task) -> None:
        if not task.cancelled() and task.exception():
            logger.error("Repeated-failure delivery failed: %s", task.exception())

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

    def _run_adapter(self, entry: ScheduleEntry) -> BaseAdapter | None:
        return self._adapter

    def _create_run_session(self, entry: ScheduleEntry) -> tuple[str, bool]:
        """Create a Session record for a schedule run.

        Returns:
            The conv_id, and whether this run opened the record. A schedule with
            `session_id` reuses one record across every run, so only the run that
            opened one may discard it.
        """
        if entry.session_id:
            conv_id = f"sched_{entry.session_id}"
        else:
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            safe_id = entry.id.replace(":", "_")
            conv_id = f"sched_{safe_id}_{ts}"

        adapter = self._run_adapter(entry)
        if not adapter:
            return conv_id, False
        sched_session = Session(
            id=conv_id,
            source=SessionSource.SCHEDULE.value,
            status=SessionStatus.RUNNING.value,
            parent_id=entry.id,
            prompt=entry.prompt or entry.command or "",
            title=entry.id,
        )
        try:
            adapter.session_store.create_session(sched_session)
        except ValueError:
            return conv_id, False
        return conv_id, True

    def _update_run_session(self, conv_id: str, entry: ScheduleEntry, **fields):
        """Update a schedule run session's status."""
        adapter = self._run_adapter(entry)
        if adapter:
            try:
                adapter.session_store.update_session(conv_id, **fields)
            except ValueError as e:
                logger.warning("Schedule '%s' session update failed for %s: %s", entry.id, conv_id, e)

    async def _run_script(self, entry: ScheduleEntry) -> RunResult:
        """Run a shell command directly (no LLM)."""
        logger.info("Schedule '%s' executing script: %s", entry.id, entry.command[:100])
        conv_id, _opened = self._create_run_session(entry)

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

        adapter = self._adapter
        if entry.inject_history and adapter:
            await self._deliver_result(adapter, entry, result, resolved_channels)

        return RunResult(output=result)

    async def _run_agent(self, entry: ScheduleEntry) -> RunResult:
        adapter = self._adapter
        if not adapter:
            raise ValueError("No adapter available to run schedules")
        logger.info("Schedule '%s' executing: %s", entry.id, entry.prompt[:100])
        conv_id, opened_session = self._create_run_session(entry)
        user_id = "scheduler"
        metadata = {
            "schedule_id": entry.id,
            "running_tasks": self.scheduler.get_running_ids(),
            "conv_id_override": conv_id,
            "actual_fire_time": datetime.now(timezone.utc).isoformat(),
        }
        if entry.last_scheduled_for:
            metadata["scheduled_for"] = entry.last_scheduled_for
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
            temp_token = self._token_store.issue(schedule_id=entry.id)
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
            # A guard that declines is the schedule working, not a run that happened.
            # The scheduler logs the skip and records it in the entry's run history.
            if opened_session:
                adapter.session_store.delete_session(conv_id)
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
        except Exception as e:
            # Anything else escaping handle_message must still leave the session in a terminal
            # state, otherwise the sidebar pins the card at "Starting..." indefinitely.
            self._update_run_session(conv_id, entry, status=SessionStatus.FAILED.value, error=str(e))
            raise
        finally:
            if temp_token and self._token_store:
                self._token_store.revoke(temp_token)

        self._update_run_session(conv_id, entry, status=SessionStatus.COMPLETED.value, result=result[:2000])
        logger.info("Schedule '%s' completed", entry.id)

        await self._publish_result(adapter, entry, result[:_MAX_RESULT_CHARS], resolved_channels)
        await self._handle_on_complete(entry, result)

        status, error = _recorded_run_outcome(conv_id)
        return RunResult(output=result, session_id=conv_id, status=status, error=error)

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
                # Webhook / web-push channels have no per-user chat session to
                # process the result on - deliver the raw result instead of
                # silently dropping it (auto_reply is set whenever any notify
                # channel is configured, not just Discord ones).
                try:
                    notification = f"**Background task `{entry.id}` completed:**\n\n{truncated_result}"
                    await asyncio.to_thread(send_notification, notification, [(_name, config)])
                except Exception as e:
                    logger.error("Auto-reply notification for '%s' via '%s' failed: %s", entry.id, _name, e)
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

    async def _publish_result(
        self,
        adapter: BaseAdapter,
        entry: ScheduleEntry,
        truncated: str,
        resolved_channels: list[tuple[str, NotificationChannelConfig]],
    ) -> None:
        if resolved_channels and entry.auto_reply:
            await self._auto_reply(adapter, entry, truncated, resolved_channels)
            return
        if resolved_channels:
            try:
                notification = f"**Schedule `{entry.id}` completed:**\n\n{truncated}"
                await asyncio.to_thread(send_notification, notification, resolved_channels)
            except Exception as e:
                logger.error("Auto-notify for schedule '%s' failed: %s", entry.id, e)
        if entry.inject_history:
            await self._deliver_result(adapter, entry, truncated, resolved_channels)

    def _delivery_recipients(
        self,
        adapter: BaseAdapter,
        entry: ScheduleEntry,
        resolved_channels: list[tuple[str, NotificationChannelConfig]],
    ) -> list[str]:
        """With several owners and nothing naming one, returns [""] so the run addresses no user."""
        users = [
            self._resolve_canonical_user(config) for _name, config in resolved_channels if config.type == "discord"
        ]
        if users:
            return users
        originating = _resolve_originating(entry, adapter.session_store)
        if originating and originating.user_id:
            return [originating.user_id]
        owners = sorted(adapter.session_store.default_primary_ids())
        if len(owners) > 1:
            logger.debug(
                "Schedule '%s' names no user and %d could be meant; delivering only to a named session",
                entry.id,
                len(owners),
            )
            return [""]
        return owners or [""]

    async def _deliver_result(
        self,
        adapter: BaseAdapter,
        entry: ScheduleEntry,
        truncated_result: str,
        resolved_channels: list[tuple[str, NotificationChannelConfig]],
        *,
        source: str = "schedule",
        kind: str | None = None,
        title: str | None = None,
    ) -> None:
        if not self._session_runner:
            logger.debug("Schedule '%s' has no session runner; skipping delivery", entry.id)
            return
        recipients = self._delivery_recipients(adapter, entry, resolved_channels)
        sessions = resolve_delivery_sessions(entry, recipients, adapter.session_store, adapter.event_bus)
        if not sessions:
            logger.debug("Schedule '%s' has no delivery target; skipping delivery", entry.id)
            return
        for session in sessions:
            try:
                await asyncio.to_thread(
                    self._session_runner.deliver_to_session,
                    session.id,
                    truncated_result,
                    source=source,
                    kind=kind or entry.delivery_kind,
                    title=title or entry.incident_title,
                    metadata={"schedule_id": entry.id},
                    notify_channels=resolved_channels,
                )
            except Exception as e:
                logger.error("Delivery for schedule '%s' to session '%s' failed: %s", entry.id, session.id, e)

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
                entry.chain_depth,
                MAX_CHAIN_DEPTH,
                entry.id,
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

        if self._session_runner.is_session_running(session_id):
            try:
                await asyncio.to_thread(
                    self._session_runner.deliver_to_session,
                    session_id,
                    message,
                    source="completion_callback",
                    metadata={"schedule_id": entry.id, "completion_callback": True},
                )
            except Exception as e:
                logger.error("Completion delivery for task '%s' failed: %s", entry.id, e)
            return

        from tsugite_daemon.session_runner import set_current_chain_depth

        set_current_chain_depth(entry.chain_depth + 1)
        try:
            await self._session_runner.reply_to_session(
                session_id,
                message,
                source="completion_callback",
                metadata={"schedule_id": entry.id, "completion_callback": True},
            )
        except Exception as e:
            logger.error("on_complete reply to session '%s' failed: %s", session_id, e)
        finally:
            set_current_chain_depth(0)
