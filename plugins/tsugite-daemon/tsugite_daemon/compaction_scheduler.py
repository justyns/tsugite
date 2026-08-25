"""Scheduled auto-compaction for daemon sessions."""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional
from zoneinfo import ZoneInfo

from cronsim import CronSim

from tsugite_daemon.config import RuntimeDefaults
from tsugite_daemon.session_store import SessionStore

logger = logging.getLogger(__name__)


class CompactionScheduler:
    """Fires scheduled compactions based on auto_compact config in daemon.yaml."""

    def __init__(
        self,
        runtime: RuntimeDefaults,
        session_store: SessionStore,
        adapter,
    ):
        self._runtime = runtime
        self._session_store = session_store
        self._adapter = adapter
        self._running = False
        self._wakeup = asyncio.Event()

    async def start(self):
        self._running = True
        logger.info("Compaction scheduler started (%s)", self._runtime.auto_compact.schedule)
        await self._main_loop()

    async def stop(self):
        self._running = False
        self._wakeup.set()

    def _compute_next_fire_time(self, now_utc: datetime) -> Optional[datetime]:
        """Next scheduled fire time for the auto-compact cron, in UTC.

        Cron expressions are interpreted in the configured timezone
        (`runtime.timezone`, IANA name). Empty timezone falls back to UTC.
        """
        schedule = self._runtime.auto_compact and self._runtime.auto_compact.schedule
        if not schedule:
            return None
        tz = ZoneInfo(self._runtime.timezone) if self._runtime.timezone else timezone.utc
        now_local = now_utc.astimezone(tz)
        try:
            next_local = next(CronSim(schedule, now_local))
        except StopIteration:
            return None
        return next_local.astimezone(timezone.utc)

    async def _main_loop(self):
        while self._running:
            next_fire = self._compute_next_fire_time(datetime.now(timezone.utc))

            if next_fire is None:
                self._wakeup.clear()
                try:
                    await asyncio.wait_for(self._wakeup.wait(), timeout=60)
                except asyncio.TimeoutError:
                    pass
                continue

            delay = (next_fire - datetime.now(timezone.utc)).total_seconds()
            if delay > 0:
                self._wakeup.clear()
                try:
                    await asyncio.wait_for(self._wakeup.wait(), timeout=delay)
                except asyncio.TimeoutError:
                    pass
                if not self._running:
                    break

            if next_fire <= datetime.now(timezone.utc):
                await self._check_sessions()

            await asyncio.sleep(1)

    async def _check_sessions(self):
        auto_compact = self._runtime.auto_compact
        if not auto_compact:
            return

        sessions = self._session_store.list_interactive()
        if not sessions:
            return

        adapter = self._adapter
        if not adapter:
            logger.warning("No adapter available, skipping scheduled compaction")
            return

        # A session that's idle but already loaded with retained context (e.g. carried
        # over from a previous compaction) should still be compacted on schedule.
        # `min_turns` alone undercounts because retained events from prior compactions
        # don't bump message_count.
        from tsugite_daemon.memory import RETENTION_BUDGET_RATIO

        context_limit = self._session_store.get_context_limit()
        retention_budget = int(context_limit * RETENTION_BUDGET_RATIO)

        for session in sessions:
            few_turns = session.message_count < auto_compact.min_turns
            small_context = session.cumulative_tokens < retention_budget
            if few_turns and small_context:
                logger.debug(
                    "Skipping scheduled compaction: %d turns < %d and %d tokens < retention budget %d",
                    session.message_count,
                    auto_compact.min_turns,
                    session.cumulative_tokens,
                    retention_budget,
                )
                continue

            user_id = session.user_id or ""
            sid = session.id
            # Never rotate a session mid-turn: the compaction snapshot misses
            # every event the in-flight turn writes after it, so the exchange
            # vanishes from the successor. The next cycle retries once the turn
            # settles.
            if session.has_live_work:
                logger.info("Skipping scheduled compaction of '%s': turn in flight", sid)
                continue
            if not self._session_store.begin_compaction(user_id, session_id=sid):
                logger.debug("Compaction already in progress, skipping")
                continue

            logger.info("Scheduled compaction triggered (%d turns)", session.message_count)

            try:
                new_session = await adapter._compact_session(sid, reason="scheduled")
                if new_session is None:
                    logger.info("Scheduled compaction skipped (nothing to compact)")
                else:
                    logger.info("Scheduled compaction completed (old=%s new=%s)", sid, new_session.id)
            except Exception:
                logger.exception("Scheduled compaction failed")
            finally:
                self._session_store.end_compaction(user_id, session_id=sid)
