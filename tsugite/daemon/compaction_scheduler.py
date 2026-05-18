"""Scheduled auto-compaction for daemon agents."""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict

from cronsim import CronSim

from tsugite.daemon.config import AgentConfig
from tsugite.daemon.session_store import SessionStore

logger = logging.getLogger(__name__)


class CompactionScheduler:
    """Fires scheduled compactions based on auto_compact config in daemon.yaml."""

    def __init__(
        self,
        agents: Dict[str, AgentConfig],
        session_store: SessionStore,
        adapters: dict,
    ):
        self._agents = agents
        self._session_store = session_store
        self._adapters = adapters
        self._running = False
        self._wakeup = asyncio.Event()

    async def start(self):
        self._running = True
        logger.info(
            "Compaction scheduler started for %d agent(s): %s",
            len(self._agents),
            ", ".join(self._agents.keys()),
        )
        await self._main_loop()

    async def stop(self):
        self._running = False
        self._wakeup.set()

    async def _main_loop(self):
        while self._running:
            now = datetime.now(timezone.utc)

            # Compute next fire time per agent in a single pass
            agent_fire_times: dict[str, datetime] = {}
            earliest = None
            for agent_name, agent_config in self._agents.items():
                schedule = agent_config.auto_compact and agent_config.auto_compact.schedule
                if not schedule:
                    continue
                try:
                    agent_next = next(CronSim(schedule, now))
                    agent_fire_times[agent_name] = agent_next
                    if earliest is None or agent_next < earliest:
                        earliest = agent_next
                except StopIteration:
                    continue

            if earliest is None:
                self._wakeup.clear()
                try:
                    await asyncio.wait_for(self._wakeup.wait(), timeout=60)
                except asyncio.TimeoutError:
                    pass
                continue

            delay = (earliest - datetime.now(timezone.utc)).total_seconds()
            if delay > 0:
                self._wakeup.clear()
                try:
                    await asyncio.wait_for(self._wakeup.wait(), timeout=delay)
                except asyncio.TimeoutError:
                    pass
                if not self._running:
                    break

            # Only fire agents whose scheduled time has arrived
            now = datetime.now(timezone.utc)
            for agent_name, fire_time in agent_fire_times.items():
                if fire_time > now:
                    continue
                await self._check_agent(agent_name, self._agents[agent_name])

            await asyncio.sleep(1)

    async def _check_agent(self, agent_name: str, agent_config: AgentConfig):
        auto_compact = agent_config.auto_compact
        if not auto_compact:
            return

        sessions = self._session_store.list_interactive_by_agent(agent_name)
        if not sessions:
            return

        adapter = self._adapters.get(agent_name)
        if not adapter:
            logger.warning("No adapter found for agent '%s', skipping scheduled compaction", agent_name)
            return

        # A session that's idle but already loaded with retained context (e.g. carried
        # over from a previous compaction) should still be compacted on schedule.
        # `min_turns` alone undercounts because retained events from prior compactions
        # don't bump message_count.
        from tsugite.daemon.memory import RETENTION_BUDGET_RATIO

        context_limit = self._session_store.get_context_limit(agent_name)
        retention_budget = int(context_limit * RETENTION_BUDGET_RATIO)

        for session in sessions:
            few_turns = session.message_count < auto_compact.min_turns
            small_context = session.cumulative_tokens < retention_budget
            if few_turns and small_context:
                logger.debug(
                    "[%s] Skipping scheduled compaction: %d turns < %d and %d tokens < retention budget %d",
                    agent_name,
                    session.message_count,
                    auto_compact.min_turns,
                    session.cumulative_tokens,
                    retention_budget,
                )
                continue

            user_id = session.user_id or ""
            sid = session.id
            if not self._session_store.begin_compaction(user_id, agent_name, session_id=sid):
                logger.debug("[%s] Compaction already in progress, skipping", agent_name)
                continue

            logger.info(
                "[%s] Scheduled compaction triggered (%d turns)",
                agent_name,
                session.message_count,
            )

            try:
                new_session = await adapter._compact_session(sid, reason="scheduled")
                if new_session is None:
                    logger.info("[%s] Scheduled compaction skipped (nothing to compact)", agent_name)
                else:
                    logger.info(
                        "[%s] Scheduled compaction completed (old=%s new=%s)",
                        agent_name,
                        sid,
                        new_session.id,
                    )
            except Exception:
                logger.exception("[%s] Scheduled compaction failed", agent_name)
            finally:
                self._session_store.end_compaction(user_id, agent_name, session_id=sid)
