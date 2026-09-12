"""Scheduled-task usage attribution: the schedule_id marker the daemon Usage
tab's per-schedule breakdown depends on.

This covers the producer half. SchedulerAdapter._run_agent stamps
metadata["schedule_id"] onto the ChannelContext it hands to the agent, so a run
knows which schedule spawned it. The consumer half, where the runner turns that
marker into the usage row's schedule_name, is driven end to end in
tests/daemon/test_usage_single_row.py.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from tsugite_daemon.scheduler import ScheduleEntry


def _make_scheduler_adapter(tmp_path, agent_name="bot"):
    from tsugite_daemon.adapters.scheduler_adapter import SchedulerAdapter

    adapter_mock = AsyncMock()
    adapter_mock.handle_message = AsyncMock(return_value="done")
    sa = SchedulerAdapter(
        adapter=adapter_mock,
        schedules_path=tmp_path / "schedules.json",
    )
    return sa, adapter_mock


@pytest.mark.asyncio
async def test_run_agent_stamps_schedule_id_in_metadata(tmp_path):
    sa, adapter_mock = _make_scheduler_adapter(tmp_path)
    entry = ScheduleEntry(id="morning-report", prompt="hi", schedule_type="cron", cron_expr="0 9 * * *")

    await sa._run_agent(entry)

    ctx = adapter_mock.handle_message.call_args[1]["channel_context"]
    assert ctx.metadata["schedule_id"] == "morning-report"
    assert ctx.source == "scheduler"
