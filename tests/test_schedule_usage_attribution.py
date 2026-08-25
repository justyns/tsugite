"""Scheduled-task usage attribution: the schedule_id marker the daemon Usage
tab's per-schedule breakdown depends on.

Two seams keep the chain honest:
- Producer: SchedulerAdapter._run_agent stamps metadata["schedule_id"] onto the
  ChannelContext it hands to the agent (so a run knows which schedule spawned it).
- Consumer: BaseAdapter.handle_message forwards that marker into the UsageStore
  record as schedule_name. handle_message can't be unit-driven to its record()
  call without a full fake turn, so this guards the exact forwarding expression
  against regression (mirrors test_session_source's source-inspection precedent).
"""

from __future__ import annotations

import inspect
import re
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


def test_handle_message_forwards_schedule_id_to_usage_record():
    """The usage record() call must derive schedule_name from the channel
    context's schedule_id marker, so scheduled turns land in by_schedule()."""
    from tsugite_daemon.adapters.base import BaseAdapter

    src = inspect.getsource(BaseAdapter._handle_message_inner)
    collapsed = re.sub(r"\s+", " ", src)
    assert 'schedule_name=(channel_context.metadata or {}).get("schedule_id")' in collapsed
