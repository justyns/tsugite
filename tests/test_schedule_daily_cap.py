from datetime import datetime, timedelta
from unittest.mock import AsyncMock, Mock

import pytest
from tsugite_daemon.adapters.scheduler_adapter import SchedulerAdapter
from tsugite_daemon.scheduler import EXECUTION_SCRIPT, EXECUTION_SESSION_MESSAGE, RunResult, ScheduleEntry, Scheduler

from tsugite.history import SqliteHistoryBackend, set_history_backend


def _entry(id, **kw):
    return ScheduleEntry(id=id, prompt="hi", schedule_type="cron", cron_expr="*/5 * * * *", **kw)


@pytest.fixture
def callbacks():
    return {name: AsyncMock(return_value=RunResult(output="ok")) for name in ("run", "script", "message")}


@pytest.fixture
def history_backend(tmp_path):
    backend = SqliteHistoryBackend(tmp_path / "history.db")
    set_history_backend(backend)
    yield backend
    set_history_backend(None)


@pytest.fixture
def on_cap():
    return Mock()


@pytest.fixture
def scheduler(tmp_path, callbacks, on_cap):
    sched = Scheduler(
        tmp_path / "schedules.json",
        callbacks["run"],
        script_callback=callbacks["script"],
        session_message_callback=callbacks["message"],
        max_scheduled_agent_runs_per_day=2,
        on_daily_cap=on_cap,
    )
    entries = (
        _entry("a"),
        _entry("b", execution_type=EXECUTION_SESSION_MESSAGE, target_session="primary"),
        _entry("s", execution_type=EXECUTION_SCRIPT, command="true"),
    )
    sched._schedules.update({entry.id: entry for entry in entries})
    return sched


@pytest.mark.asyncio
async def test_cap_skips_unattended_agent_fires_and_notifies_once(scheduler, callbacks, on_cap):
    for _ in range(2):
        await scheduler._fire_schedule(scheduler.get("a"))
    for _ in range(2):
        await scheduler._fire_schedule(scheduler.get("b"))
    await scheduler._fire_schedule(scheduler.get("s"))
    await scheduler._fire_schedule(scheduler.get("a"), manual=True)

    assert callbacks["run"].await_count == 3  # two unattended fires and the manual one
    assert callbacks["message"].await_count == 0
    assert callbacks["script"].await_count == 1
    skipped = scheduler.get("b")
    assert (skipped.last_status, skipped.last_error, skipped.consecutive_failures) == ("skipped", "daily_cap", 0)
    assert skipped.run_history[-1]["error"] == "daily_cap"
    on_cap.assert_called_once()


@pytest.mark.asyncio
async def test_count_resets_at_the_next_local_midnight(scheduler, callbacks, on_cap):
    for _ in range(3):
        await scheduler._fire_schedule(scheduler.get("a"))
    assert callbacks["run"].await_count == 2

    tomorrow = scheduler._day_start() + timedelta(days=1)
    scheduler._day_start = lambda: tomorrow
    for _ in range(3):
        await scheduler._fire_schedule(scheduler.get("a"))

    assert callbacks["run"].await_count == 4
    assert on_cap.call_count == 2


@pytest.mark.asyncio
async def test_once_schedules_neither_count_nor_get_skipped(scheduler, callbacks, on_cap):
    once = ScheduleEntry(id="o", prompt="hi", schedule_type="once", run_at="2099-01-01T00:00:00+00:00")
    await scheduler._fire_schedule(once)
    for _ in range(2):
        await scheduler._fire_schedule(scheduler.get("a"))
    await scheduler._fire_schedule(once)

    assert callbacks["run"].await_count == 4
    on_cap.assert_not_called()


@pytest.mark.asyncio
async def test_startup_rebuilds_the_count_from_history(scheduler, history_backend):
    session = history_backend.create("agent", "model")
    for source, extra in (
        ("scheduler", {}),
        ("schedule_message", {}),
        ("scheduler", {"uncapped": True}),
        ("http", {}),
    ):
        session.record("user_input", text="hi", channel={"source": source, **extra})
    session.record(
        "user_input", text="old", channel={"source": "scheduler"}, ts=datetime.now().astimezone() - timedelta(days=2)
    )

    scheduler._rebuild_daily_count()

    assert scheduler._daily_count == 2


@pytest.mark.asyncio
async def test_manual_and_once_runs_mark_the_channel_metadata(tmp_path):
    adapter = AsyncMock()
    adapter.handle_message = AsyncMock(return_value="done")
    sa = SchedulerAdapter(adapter=adapter, schedules_path=tmp_path / "schedules.json")

    await sa._run_agent(_entry("a", manual_run=True))
    await sa._run_agent(ScheduleEntry(id="o", prompt="hi", schedule_type="once", run_at="2099-01-01T00:00:00+00:00"))
    await sa._run_agent(_entry("a"))

    manual, once, unattended = (
        call.kwargs["channel_context"].metadata for call in adapter.handle_message.await_args_list
    )
    assert manual["uncapped"] is True
    assert once["uncapped"] is True
    assert "uncapped" not in unattended
