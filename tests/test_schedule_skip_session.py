"""A run_if-skipped schedule must not leave a session row behind.

The guard fires deep inside handle_message, after the run's session already
exists, so the skip path is the only place that can decide the row was never
worth keeping. Genuine cancellation is a different path (session_runner's
asyncio.CancelledError handler) and is not involved here.
"""

from unittest.mock import AsyncMock

import pytest
from tsugite_daemon.scheduler import ScheduleEntry
from tsugite_daemon.session_store import SessionSource, SessionStatus, SessionStore

from tsugite.agent_runner.models import AgentSkippedError
from tsugite.exceptions import AgentExecutionError


def _make_scheduler_adapter(tmp_path):
    from tsugite_daemon.adapters.scheduler_adapter import SchedulerAdapter

    store = SessionStore(tmp_path / "session_store.json")
    adapter_mock = AsyncMock()
    adapter_mock.agent_name = "bot"
    adapter_mock.handle_message = AsyncMock(return_value="done")
    adapter_mock.session_store = store

    sa = SchedulerAdapter(adapters={"bot": adapter_mock}, schedules_path=tmp_path / "schedules.json")
    return sa, adapter_mock, store


def _entry():
    return ScheduleEntry(id="job1", agent="bot", prompt="hi", schedule_type="cron", cron_expr="*/5 * * * *")


def _schedule_sessions(store):
    return store.list_sessions(source=SessionSource.SCHEDULE.value)


@pytest.mark.asyncio
async def test_a_skipped_run_leaves_no_session_behind(tmp_path):
    sa, adapter_mock, store = _make_scheduler_adapter(tmp_path)
    adapter_mock.handle_message = AsyncMock(side_effect=AgentSkippedError("run_if guard"))

    with pytest.raises(AgentSkippedError):
        await sa._run_agent(_entry())

    assert _schedule_sessions(store) == []


@pytest.mark.asyncio
async def test_a_run_that_passes_the_guard_keeps_its_session(tmp_path):
    sa, _adapter, store = _make_scheduler_adapter(tmp_path)

    await sa._run_agent(_entry())

    sessions = _schedule_sessions(store)
    assert [s.status for s in sessions] == [SessionStatus.COMPLETED.value]


@pytest.mark.asyncio
async def test_a_failed_run_keeps_its_session(tmp_path):
    """Only the skip is a non-event; a run that started and broke is worth a row."""
    sa, adapter_mock, store = _make_scheduler_adapter(tmp_path)
    adapter_mock.handle_message = AsyncMock(side_effect=AgentExecutionError("boom"))

    with pytest.raises(AgentExecutionError):
        await sa._run_agent(_entry())

    sessions = _schedule_sessions(store)
    assert [s.status for s in sessions] == [SessionStatus.FAILED.value]


@pytest.mark.asyncio
async def test_a_script_run_still_reaches_a_terminal_status(tmp_path):
    """Script schedules share `_create_run_session`, and a conv_id it cannot use
    fails silently: `_update_run_session` swallows the ValueError and the row
    stays RUNNING forever."""
    sa, _adapter, store = _make_scheduler_adapter(tmp_path)
    entry = _entry()
    entry.execution_type = "script"
    entry.command = "true"

    await sa._run_script(entry)

    assert [s.status for s in _schedule_sessions(store)] == [SessionStatus.COMPLETED.value]


@pytest.mark.asyncio
async def test_a_skip_keeps_a_session_the_schedule_reuses_across_runs(tmp_path):
    """`session_id` pins one conv_id for every run of a schedule, so its row carries
    the accumulated conversation. A skip must drop only a row it opened itself."""
    sa, adapter_mock, store = _make_scheduler_adapter(tmp_path)
    entry = _entry()
    entry.session_id = "standing-convo"
    await sa._run_agent(entry)

    adapter_mock.handle_message = AsyncMock(side_effect=AgentSkippedError("run_if guard"))
    with pytest.raises(AgentSkippedError):
        await sa._run_agent(entry)

    sessions = _schedule_sessions(store)
    assert [s.status for s in sessions] == [SessionStatus.COMPLETED.value]
