"""A script run's session carries the script's output.

The run row stores the output in its `result` field, which no client renders,
so the run's transcript opened empty. The output lands in the run session's own
history as a delivery. Ordering matters: `deliver_to_session` refuses a session
already in a finished status, so the delivery precedes the status update.
"""

from unittest.mock import AsyncMock

import pytest
from tsugite_daemon.scheduler import ScheduleEntry
from tsugite_daemon.session_runner import SessionRunner
from tsugite_daemon.session_store import SessionSource, SessionStatus, SessionStore


def _make_scheduler_adapter(tmp_path):
    from tsugite_daemon.adapters.scheduler_adapter import SchedulerAdapter

    store = SessionStore(tmp_path / "session_store.json")
    adapter_mock = AsyncMock()
    adapter_mock.agent_name = "bot"
    adapter_mock.session_store = store
    adapter_mock.event_bus = None

    sa = SchedulerAdapter(adapter=adapter_mock, schedules_path=tmp_path / "schedules.json")
    sa.set_session_runner(SessionRunner(store=store, adapter=None))
    return sa, store


def _script_entry(command: str) -> ScheduleEntry:
    return ScheduleEntry(
        id="job1",
        prompt="",
        schedule_type="cron",
        cron_expr="*/5 * * * *",
        execution_type="script",
        command=command,
    )


def _run_session(store: SessionStore):
    sessions = store.list_sessions(source=SessionSource.SCHEDULE.value)
    assert len(sessions) == 1
    return sessions[0]


def _deliveries(store: SessionStore, session_id: str) -> list[dict]:
    return [e for e in store.read_events(session_id) if e["type"] == "delivery"]


@pytest.mark.asyncio
async def test_a_completed_script_run_records_its_output_in_its_own_session(tmp_path, history_dir):
    sa, store = _make_scheduler_adapter(tmp_path)

    await sa._run_script(_script_entry("echo ingested 42 docs"))

    session = _run_session(store)
    assert session.status == SessionStatus.COMPLETED.value
    deliveries = _deliveries(store, session.id)
    assert len(deliveries) == 1
    assert "ingested 42 docs" in deliveries[0]["message"]
    assert deliveries[0]["source"] == "schedule_script"


@pytest.mark.asyncio
async def test_a_failed_script_run_records_its_stderr_and_exit_code(tmp_path, history_dir):
    sa, store = _make_scheduler_adapter(tmp_path)

    with pytest.raises(RuntimeError):
        await sa._run_script(_script_entry("echo disk full >&2; exit 3"))

    session = _run_session(store)
    assert session.status == SessionStatus.FAILED.value
    deliveries = _deliveries(store, session.id)
    assert len(deliveries) == 1
    assert "disk full" in deliveries[0]["message"]
    assert "3" in deliveries[0]["message"]
