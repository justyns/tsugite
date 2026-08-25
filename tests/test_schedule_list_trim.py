"""schedule_list (agent tool) stays compact.

A deployment with many schedules, each keeping up to 20 run_history entries,
made schedule_list() return 100+ KB - which the exec-output cap then truncated
mid-object into a malformed blob. last_run/last_status/run_count already
summarize health for the list view; full history stays in schedule_status(id).
"""

import asyncio
from threading import Thread
from unittest.mock import MagicMock


def _run_schedule_list(entries):
    from tsugite.tools.schedule import schedule_list, set_scheduler

    loop = asyncio.new_event_loop()
    t = Thread(target=loop.run_forever, daemon=True)
    t.start()
    try:
        mock_sched = MagicMock()
        mock_sched.list.return_value = entries
        set_scheduler(mock_sched, loop)
        return schedule_list()
    finally:
        loop.call_soon_threadsafe(loop.stop)
        t.join(timeout=2)
        loop.close()
        set_scheduler(None)


def test_schedule_list_omits_run_history_but_keeps_summary():
    from tsugite_daemon.scheduler import ScheduleEntry

    entry = ScheduleEntry(id="daily", prompt="hi", schedule_type="cron", cron_expr="0 9 * * *")
    entry.run_history = [
        {"timestamp": "2026-01-01T00:00:00+00:00", "status": "success", "error": None, "session_id": "s"}
    ] * 20
    entry.last_status = "success"
    entry.run_count = 20

    result = _run_schedule_list([entry])

    assert len(result) == 1
    assert "run_history" not in result[0], "list view must not dump per-run history"
    assert result[0]["last_status"] == "success"
    assert result[0]["run_count"] == 20
