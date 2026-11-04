"""skip_hooks on spawn_job and schedules reaches the run's channel metadata."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from tsugite_daemon.adapters.scheduler_adapter import SchedulerAdapter
from tsugite_daemon.scheduler import ScheduleEntry
from tsugite_daemon.session_store import SessionStore


@pytest.mark.parametrize(
    "kwargs, expected",
    [({}, []), ({"skip_hooks": ["post_tool"]}, ["post_tool"]), ({"hooks": False}, ["*"])],
)
def test_spawn_job_forwards_skip_hooks(monkeypatch, kwargs, expected):
    import tsugite.tools.jobs as jobs_tool

    monkeypatch.setattr(jobs_tool, "_jobs_orchestrator", SimpleNamespace(create_and_start_job=None))
    monkeypatch.setattr("tsugite_daemon.session_runner.get_current_session_id", lambda: "parent-1")
    seen: dict = {}

    def fake_call(fn, *args, timeout=30, **kw):
        seen.update(kw)
        return SimpleNamespace(id="job-1", notify_when="never", state="running"), SimpleNamespace(id="s-1")

    monkeypatch.setattr(jobs_tool, "_call", fake_call)

    jobs_tool.spawn_job(prompt="task", **kwargs)

    assert seen["skip_hooks"] == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("skip_hooks, present", [(["pre_context_build"], True), ([], False)])
async def test_schedule_run_passes_skip_hooks(tmp_path, skip_hooks, present):
    adapter = MagicMock()
    adapter.session_store = SessionStore(tmp_path / "store.json")
    adapter.handle_message = AsyncMock(return_value="ok")
    sched = SchedulerAdapter(adapter=adapter, schedules_path=tmp_path / "schedules.json")
    entry = ScheduleEntry(
        id="s1", prompt="go", schedule_type="once", run_at="2099-01-01T00:00:00Z", skip_hooks=skip_hooks
    )

    await sched._run_agent(entry)

    metadata = adapter.handle_message.call_args.kwargs["channel_context"].metadata
    assert metadata.get("skip_hooks") == (skip_hooks if present else None)
