"""Schedule-session retention runs through the backend (delete_session), not file globs."""

import pytest
from tsugite_daemon.scheduler import Scheduler

from tsugite.history import get_history_backend
from tsugite.history.sqlite_conn import close_all


@pytest.fixture
def backend():
    b = get_history_backend()  # sqlite default, XDG-isolated per test
    yield b
    close_all()


def test_delete_session_removes_row_and_events(backend):
    s = backend.create("chat", "m", session_id="to-delete")
    s.record("user_input", text="x")
    assert backend.delete_session("to-delete") is True
    assert not backend.exists("to-delete")
    assert backend.delete_session("to-delete") is False  # already gone


def test_cleanup_history_keeps_newest_per_schedule(backend, tmp_path):
    ids = []
    for i in range(4):
        s = backend.create("sched", "m", session_id=f"sched_test_{i}")
        s.record("user_input", text=f"run {i}")
        ids.append(s.session_id)

    scheduler = Scheduler(tmp_path / "schedules.json", lambda *a, **k: None)
    scheduler._schedules = {"test": None}  # only the keys are read (active prefix "sched_test_")

    removed = scheduler.cleanup_history(max_age_days=3650, max_files_per_schedule=2)

    assert removed == 2  # two oldest runs dropped
    surviving = [i for i in ids if backend.exists(i)]
    assert len(surviving) == 2
