"""Frontend rendering of `job_status` tiles persisted in the parent session JSONL.

Selectors target the redesigned tile (.jx) - the old `.job-status` class is
gone. See test_jobs_tile_redesign.py for the full per-state coverage including
collapse/expand, AC chips, attempts expander, dialogs, and the notify turn.
"""

from unittest.mock import patch

from tsugite.history.storage import SessionStorage


def _seed_job_status_events(e2e_adapter, e2e_tmp, user_id, job_id, states):
    """Seed a parent session JSONL with a user_input + sequence of job_status events."""
    session = e2e_adapter.session_store.get_or_create_interactive(user_id, "test-agent")
    history_dir = e2e_tmp / f"history-jobs-{job_id}"
    history_dir.mkdir(exist_ok=True)
    session_path = history_dir / f"{session.id}.jsonl"
    if session_path.exists():
        session_path.unlink()

    storage = SessionStorage.create("test-agent", model="test", session_path=session_path)
    storage.record("user_input", text="/job write a haiku")
    for state in states:
        storage.record(
            "job_status",
            job_id=job_id,
            parent_session_id=session.id,
            worker_session_id="worker-abcdef12",
            state=state,
            prompt="write a haiku about parallel processes",
            verify_attempts=0,
            error=None,
        )
    return history_dir, session


def test_job_status_tile_renders_latest_state(authenticated_page, e2e_adapter, e2e_tmp):
    """Multiple job_status events for the same job_id collapse into one tile showing the latest state."""
    page = authenticated_page
    page.locator(".console-tabs button.console-tab", has_text="Conversations").click()
    page.wait_for_function("Alpine.store('app').view === 'conversations'", timeout=3000)

    user_id = page.evaluate("Alpine.store('app').userId")
    job_id = "job-deadbeef"
    history_dir, _session = _seed_job_status_events(
        e2e_adapter,
        e2e_tmp,
        user_id,
        job_id,
        states=["running", "verifying", "done"],
    )

    with patch("tsugite.daemon.adapters.http.get_history_dir", return_value=history_dir):
        page.reload()
        page.wait_for_selector(".console-turn.user", timeout=5000)
        page.wait_for_selector(".jx", timeout=5000)

        tiles = page.locator(".jx")
        assert tiles.count() == 1, (
            f"Multiple job_status events for the same job_id must coalesce into one tile, got {tiles.count()}"
        )
        tile = tiles.first
        assert tile.get_attribute("data-state") == "done"
        assert tile.get_attribute("data-job-id") == job_id
        # The prompt text appears in either the inline-title (collapsed) or jx-title (expanded).
        text = tile.text_content() or ""
        assert "write a haiku" in text


def test_job_status_tile_shows_error_in_stuck_state(authenticated_page, e2e_adapter, e2e_tmp):
    page = authenticated_page
    page.locator(".console-tabs button.console-tab", has_text="Conversations").click()
    page.wait_for_function("Alpine.store('app').view === 'conversations'", timeout=3000)

    user_id = page.evaluate("Alpine.store('app').userId")
    session = e2e_adapter.session_store.get_or_create_interactive(user_id, "test-agent")
    history_dir = e2e_tmp / "history-jobs-stuck"
    history_dir.mkdir(exist_ok=True)
    session_path = history_dir / f"{session.id}.jsonl"
    if session_path.exists():
        session_path.unlink()
    storage = SessionStorage.create("test-agent", model="test", session_path=session_path)
    storage.record("user_input", text="/job do thing")
    storage.record(
        "job_status",
        job_id="job-stuck001",
        parent_session_id=session.id,
        worker_session_id="worker-x",
        state="stuck",
        prompt="do thing",
        verify_attempts=3,
        error="Verifier failed after max attempts:\n- AC1: nope",
    )

    with patch("tsugite.daemon.adapters.http.get_history_dir", return_value=history_dir):
        page.reload()
        # Stuck tile defaults to expanded so the issue note is visible.
        tile = page.wait_for_selector('.jx[data-state="stuck"]', timeout=5000)
        text = tile.text_content() or ""
        assert "stuck" in text.lower()
        assert "Verifier failed" in text
