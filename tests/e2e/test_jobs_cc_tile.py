"""A cc-executor job tile embeds its worker terminal via `worker_terminal_id`
alone - it has no `worker_session_id` (no tsugite worker Session is spawned; the
PTY-backed claude process is the worker). The tile must still show the embedded
terminal, and an llm-only job (neither id) must still show none.
"""

import json

from tsugite.history.storage import SessionStorage


def _seed_job(
    e2e_adapter, e2e_tmp, user_id, job_id, *, worker_terminal_id=None, worker_session_id=None, executor="agent"
):
    session = e2e_adapter.session_store.get_or_create_interactive(user_id, "test-agent")
    history_dir = e2e_tmp / f"history-cc-{job_id}"
    history_dir.mkdir(exist_ok=True)
    session_path = history_dir / f"{session.id}.jsonl"
    if session_path.exists():
        session_path.unlink()
    storage = SessionStorage.create("test-agent", model="test", session_path=session_path)
    storage.record("user_input", text="/job build the thing")
    storage.record(
        "job_status",
        job_id=job_id,
        parent_session_id=session.id,
        worker_session_id=worker_session_id,
        worker_terminal_id=worker_terminal_id,
        executor=executor,
        state="running",
        prompt="build the thing",
        verify_attempts=0,
        error=None,
    )
    return history_dir


def _stub_terminal_stream(page):
    def handle_stream(route, request):
        route.fulfill(
            status=200,
            headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache"},
            body=f"event: output\ndata: {json.dumps({'chunk': 'claude> working...'})}\n\n",
        )

    page.route("**/api/terminals/*/stream", handle_stream)


def test_cc_job_tile_embeds_terminal_via_worker_terminal_id(authenticated_page, e2e_adapter, e2e_tmp):
    from unittest.mock import patch

    page = authenticated_page
    _stub_terminal_stream(page)
    page.locator(".console-tabs button.console-tab", has_text="Conversations").click()
    page.wait_for_function("Alpine.store('app').view === 'conversations'", timeout=3000)

    user_id = page.evaluate("Alpine.store('app').userId")
    history_dir = _seed_job(e2e_adapter, e2e_tmp, user_id, "job-cc0001", worker_terminal_id="term-cc01", executor="cc")

    with patch("tsugite.history.storage.get_history_dir", return_value=history_dir):
        page.reload()
        page.wait_for_selector(".jx", timeout=5000)
        # The cc job has no worker_session_id, only worker_terminal_id, yet its
        # tile must still mount the terminal embed.
        page.wait_for_selector("[data-job-terminal]", state="attached", timeout=5000)


def test_llm_only_job_tile_has_no_terminal(authenticated_page, e2e_adapter, e2e_tmp):
    from unittest.mock import patch

    page = authenticated_page
    page.locator(".console-tabs button.console-tab", has_text="Conversations").click()
    page.wait_for_function("Alpine.store('app').view === 'conversations'", timeout=3000)

    user_id = page.evaluate("Alpine.store('app').userId")
    history_dir = _seed_job(e2e_adapter, e2e_tmp, user_id, "job-llm001", executor="agent")

    with patch("tsugite.history.storage.get_history_dir", return_value=history_dir):
        page.reload()
        page.wait_for_selector(".jx", timeout=5000)
        assert page.locator("[data-job-terminal]").count() == 0
