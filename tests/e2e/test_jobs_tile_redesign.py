"""E2E tests for the redesigned job tile (Direction B), dialogs, and notify turn.

Covers:
  - All 7 states (queued / running / verifying / done / stuck / errored / cancelled)
    render the correct accent color via data-state + state-glyph label.
  - Collapse/expand toggles the body.
  - AC chips render with per-criterion verdicts when `acceptance_criteria` is in the payload.
  - Attempts expander shows when there's > 1 attempt.
  - Retry-with-hint dialog opens, takes a hint + suggestion chip, submits a POST.
  - Mark-done dialog opens, lists criteria with pass/fail, submits a POST.
  - Notify message (job_notify): a job-finished wake-up renders with the job
    gutter chip (state-tinted) instead of a plain user bubble.
"""

import json
from unittest.mock import patch

from tsugite.history.storage import SessionStorage

JOB_STATES = ["queued", "running", "verifying", "done", "stuck", "errored", "cancelled"]


def _new_session(e2e_adapter, e2e_tmp, user_id, tag):
    session = e2e_adapter.session_store.get_or_create_interactive(user_id, "test-agent")
    history_dir = e2e_tmp / f"history-jobs-{tag}"
    history_dir.mkdir(exist_ok=True)
    session_path = history_dir / f"{session.id}.jsonl"
    if session_path.exists():
        session_path.unlink()
    storage = SessionStorage.create("test-agent", model="test", session_path=session_path)
    return history_dir, session, storage


def _open_conv(page):
    page.locator(".console-tabs button.console-tab", has_text="Conversations").click()
    page.wait_for_function("Alpine.store('app').view === 'conversations'", timeout=3000)


def test_redesigned_tile_renders_all_seven_states(authenticated_page, e2e_adapter, e2e_tmp):
    page = authenticated_page
    _open_conv(page)
    user_id = page.evaluate("Alpine.store('app').userId")
    history_dir, session, storage = _new_session(e2e_adapter, e2e_tmp, user_id, "all-states")
    storage.record("user_input", text="/job seed batch of tiles")
    for i, state in enumerate(JOB_STATES):
        storage.record(
            "job_status",
            job_id=f"job-state{i:02x}",
            parent_session_id=session.id,
            state=state,
            prompt=f"prompt for {state}",
            verify_attempts=0,
        )

    with patch("tsugite.history.storage.get_history_dir", return_value=history_dir):
        page.reload()
        page.wait_for_selector(".jx", timeout=5000)
        tiles = page.locator(".jx")
        assert tiles.count() == len(JOB_STATES), f"Expected {len(JOB_STATES)} tiles, got {tiles.count()}"
        states_rendered = {tiles.nth(i).get_attribute("data-state") for i in range(tiles.count())}
        assert states_rendered == set(JOB_STATES), f"All seven states must render; got {sorted(states_rendered)}"


def test_tile_collapse_expand_toggle(authenticated_page, e2e_adapter, e2e_tmp):
    page = authenticated_page
    _open_conv(page)
    user_id = page.evaluate("Alpine.store('app').userId")
    history_dir, session, storage = _new_session(e2e_adapter, e2e_tmp, user_id, "toggle")
    storage.record("user_input", text="/job done thing")
    # done state default-collapses, so the body is hidden initially.
    storage.record(
        "job_status",
        job_id="job-toggle01",
        parent_session_id=session.id,
        state="done",
        prompt="done thing",
        worker_session_id="worker-tg",
    )

    with patch("tsugite.history.storage.get_history_dir", return_value=history_dir):
        page.reload()
        page.wait_for_selector(".jx", timeout=5000)
        tile = page.locator(".jx").first
        # Body hidden initially (done default-collapses), header visible.
        assert tile.locator(".jx-head").is_visible()
        assert tile.locator(".jx-body").count() == 0
        # Click header to expand.
        tile.locator(".jx-head").click()
        page.wait_for_selector(".jx .jx-body", timeout=2000)
        assert tile.locator(".jx-body").is_visible()


def test_ac_chips_render_with_per_criterion_verdicts(authenticated_page, e2e_adapter, e2e_tmp):
    page = authenticated_page
    _open_conv(page)
    user_id = page.evaluate("Alpine.store('app').userId")
    history_dir, session, storage = _new_session(e2e_adapter, e2e_tmp, user_id, "acs")
    storage.record("user_input", text="/job ac test")
    storage.record(
        "job_status",
        job_id="job-ac001",
        parent_session_id=session.id,
        state="stuck",  # stuck default-expands so chips are visible
        prompt="thing with criteria",
        verify_attempts=2,
        acceptance_criteria=[
            "All existing tests still pass",
            "Snapshot at 375px viewport matches",
            "Shell command exits 0",
            "Output is a valid haiku",
        ],
        result={
            "ac_results": [
                {"ac_text": "All existing tests still pass", "pass": False, "reason": "2 failures"},
                {"ac_text": "Snapshot at 375px viewport matches", "pass": True, "reason": "ok"},
            ],
        },
    )

    with patch("tsugite.history.storage.get_history_dir", return_value=history_dir):
        page.reload()
        page.wait_for_selector(".jx .jt-acs", timeout=5000)
        chips = page.locator(".jx .jt-ac")
        assert chips.count() == 4
        # Chip labels show the criterion text (not "[object Object]").
        labels = [chips.nth(i).text_content() for i in range(chips.count())]
        assert any("All existing tests still pass" in t for t in labels)
        assert not any("[object Object]" in t for t in labels)
        # Pass + fail markers present.
        assert page.locator(".jx .jt-ac.pass").count() == 1
        assert page.locator(".jx .jt-ac.fail").count() == 1


def test_attempts_expander_shows_when_multiple(authenticated_page, e2e_adapter, e2e_tmp):
    page = authenticated_page
    _open_conv(page)
    user_id = page.evaluate("Alpine.store('app').userId")
    history_dir, session, storage = _new_session(e2e_adapter, e2e_tmp, user_id, "attempts")
    storage.record("user_input", text="/job retry stuff")
    storage.record(
        "job_status",
        job_id="job-att001",
        parent_session_id=session.id,
        state="stuck",
        prompt="retry stuff",
        verify_attempts=2,
        attempts=[
            {
                "index": 0,
                "kind": "initial",
                "worker_session_id": "w1",
                "verifier_session_id": "v1",
                "verifier_pass": False,
            },
            {
                "index": 1,
                "kind": "retry",
                "worker_session_id": "w2",
                "verifier_session_id": "v2",
                "verifier_pass": False,
            },
            {
                "index": 2,
                "kind": "retry",
                "worker_session_id": "w3",
                "verifier_session_id": "v3",
                "verifier_pass": False,
            },
        ],
    )

    with patch("tsugite.history.storage.get_history_dir", return_value=history_dir):
        page.reload()
        page.wait_for_selector(".jx .jt-attempts", timeout=5000)
        rows = page.locator(".jx .jt-attempt")
        assert rows.count() == 3
        # Last attempt is the current one (cur class).
        assert page.locator(".jx .jt-attempt.cur").count() == 1


def test_no_terminal_for_job_without_worker(authenticated_page, e2e_adapter, e2e_tmp):
    """LLM-only jobs (no worker_session_id) don't render the terminal embed."""
    page = authenticated_page
    _open_conv(page)
    user_id = page.evaluate("Alpine.store('app').userId")
    history_dir, session, storage = _new_session(e2e_adapter, e2e_tmp, user_id, "noterm")
    storage.record("user_input", text="/job haiku")
    storage.record(
        "job_status",
        job_id="job-haiku01",
        parent_session_id=session.id,
        state="stuck",  # expanded
        prompt="write a haiku",
        verify_attempts=0,
        # No worker_session_id → tile renders without terminal embed.
    )

    with patch("tsugite.history.storage.get_history_dir", return_value=history_dir):
        page.reload()
        page.wait_for_selector(".jx", timeout=5000)
        # The terminal pane is conditionally rendered only when worker_session_id is set.
        assert page.locator(".jx [data-job-terminal]").count() == 0


def test_retry_with_hint_dialog_opens_and_submits(authenticated_page, e2e_adapter, e2e_tmp, base_url):
    page = authenticated_page
    _open_conv(page)
    user_id = page.evaluate("Alpine.store('app').userId")
    history_dir, session, storage = _new_session(e2e_adapter, e2e_tmp, user_id, "retrydlg")
    storage.record("user_input", text="/job retry test")
    storage.record(
        "job_status",
        job_id="job-retry01",
        parent_session_id=session.id,
        state="stuck",
        prompt="retry test target",
        verify_attempts=2,
        error="Verifier rejected attempt 2 - assertion mismatch.",
    )

    # Intercept the POST so we don't need a running orchestrator.
    posted = {"body": None, "called": False}

    def _handle(route):
        posted["called"] = True
        try:
            posted["body"] = route.request.post_data_json
        except Exception:
            posted["body"] = route.request.post_data
        route.fulfill(status=200, body=json.dumps({"status": "running"}), content_type="application/json")

    page.route("**/api/jobs/job-retry01/retry", _handle)

    with patch("tsugite.history.storage.get_history_dir", return_value=history_dir):
        page.reload()
        page.wait_for_selector(".jx", timeout=5000)
        # Click the "retry with hint…" button in the tile footer.
        page.locator(".jx .jt-btn.primary", has_text="retry").click()
        page.wait_for_selector(".tsu-modal-backdrop.retry-hint-modal", state="visible", timeout=2000)
        # Dialog is visible with verdict note.
        dlg = page.locator(".tsu-modal-backdrop.retry-hint-modal .tsu-modal")
        assert "Verifier rejected" in (dlg.text_content() or "")
        # Type a hint.
        page.locator(".tsu-modal-backdrop.retry-hint-modal textarea#retry-hint-text").fill(
            "try again with a smaller diff"
        )
        # Submit.
        page.locator(".tsu-modal-backdrop.retry-hint-modal button.tsu-btn.--primary").click()
        page.wait_for_function(
            "() => { const el = document.querySelector('.tsu-modal-backdrop.retry-hint-modal');"
            " return !el || el.style.display === 'none'; }",
            timeout=2000,
        )
        assert posted["called"], "POST /api/jobs/.../retry was not called"
        assert (posted["body"] or {}).get("hint") == "try again with a smaller diff"


def test_mark_done_dialog_opens_and_submits(authenticated_page, e2e_adapter, e2e_tmp, base_url):
    page = authenticated_page
    _open_conv(page)
    user_id = page.evaluate("Alpine.store('app').userId")
    history_dir, session, storage = _new_session(e2e_adapter, e2e_tmp, user_id, "donedlg")
    storage.record("user_input", text="/job markdone test")
    storage.record(
        "job_status",
        job_id="job-md001",
        parent_session_id=session.id,
        state="stuck",
        prompt="markdone test target",
        verify_attempts=3,
        acceptance_criteria=["Tests pass", "Lint passes"],
        result={
            "ac_results": [
                {"ac_text": "Tests pass", "pass": False, "reason": "1 fail"},
                {"ac_text": "Lint passes", "pass": True, "reason": "ok"},
            ],
        },
    )

    posted = {"body": None, "called": False}

    def _handle(route):
        posted["called"] = True
        try:
            posted["body"] = route.request.post_data_json
        except Exception:
            posted["body"] = route.request.post_data
        route.fulfill(status=200, body=json.dumps({"status": "done"}), content_type="application/json")

    page.route("**/api/jobs/job-md001/mark-done", _handle)

    with patch("tsugite.history.storage.get_history_dir", return_value=history_dir):
        page.reload()
        page.wait_for_selector(".jx", timeout=5000)
        # Click "mark done" button (NOT "retry with hint…").
        page.locator(".jx .jt-btn", has_text="mark done").first.click()
        page.wait_for_selector(".tsu-modal-backdrop.mark-done-modal", state="visible", timeout=2000)
        dlg = page.locator(".tsu-modal-backdrop.mark-done-modal .tsu-modal")
        # The recap lists both criteria.
        ac_rows = dlg.locator(".dlg-ac")
        assert ac_rows.count() == 2
        # One fail + one pass class.
        assert dlg.locator(".dlg-ac.fail").count() == 1
        assert dlg.locator(".dlg-ac.pass").count() == 1
        # Add a reason and submit.
        dlg.locator("textarea.mono").fill("flaky test, will fix later")
        dlg.locator("button.tsu-btn.--warn").click()
        page.wait_for_function(
            "() => { const el = document.querySelector('.tsu-modal-backdrop.mark-done-modal');"
            " return !el || el.style.display === 'none'; }",
            timeout=2000,
        )
        assert posted["called"]
        assert (posted["body"] or {}).get("reason") == "flaky test, will fix later"


def test_job_notify_turn_renders_with_state_tinted_gutter(authenticated_page, e2e_adapter, e2e_tmp):
    """A job-finished wake-up message renders as its own turn type (job gutter chip),
    not as a plain user bubble. The gutter chip is tinted per outcome (done/stuck/errored).
    """
    page = authenticated_page
    _open_conv(page)
    user_id = page.evaluate("Alpine.store('app').userId")
    history_dir, session, storage = _new_session(e2e_adapter, e2e_tmp, user_id, "notify")
    # First a real user prompt, then the wake-up the orchestrator posts.
    storage.record("user_input", text="/job write me a haiku about parallel processes")
    storage.record("model_response", raw_content="ok", model="test", provider="test")
    storage.record(
        "user_input",
        text="Job job-8e3bdone finished with state 'done': write me a haiku about parallel processes. Use get_job('job-8e3bdone') for details.",
    )

    with patch("tsugite.history.storage.get_history_dir", return_value=history_dir):
        page.reload()
        # The notify turn renders as a .jn block with the job gutter chip.
        page.wait_for_selector(".jn", timeout=5000)
        notify = page.locator(".jn").first
        assert notify.get_attribute("data-state") == "done"
        assert notify.get_attribute("data-job-id") == "job-8e3bdone"
        # The gutter chip carries the state class for tinting.
        gutter_chips = page.locator(".console-turn-gutter .role.job.done")
        assert gutter_chips.count() >= 1
        # The lead carries the prompt.
        assert "haiku" in (notify.text_content() or "").lower()


def test_job_update_reaches_tile_while_viewing_another_session(authenticated_page, e2e_adapter, e2e_tmp):
    """A job's state updates must reach its parent chat's tile even while the user
    is viewing a DIFFERENT session (e.g. after clicking "open worker"). Regression
    for the tile getting stuck on 'running': updates whose parent_session_id wasn't
    the on-screen session were dropped, and revisits don't reload history."""
    page = authenticated_page
    _open_conv(page)
    user_id = page.evaluate("Alpine.store('app').userId")
    history_dir, session, storage = _new_session(e2e_adapter, e2e_tmp, user_id, "cross")
    storage.record("user_input", text="/job long task")
    storage.record(
        "job_status",
        job_id="job-cross01",
        parent_session_id=session.id,
        state="running",
        prompt="a long-running job",
        worker_session_id="session-worker-zzz",
        verify_attempts=0,
    )

    view_sel = "[x-data*=conversationsView]"
    with patch("tsugite.history.storage.get_history_dir", return_value=history_dir):
        page.reload()
        page.wait_for_selector(".jx", timeout=5000)
        # Open the parent chat so its tile lives in sessionsState.
        page.evaluate(
            f"Alpine.$data(document.querySelector('{view_sel}')).selectSessionById('{session.id}', {{follow:false}})"
        )
        page.wait_for_timeout(500)
        # Navigate away to the worker session (simulate "open worker").
        page.evaluate(f"Alpine.$data(document.querySelector('{view_sel}')).selectedSessionId = 'session-worker-zzz'")
        # Daemon broadcasts the job completing while we're on the worker.
        page.evaluate(
            f"Alpine.store('app').lastEvent = {{ type:'job_update', "
            f"data:{{ job_id:'job-cross01', parent_session_id:'{session.id}', state:'done' }}, _ts: Date.now() }}"
        )
        page.wait_for_timeout(400)
        owner_state = page.evaluate(
            f"(() => {{ const v = Alpine.$data(document.querySelector('{view_sel}')); "
            f"const st = v.sessionsState['{session.id}']; "
            f"const t = st && st.messages.find(m => m.type==='job_status'); return t ? t.state : 'missing'; }})()"
        )
        assert owner_state == "done", f"parent tile must update to done while viewing the worker, got {owner_state!r}"
