"""E2E tests for the Jobs top-level tab and its new-job GUI modal.

Stubs out /api/jobs (list endpoint) and /api/agents/<a>/commands/job so the
frontend can exercise board/table layout switching, filter pills, the new-job
modal's live command preview, and mobile collapse without a real orchestrator.
"""

from __future__ import annotations

import json


def _make_jobs(*, with_stuck=True):
    """Realistic spread across all states the board groups into 4 columns."""
    jobs = [
        {
            "job_id": "job-r1",
            "parent_session_id": "parent-aaaaaa",
            "worker_session_id": "worker-aaaaaa",
            "verifier_session_id": None,
            "state": "running",
            "prompt": "Implement issue 287 inline tile spacing fix",
            "verify_attempts": 0,
            "error": None,
            "attempts": [],
            "acceptance_criteria": ["Tile padding matches token"],
            "agent": "odyn",
            "model": None,
            "created_at": "2026-05-30T10:00:00Z",
            "updated_at": "2026-05-30T10:02:14Z",
            "resolved_at": None,
            "spawned_by": "user-slash",
        },
        {
            "job_id": "job-v1",
            "parent_session_id": "parent-bbbbbb",
            "worker_session_id": "worker-bbbbbb",
            "verifier_session_id": "verifier-bbbbbb",
            "state": "verifying",
            "prompt": "Add retry-with-backoff to webhook sender",
            "verify_attempts": 1,
            "error": None,
            "attempts": [{"index": 0, "kind": "initial"}],
            "acceptance_criteria": ["Retries 3x", "Gives up after max", "No retry on 4xx"],
            "agent": "odyn",
            "model": None,
            "created_at": "2026-05-30T09:58:00Z",
            "updated_at": "2026-05-30T10:02:02Z",
            "resolved_at": None,
            "spawned_by": "user-slash",
        },
        {
            "job_id": "job-q1",
            "parent_session_id": "parent-cccccc",
            "worker_session_id": None,
            "verifier_session_id": None,
            "state": "queued",
            "prompt": "Add a dark-mode toggle to the settings pane",
            "verify_attempts": 0,
            "error": None,
            "attempts": [],
            "acceptance_criteria": ["Toggle persists", "Respects prefers-color-scheme"],
            "agent": "assistant",
            "model": None,
            "created_at": "2026-05-30T10:01:40Z",
            "updated_at": "2026-05-30T10:01:40Z",
            "resolved_at": None,
            "spawned_by": "user-slash",
        },
        {
            "job_id": "job-d1",
            "parent_session_id": "parent-dddddd",
            "worker_session_id": "worker-dddddd",
            "verifier_session_id": "verifier-dddddd",
            "state": "done",
            "prompt": "Write me a haiku about parallel processes",
            "verify_attempts": 1,
            "error": None,
            "attempts": [{"index": 0, "kind": "initial", "verifier_pass": True}],
            "acceptance_criteria": ["5-7-5 syllables", "References concurrency"],
            "agent": "assistant",
            "model": None,
            "created_at": "2026-05-30T09:55:00Z",
            "updated_at": "2026-05-30T09:55:18Z",
            "resolved_at": "2026-05-30T09:55:18Z",
            "spawned_by": "user-slash",
        },
        {
            "job_id": "job-c1",
            "parent_session_id": "parent-eeeeee",
            "worker_session_id": "worker-eeeeee",
            "verifier_session_id": None,
            "state": "cancelled",
            "prompt": "Generate 200 test fixtures",
            "verify_attempts": 0,
            "error": "cancelled by user",
            "attempts": [],
            "acceptance_criteria": ["200 valid fixtures"],
            "agent": "assistant",
            "model": None,
            "created_at": "2026-05-30T09:50:00Z",
            "updated_at": "2026-05-30T09:51:40Z",
            "resolved_at": "2026-05-30T09:51:40Z",
            "spawned_by": "user-slash",
        },
    ]
    if with_stuck:
        jobs.extend(
            [
                {
                    "job_id": "job-s1",
                    "parent_session_id": "parent-ffffff",
                    "worker_session_id": "worker-ffffff",
                    "verifier_session_id": "verifier-ffffff",
                    "state": "stuck",
                    "prompt": "Refactor the agent loop",
                    "verify_attempts": 3,
                    "error": "verifier rejected 3 attempts",
                    "attempts": [
                        {"index": 0, "kind": "initial", "verifier_pass": False},
                        {"index": 1, "kind": "retry", "verifier_pass": False},
                        {"index": 2, "kind": "retry", "verifier_pass": False},
                    ],
                    "acceptance_criteria": ["All existing tests pass", "Adds nested-call tests", "No new dependencies"],
                    "agent": "odyn",
                    "model": None,
                    "created_at": "2026-05-30T09:40:00Z",
                    "updated_at": "2026-05-30T10:03:00Z",
                    "resolved_at": "2026-05-30T10:03:00Z",
                    "spawned_by": "user-slash",
                },
                {
                    "job_id": "job-e1",
                    "parent_session_id": "parent-gggggg",
                    "worker_session_id": "worker-gggggg",
                    "verifier_session_id": None,
                    "state": "errored",
                    "prompt": "Migrate workspace DB to v3",
                    "verify_attempts": 0,
                    "error": "database is locked",
                    "attempts": [],
                    "acceptance_criteria": ["Migration is reversible", "Existing rows preserved"],
                    "agent": "odyn",
                    "model": None,
                    "created_at": "2026-05-30T09:55:00Z",
                    "updated_at": "2026-05-30T10:01:30Z",
                    "resolved_at": "2026-05-30T10:01:30Z",
                    "spawned_by": "user-slash",
                },
            ]
        )
    return jobs


def _stub_jobs_api(page, jobs):
    """Intercept GET /api/jobs (with optional ?state= filter) using a deterministic fixture."""

    def handle(route, request):
        # Reuse the same fixture for filtered + unfiltered calls. The frontend
        # /api/jobs?state=stuck call from the badge syncer uses the alias.
        url = request.url
        # Crude query-string parse: testing the route handler precision is the
        # backend's job (see tests/daemon/test_jobs_list_endpoint.py).
        filtered = list(jobs)
        if "state=stuck" in url:
            filtered = [j for j in jobs if j["state"] in {"stuck", "errored"}]
        elif "state=active" in url:
            filtered = [j for j in jobs if j["state"] in {"running", "verifying"}]
        elif "state=resolved" in url:
            filtered = [j for j in jobs if j["state"] in {"done", "cancelled"}]
        elif "state=queued" in url:
            filtered = [j for j in jobs if j["state"] == "queued"]
        route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({"jobs": filtered}),
        )

    page.route("**/api/jobs**", handle)


def _open_jobs_tab(page):
    """Switch to the Jobs tab, then force the jobsView to re-load through the stub.

    Stub routes are installed after the page boots (the authenticated_page fixture
    already navigated), so the view's initial load() ran against the real (503)
    endpoint. We re-trigger it via $data so the stub fixture takes effect.
    """
    page.evaluate("Alpine.store('app').view = 'jobs'")
    page.wait_for_function("Alpine.store('app').view === 'jobs'", timeout=3000)
    # Give Alpine a tick to mount the x-data scope, then drive a fresh load().
    page.wait_for_function(
        "() => !!document.querySelector('[x-data*=jobsView]')",
        timeout=3000,
    )
    page.evaluate(
        "(async () => { const v = Alpine.$data(document.querySelector('[x-data*=jobsView]')); await v.load(); })()"
    )


def test_jobs_tab_renders_in_top_bar(authenticated_page):
    page = authenticated_page
    tab = page.locator(".console-tabs button.console-tab", has_text="jobs").first
    tab.wait_for(state="visible", timeout=3000)
    assert "jobs" in (tab.text_content() or "").lower()


def test_needs_you_badge_shows_count_from_api(authenticated_page):
    page = authenticated_page
    _stub_jobs_api(page, _make_jobs(with_stuck=True))
    # Force a badge refresh now that the stub is in place.
    page.evaluate("window.tsugiteLoadJobsNeedsYou()")
    page.wait_for_function("Alpine.store('app').jobsNeedsYou === 2", timeout=3000)
    badge = page.locator(".console-tabs button.console-tab", has_text="jobs").locator(".badge")
    badge.wait_for(state="visible", timeout=3000)
    assert (badge.text_content() or "").strip() == "2"


def test_needs_you_badge_hidden_when_zero(authenticated_page):
    page = authenticated_page
    _stub_jobs_api(page, _make_jobs(with_stuck=False))
    page.evaluate("window.tsugiteLoadJobsNeedsYou()")
    page.wait_for_function("Alpine.store('app').jobsNeedsYou === 0", timeout=3000)
    badge = page.locator(".console-tabs button.console-tab", has_text="jobs").locator(".badge")
    assert badge.count() == 0


def test_board_view_shows_four_columns(authenticated_page):
    page = authenticated_page
    _stub_jobs_api(page, _make_jobs())
    _open_jobs_tab(page)

    page.wait_for_selector(".jb-board .jb-col", timeout=5000)
    cols = page.locator(".jb-board .jb-col")
    assert cols.count() == 4

    head_text = page.locator(".jb-board .jb-col-head").all_text_contents()
    joined = " ".join(head_text).lower()
    assert "active" in joined
    assert "queued" in joined
    assert "needs you" in joined
    assert "resolved" in joined


def test_board_groups_jobs_into_correct_columns(authenticated_page):
    page = authenticated_page
    _stub_jobs_api(page, _make_jobs())
    _open_jobs_tab(page)

    page.wait_for_selector(".jb-board .jb-col", timeout=5000)
    cols = page.locator(".jb-board .jb-col")
    # Order matches the boardColumns order in jobs.js: active, queued, needs, done.
    active_col = cols.nth(0)
    queued_col = cols.nth(1)
    needs_col = cols.nth(2)
    done_col = cols.nth(3)

    # active: running + verifying = 2
    assert "2" in (active_col.locator(".jb-col-head .n").text_content() or "")
    # queued: 1
    assert "1" in (queued_col.locator(".jb-col-head .n").text_content() or "")
    # needs you: stuck + errored = 2
    assert "2" in (needs_col.locator(".jb-col-head .n").text_content() or "")
    # resolved: done + cancelled = 2
    assert "2" in (done_col.locator(".jb-col-head .n").text_content() or "")


def test_layout_toggle_switches_board_to_table(authenticated_page):
    page = authenticated_page
    _stub_jobs_api(page, _make_jobs())
    _open_jobs_tab(page)

    page.wait_for_selector(".jb-board", timeout=5000)
    assert page.locator(".jobs-table").count() == 0

    page.locator(".jobs-layout-toggle button", has_text="table").click()
    page.wait_for_selector(".jobs-table", timeout=3000)
    assert page.locator(".jb-board").count() == 0

    # Toggling back to board hides the table.
    page.locator(".jobs-layout-toggle button", has_text="board").click()
    page.wait_for_selector(".jb-board", timeout=3000)
    assert page.locator(".jobs-table").count() == 0


def test_summary_pill_click_filters_list(authenticated_page):
    page = authenticated_page
    _stub_jobs_api(page, _make_jobs())
    _open_jobs_tab(page)

    page.wait_for_selector(".js-pill", timeout=5000)
    # Click the "needs you" pill - should leave only the 2 stuck/errored cards visible.
    page.locator(".js-pill", has_text="needs you").click()

    page.wait_for_function(
        "() => { const cards = document.querySelectorAll('.jb-board .jb-col-body .jb'); return cards.length === 2;}",
        timeout=3000,
    )


def test_empty_state_teaches_slash_command(authenticated_page):
    page = authenticated_page
    _stub_jobs_api(page, [])
    _open_jobs_tab(page)

    page.wait_for_selector(".jobs-empty", timeout=5000)
    body = page.locator(".jobs-empty").text_content() or ""
    assert "no jobs" in body.lower()
    assert "/job" in body


def test_new_job_modal_opens_and_shows_command_preview(authenticated_page):
    page = authenticated_page
    _stub_jobs_api(page, _make_jobs())
    _open_jobs_tab(page)
    page.wait_for_selector(".jobs-newbtn", timeout=5000)

    page.locator(".jobs-newbtn").first.click()
    page.wait_for_selector(".tsu-modal-backdrop.new-job-modal", state="visible", timeout=3000)

    # Type a task + an acceptance criterion; the preview must update reactively.
    page.locator(".new-job-modal textarea.nj-grow").fill("implement the dark-mode toggle")
    page.locator(".new-job-modal .nj-ac-add").click()
    page.locator(".new-job-modal .nj-ac-row input[type='text']").last.fill("toggle persists")

    preview = page.locator('[data-testid="nj-command-preview"]')
    preview_text = preview.text_content() or ""
    assert "/job" in preview_text
    assert "--max-attempts 3" in preview_text
    # --verify / --fg were dead preview flags cmd_job never accepted; the
    # preview must only render flags that actually work when pasted.
    assert "--verify" not in preview_text
    assert "--fg" not in preview_text
    assert "--ac" in preview_text
    assert "toggle persists" in preview_text
    assert "implement the dark-mode toggle" in preview_text


def test_new_job_modal_submits_to_command_endpoint(authenticated_page):
    page = authenticated_page
    _stub_jobs_api(page, _make_jobs())

    captured = {"called": False, "body": None, "url": None}

    def handle_command(route, request):
        captured["called"] = True
        captured["url"] = request.url
        captured["body"] = request.post_data_json
        route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({"result": "Job job-newone spawned"}),
        )

    page.route("**/api/agents/*/commands/job", handle_command)

    _open_jobs_tab(page)
    page.wait_for_selector(".jobs-newbtn", timeout=5000)
    page.locator(".jobs-newbtn").first.click()
    page.wait_for_selector(".tsu-modal-backdrop.new-job-modal", state="visible", timeout=3000)

    page.locator(".new-job-modal textarea.nj-grow").fill("draft release notes")
    page.locator(".new-job-modal .nj-ac-add").click()
    page.locator(".new-job-modal .nj-ac-row input[type='text']").last.fill("groups by heading")

    page.locator(".new-job-modal button.tsu-btn.--primary").click()

    page.wait_for_function("(() => true)()", timeout=2000)
    page.wait_for_function(
        "() => { const el = document.querySelector('.tsu-modal-backdrop.new-job-modal');"
        " return !el || el.style.display === 'none'; }",
        timeout=3000,
    )

    assert captured["called"], "POST to /api/agents/<a>/commands/job was never made"
    assert captured["url"] and "/commands/job" in captured["url"]
    body = captured["body"] or {}
    assert body.get("prompt") == "draft release notes"
    assert "acceptance_criteria" in body
    assert "groups by heading" in body["acceptance_criteria"]
    assert body.get("user_id"), "user_id must be passed for slash-command auth"


def test_mobile_board_collapses_to_single_column(authenticated_page):
    """At <=640px the board grid collapses to a single column (per jobs-tab.css)."""
    page = authenticated_page
    _stub_jobs_api(page, _make_jobs())
    _open_jobs_tab(page)
    page.wait_for_selector(".jb-board .jb-col", timeout=5000)

    page.set_viewport_size({"width": 390, "height": 800})

    # The grid-template-columns on .jb-board switches to '1fr' at the mobile breakpoint.
    template = page.evaluate("() => getComputedStyle(document.querySelector('.jb-board')).gridTemplateColumns")
    # In single-column mode the value should be just one length token, not four.
    parts = (template or "").split()
    assert len(parts) == 1, f"expected single column on mobile, got grid-template-columns={template!r}"
