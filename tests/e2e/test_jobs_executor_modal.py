"""The new-job modal exposes an executor dropdown, but only when the daemon
reports a non-agent executor (e.g. cc-driver) at GET /api/executors. The chosen
executor rides along in the /job command submission.
"""

from __future__ import annotations

import json


def _stub_jobs_api(page):
    page.route(
        "**/api/jobs**",
        lambda route: route.fulfill(status=200, content_type="application/json", body=json.dumps({"jobs": []})),
    )


def _stub_executors(page, executors):
    page.route(
        "**/api/executors",
        lambda route: route.fulfill(
            status=200, content_type="application/json", body=json.dumps({"executors": executors})
        ),
    )


def _open_new_job_modal(page, executors):
    _stub_executors(page, executors)
    page.evaluate("Alpine.store('app').view = 'jobs'")
    page.wait_for_function("Alpine.store('app').view === 'jobs'", timeout=3000)
    page.wait_for_selector(".jobs-newbtn", timeout=5000)
    page.locator(".jobs-newbtn").first.click()
    page.wait_for_selector(".tsu-modal-backdrop.new-job-modal", state="visible", timeout=3000)


def test_executor_dropdown_shown_when_executor_registered(authenticated_page):
    page = authenticated_page
    _stub_jobs_api(page)
    _open_new_job_modal(page, ["agent", "cc"])

    dropdown = page.locator('.new-job-modal [data-testid="nj-executor"]')
    dropdown.wait_for(state="visible", timeout=3000)
    options = dropdown.locator("option").all_text_contents()
    assert "agent" in options
    assert "cc" in options


def test_executor_dropdown_hidden_without_registered_executors(authenticated_page):
    page = authenticated_page
    _stub_jobs_api(page)
    _open_new_job_modal(page, ["agent"])

    assert page.locator('.new-job-modal [data-testid="nj-executor"]').count() == 0


def test_new_job_submits_selected_executor(authenticated_page):
    page = authenticated_page
    _stub_jobs_api(page)

    captured = {"body": None}

    def handle_command(route, request):
        captured["body"] = request.post_data_json
        route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({"result": "Job job-cc spawned (executor: cc)"}),
        )

    page.route("**/api/agents/*/commands/job", handle_command)

    _open_new_job_modal(page, ["agent", "cc"])
    page.locator(".new-job-modal textarea.nj-grow").fill("drive claude to green the build")
    page.locator('.new-job-modal [data-testid="nj-executor"]').select_option("cc")
    page.locator(".new-job-modal button.tsu-btn.--primary").click()

    page.wait_for_function(
        "() => { const el = document.querySelector('.tsu-modal-backdrop.new-job-modal');"
        " return !el || el.style.display === 'none'; }",
        timeout=3000,
    )

    body = captured["body"] or {}
    assert body.get("executor") == "cc", f"executor must be submitted, got {body!r}"
    assert body.get("prompt") == "drive claude to green the build"
