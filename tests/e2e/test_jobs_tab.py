"""Jobs view: a seeded job renders on the board and its drawer opens; a job
created through the real structured POST /api/jobs endpoint also shows up.
"""

import uuid

from playwright.sync_api import expect
from tsugite_daemon.job_store import Job

from .helpers import E2E_USER_ID, auth_headers, open_view


def test_seeded_job_renders_and_drawer_opens(authenticated_page, job_store):
    job = Job(
        id=f"e2e-job-{uuid.uuid4().hex[:8]}",
        parent_session_id="e2e-parent-session",
        prompt="Seed job for e2e listing test",
        state="running",
        agent="test-agent",
    )
    job_store.add(job)

    page = authenticated_page
    open_view(page, "jobs")

    card = page.locator(f'[data-testid="job-card-{job.id}"]')
    expect(card).to_be_visible()
    expect(card).to_contain_text(job.prompt)

    card.click()
    expect(page.locator('[data-testid="job-drawer"]')).to_be_visible()


def test_job_created_via_structured_endpoint_appears(
    authenticated_page, base_url, e2e_auth_token, jobs_backend, mock_chat
):
    """POST /api/jobs synchronously persists the Job before returning; the
    worker run it schedules in the background is routed through the mocked
    `handle_message`, so no real LLM/provider is touched.
    """
    mock_chat("Job accepted")

    page = authenticated_page
    resp = page.request.post(
        f"{base_url}/api/jobs",
        headers=auth_headers(e2e_auth_token),
        data={"agent": "test-agent", "user_id": E2E_USER_ID, "task": "Do a small thing"},
    )
    assert resp.ok, resp.text()
    job_id = resp.json()["job_id"]

    open_view(page, "jobs")
    expect(page.locator(f'[data-testid="job-card-{job_id}"]')).to_be_visible()
