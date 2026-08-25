"""Schedules view: a seeded schedule row renders and the enable toggle
persists server-side (survives a reload, not just an optimistic UI flip)."""

from playwright.sync_api import expect
from tsugite_daemon.scheduler import ScheduleEntry

from .helpers import open_view, wait_for_authed


def test_seeded_schedule_renders_and_toggle_persists(authenticated_page, scheduler_backend):
    entry = ScheduleEntry(
        id="e2e-schedule",
        prompt="Say hello every morning",
        schedule_type="cron",
        cron_expr="0 9 * * *",
        enabled=False,
    )
    scheduler_backend.add(entry)

    page = authenticated_page
    open_view(page, "schedules")

    row = page.locator('[data-testid="schedule-row-e2e-schedule"]')
    expect(row).to_be_visible()
    expect(row).to_contain_text("e2e-schedule")

    toggle = page.locator('[data-testid="schedule-toggle-e2e-schedule"]')
    expect(toggle).to_have_attribute("aria-checked", "false")
    toggle.click()
    expect(toggle).to_have_attribute("aria-checked", "true")

    page.reload()
    wait_for_authed(page)
    open_view(page, "schedules")
    expect(page.locator('[data-testid="schedule-toggle-e2e-schedule"]')).to_have_attribute("aria-checked", "true")
