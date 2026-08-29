"""Schedules view: a seeded schedule row renders and the enable toggle
persists server-side (survives a reload, not just an optimistic UI flip)."""

import os

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


def test_a_scheduled_message_keeps_its_type_in_the_edit_drawer(authenticated_page, scheduler_backend):
    """The drawer seeds its form from the row and writes execution_type back, so a
    type it cannot show is a type it would rewrite to `agent`."""
    scheduler_backend.add(
        ScheduleEntry(
            id="e2e-reminder",
            prompt="It has been 2 hours. Check on the job.",
            schedule_type="once",
            run_at="2099-01-01T00:00:00+00:00",
            execution_type="session_message",
            target_session="e2e-chat",
        )
    )

    page = authenticated_page
    open_view(page, "schedules")

    row = page.locator('[data-testid="schedule-row-e2e-reminder"]')
    expect(row).to_contain_text("It has been 2 hours")
    row.click()

    expect(page.locator('[data-testid="schedule-form"]')).to_be_visible()
    selected = page.locator('button[data-seg][aria-pressed="true"]', has_text="session_message")
    if shot := os.environ.get("TSU_SHOT"):
        page.screenshot(path=shot, full_page=True)
    expect(selected).to_be_visible()
