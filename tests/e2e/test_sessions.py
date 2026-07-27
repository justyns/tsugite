"""Session rail: seeded sessions render, selection switches the open
conversation, and rename via the session menu round-trips to the server."""

import re

from playwright.sync_api import expect
from tsugite_daemon.session_store import Session, SessionSource

from tsugite.history.storage import generate_session_id

from .helpers import E2E_USER_ID, wait_for_authed


def _seed_session(store, title: str) -> Session:
    session = Session(
        id=generate_session_id("test-agent"),
        agent="test-agent",
        source=SessionSource.INTERACTIVE.value,
        user_id=E2E_USER_ID,
        title=title,
    )
    store.create_session(session)
    return session


def test_sessions_render_select_and_rename(authenticated_page, e2e_session_store):
    page = authenticated_page

    _seed_session(e2e_session_store, "First session")
    _seed_session(e2e_session_store, "Second session")

    page.reload()
    wait_for_authed(page)
    page.wait_for_selector('[data-testid="chat-session-menu-trigger"]', timeout=5000)

    # Scoped to the rail: the conversation header's own title button also
    # matches on text (whichever session got auto-selected on load).
    rail = page.locator('[data-testid="chat-rail"]')
    first_row = rail.get_by_role("button", name=re.compile(re.escape("First session")))
    second_row = rail.get_by_role("button", name=re.compile(re.escape("Second session")))
    expect(first_row).to_be_visible()
    expect(second_row).to_be_visible()

    header_title = page.locator('[data-testid="chat-conversation"] .title-btn')

    # Click each row in turn - proves selection follows the click regardless
    # of which one the view auto-selected on load.
    first_row.click()
    expect(header_title).to_have_text("First session")

    second_row.click()
    expect(header_title).to_have_text("Second session")

    # Rename the now-open session via the session menu (not the inline title
    # click) - the dots menu -> "Rename" affordance shares the same inline
    # header edit field.
    page.locator('[data-testid="chat-session-menu-trigger"]').click()
    page.locator('[data-testid="chat-session-menu"]').get_by_role("menuitem", name="Rename").click()

    rename_input = page.get_by_label("Rename session")
    rename_input.fill("Renamed session")
    rename_input.press("Enter")

    expect(header_title).to_have_text("Renamed session")
    # The rail row updates too (sessions.rename() patches the store optimistically
    # after the PATCH resolves), not just the header.
    expect(rail.get_by_role("button", name=re.compile(re.escape("Renamed session")))).to_be_visible()


def test_browser_back_forward_walks_conversations(authenticated_page, e2e_session_store):
    page = authenticated_page

    _seed_session(e2e_session_store, "Alpha thread")
    _seed_session(e2e_session_store, "Beta thread")

    page.reload()
    wait_for_authed(page)
    page.wait_for_selector('[data-testid="chat-session-menu-trigger"]', timeout=5000)

    rail = page.locator('[data-testid="chat-rail"]')
    header_title = page.locator('[data-testid="chat-conversation"] .title-btn')

    # Each rail pick routes through the hash, so it becomes a history entry.
    rail.get_by_role("button", name=re.compile(re.escape("Alpha thread"))).click()
    expect(header_title).to_have_text("Alpha thread")
    rail.get_by_role("button", name=re.compile(re.escape("Beta thread"))).click()
    expect(header_title).to_have_text("Beta thread")

    page.go_back()
    expect(header_title).to_have_text("Alpha thread")

    page.go_forward()
    expect(header_title).to_have_text("Beta thread")
