"""E2E tests for the redesigned session sidebar."""

from tsugite.daemon.session_store import Session, SessionSource, SessionStatus


def test_sidebar_shows_active_recent_groups(authenticated_page, e2e_session_store):
    """Sessions are grouped into Active and Recent sections."""
    page = authenticated_page
    user_id = page.evaluate("Alpine.store('app').userId")

    # Create an active session
    e2e_session_store.get_or_create_interactive(user_id, "test-agent")

    # Create a completed session
    completed = Session(
        id="completed-1",
        agent="test-agent",
        source=SessionSource.BACKGROUND.value,
        status=SessionStatus.COMPLETED.value,
        prompt="finished task",
        result="done",
    )
    e2e_session_store.create_session(completed)

    page.locator("nav button", has_text="Conversations").click()
    page.wait_for_function("Alpine.store('app').view === 'conversations'", timeout=3000)
    page.wait_for_selector(".session-group-header", timeout=5000)

    headers = page.locator(".session-group-header").all_text_contents()
    assert any("Active" in h for h in headers)
    assert any("Recent" in h for h in headers)


def test_sidebar_source_icons_render(authenticated_page, e2e_session_store):
    """Source icons appear next to session entries."""
    page = authenticated_page
    user_id = page.evaluate("Alpine.store('app').userId")

    e2e_session_store.get_or_create_interactive(user_id, "test-agent")

    page.locator("nav button", has_text="Conversations").click()
    page.wait_for_function("Alpine.store('app').view === 'conversations'", timeout=3000)
    page.wait_for_selector(".source-icon", timeout=5000)

    icons = page.locator(".source-icon")
    assert icons.count() >= 1


def test_sidebar_metadata_chips_render(authenticated_page, e2e_session_store):
    """Metadata chips appear on sessions that have metadata."""
    page = authenticated_page

    session = Session(
        id="meta-session",
        agent="test-agent",
        source=SessionSource.BACKGROUND.value,
        status=SessionStatus.COMPLETED.value,
        prompt="task with metadata",
        metadata={"type": "code", "task": "https://example.com/task/1"},
    )
    e2e_session_store.create_session(session)

    page.locator("nav button", has_text="Conversations").click()
    page.wait_for_function("Alpine.store('app').view === 'conversations'", timeout=3000)

    page.wait_for_selector("summary.session-group-header", timeout=5000)
    recent_toggle = page.locator("summary.session-group-header", has_text="Recent")
    if recent_toggle.count() > 0:
        recent_toggle.click()
        page.wait_for_timeout(300)

    chips = page.locator(".session-chip")
    assert chips.count() >= 1
