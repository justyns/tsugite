"""Mobile gesture and viewport-collapse tests.

At narrow viewports (<= 640px) the conversations view collapses to a single
pane: sidebar OR thread, not both. The mobile-hidden class on
`.console-sidebar` / `.console-main` and the `.mobile-back` button drive the
navigation. These tests pin the collapse and back-button flow.
"""

import pytest
from tsugite_daemon.session_store import Session, SessionSource

from tsugite.history.storage import generate_session_id

from .helpers import CONV_VIEW, open_conversations, reload_conversations_view

MOBILE_VIEWPORT = {"width": 375, "height": 800}


def _make_session(store, user_id):
    sid = generate_session_id("test-agent")
    s = Session(id=sid, agent="test-agent", source=SessionSource.INTERACTIVE.value, user_id=user_id)
    store.create_session(s)
    return s


@pytest.fixture
def mobile_page(authenticated_page):
    """authenticated_page narrowed to a phone-sized viewport."""
    page = authenticated_page
    page.set_viewport_size(MOBILE_VIEWPORT)
    return page


def test_mobile_collapses_to_sidebar_when_no_session_selected(mobile_page, e2e_session_store):
    """Sidebar visible, main thread hidden when nothing is selected at <=640px."""
    page = mobile_page
    user_id = page.evaluate("Alpine.store('app').userId")
    _make_session(e2e_session_store, user_id)

    open_conversations(page)
    reload_conversations_view(page)
    # autoSelectInteractive on reload picks a session, so clear it to test the
    # no-selection state directly.
    page.evaluate(f"Alpine.$data(document.querySelector({CONV_VIEW!r})).selectedSessionId = null;")
    page.wait_for_function(
        f"Alpine.$data(document.querySelector({CONV_VIEW!r})).selectedSessionId === null",
        timeout=2000,
    )

    sidebar = page.locator(".console-sidebar").first
    main = page.locator(".console-main").first
    assert sidebar.is_visible()
    # main has mobile-hidden when selectedSessionId is null.
    assert "mobile-hidden" in (main.get_attribute("class") or "")


def test_mobile_selecting_session_shows_thread_and_hides_sidebar(mobile_page, e2e_session_store):
    """Selecting a session swaps the panes: thread visible, sidebar hidden."""
    page = mobile_page
    user_id = page.evaluate("Alpine.store('app').userId")
    session = _make_session(e2e_session_store, user_id)

    open_conversations(page)
    reload_conversations_view(page)
    page.evaluate(
        f"Alpine.$data(document.querySelector({CONV_VIEW!r})).selectSessionById({session.id!r}, {{follow: false}})"
    )
    page.wait_for_function(
        f"Alpine.$data(document.querySelector({CONV_VIEW!r})).selectedSessionId === {session.id!r}",
        timeout=3000,
    )

    sidebar = page.locator(".console-sidebar").first
    main = page.locator(".console-main").first
    assert "mobile-hidden" in (sidebar.get_attribute("class") or "")
    assert main.is_visible()
    # mobile-back button is visible inside the thread header.
    assert page.locator(".console-thread-header .mobile-back").is_visible()


def test_mobile_back_button_returns_to_sidebar(mobile_page, e2e_session_store):
    """Clicking .mobile-back clears selectedSessionId and re-shows the sidebar."""
    page = mobile_page
    user_id = page.evaluate("Alpine.store('app').userId")
    session = _make_session(e2e_session_store, user_id)

    open_conversations(page)
    reload_conversations_view(page)
    page.evaluate(
        f"Alpine.$data(document.querySelector({CONV_VIEW!r})).selectSessionById({session.id!r}, {{follow: false}})"
    )
    page.wait_for_function(
        f"Alpine.$data(document.querySelector({CONV_VIEW!r})).selectedSessionId === {session.id!r}",
        timeout=3000,
    )

    page.locator(".console-thread-header .mobile-back").click()
    page.wait_for_function(
        f"Alpine.$data(document.querySelector({CONV_VIEW!r})).selectedSessionId === null",
        timeout=3000,
    )

    sidebar = page.locator(".console-sidebar").first
    main = page.locator(".console-main").first
    assert sidebar.is_visible()
    assert "mobile-hidden" in (main.get_attribute("class") or "")


def test_mobile_tab_bar_scrolls_horizontally(mobile_page):
    """At narrow widths the tab bar overflows horizontally (not wrapping/squishing)."""
    page = mobile_page
    tab_bar = page.locator(".console-tabs").first
    overflow_x = tab_bar.evaluate("el => getComputedStyle(el).overflowX")
    assert overflow_x in ("auto", "scroll"), (
        f"console-tabs should scroll horizontally at narrow widths; got {overflow_x}"
    )
