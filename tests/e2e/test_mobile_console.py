"""Mobile thread-header layout regression tests.

At <=640px the conversation thread header used to dominate the viewport
because every row kept its desktop footprint and wrapped instead of
truncating. These tests pin the compressed layout: collapsed path,
ellipsized title, hidden row-4 chips, clamped model dropdown, and a
moved-into-popover set of secondary actions.
"""

import pytest

from tsugite.daemon.session_store import Session, SessionSource, SessionStatus
from tsugite.history.storage import generate_session_id

from .helpers import open_conversations, reload_conversations_view, select_session_in_view

# iPhone 14 portrait
MOBILE_VIEWPORT = {"width": 390, "height": 844}
DESKTOP_VIEWPORT = {"width": 1440, "height": 900}


def _make_session(store, user_id, *, title=None, status=SessionStatus.ACTIVE.value, metadata=None):
    sid = generate_session_id("test-agent")
    s = Session(
        id=sid,
        agent="test-agent",
        source=SessionSource.INTERACTIVE.value,
        status=status,
        user_id=user_id,
        title=title,
        metadata=metadata or {},
    )
    store.create_session(s)
    return s


@pytest.fixture
def mobile_page(authenticated_page):
    page = authenticated_page
    page.set_viewport_size(MOBILE_VIEWPORT)
    return page


@pytest.fixture
def desktop_page(authenticated_page):
    page = authenticated_page
    page.set_viewport_size(DESKTOP_VIEWPORT)
    return page


def _select_session(page, store, *, title=None, status=SessionStatus.ACTIVE.value, metadata=None):
    user_id = page.evaluate("Alpine.store('app').userId")
    session = _make_session(store, user_id, title=title, status=status, metadata=metadata)
    open_conversations(page)
    reload_conversations_view(page)
    select_session_in_view(page, session.id)
    page.wait_for_selector(".console-thread-header", timeout=5000)
    return session


def test_thread_header_below_threshold_height(mobile_page, e2e_session_store):
    """Header must not dominate the mobile viewport (was ~400px, now <200px).

    Populate every row with realistic worst-case content (long title, topic,
    metadata chips, long working_dir) so the unfixed layout actually wraps to
    its full 8-12 visual lines.
    """
    page = mobile_page
    _select_session(
        page,
        e2e_session_store,
        title="A reasonable but somewhat long session title for testing layout",
        metadata={
            "type": "interactive",
            "status_text": "running long task",
            "topic": "implement mobile-friendly thread header for the conversations view",
            "working_dir": "/home/justyn/dev/tsugite/some/deeply/nested/path/for/testing",
        },
    )

    header = page.locator(".console-thread-header").first
    box = header.bounding_box()
    assert box is not None, "thread header should be present and visible"
    assert box["height"] < 200, (
        f"thread header is {box['height']:.0f}px tall on a 844px viewport; "
        "expected <200px so the conversation has room to breathe"
    )


def test_thread_header_does_not_overflow_viewport_width(mobile_page, e2e_session_store):
    """Header content must fit horizontally; no horizontal scroll inside the header."""
    page = mobile_page
    _select_session(
        page,
        e2e_session_store,
        title="A very long session title that absolutely should not push the header off-screen on a 390px viewport",
    )

    overflow = page.evaluate(
        "() => { const h = document.querySelector('.console-thread-header');"
        " return { scroll: h.scrollWidth, client: h.clientWidth }; }"
    )
    assert overflow["scroll"] <= overflow["client"] + 1, (
        f"thread header overflows: scrollWidth={overflow['scroll']} > clientWidth={overflow['client']}"
    )


def test_only_essential_actions_visible_inline_on_mobile(mobile_page, e2e_session_store):
    """Rename / pin / mark-complete must move into the more-actions popover at <=640px."""
    page = mobile_page
    _select_session(page, e2e_session_store)

    rename_btn = page.locator('.console-thread-header .row.top button[title="rename"].inline-only').first
    pin_btn = page.locator(".console-thread-header .row.top .hbtn.pin.inline-only").first
    archive_btn = page.locator(".console-thread-header .row.top .hbtn.archive.inline-only").first

    assert rename_btn.count() == 1, "inline rename button should still exist in DOM (just hidden)"
    assert not rename_btn.is_visible(), "rename button must not be visible inline on mobile"
    if pin_btn.count():
        assert not pin_btn.is_visible(), "pin button must not be visible inline on mobile"
    if archive_btn.count():
        assert not archive_btn.is_visible(), "mark-complete button must not be visible inline on mobile"

    mobile_back = page.locator(".console-thread-header .row.top .hbtn.mobile-back").first
    more_actions = page.locator(".console-thread-header .row.top .more-actions .hbtn").first
    assert mobile_back.is_visible(), "mobile-back button must be visible inline"
    assert more_actions.is_visible(), "more-actions trigger must be visible inline"

    state_seg = page.locator(".console-thread-header .path .state-seg").first
    assert not state_seg.is_visible(), "intermediate state path segment must collapse on mobile"


def test_more_actions_popover_contains_moved_items(mobile_page, e2e_session_store):
    """The ⋯ popover must surface rename/pin/mark-complete at <=640px."""
    page = mobile_page
    _select_session(page, e2e_session_store)

    page.locator(".console-thread-header .row.top .more-actions .hbtn").first.click()
    pop = page.locator(".console-thread-header .row.top .more-actions .setting-pop").first
    pop.wait_for(state="visible", timeout=2000)

    rename_row = pop.locator('.setting-pop-row[title="rename"]').first
    pin_row = pop.locator('.setting-pop-row[title="pin"]').first
    complete_row = pop.locator('.setting-pop-row[title="mark complete"]').first

    assert rename_row.is_visible(), "rename must appear inside the more-actions popover on mobile"
    assert pin_row.is_visible(), "pin must appear inside the more-actions popover on mobile"
    assert complete_row.is_visible(), "mark-complete must appear inside the more-actions popover on mobile"


def test_title_truncates_long_session_name(mobile_page, e2e_session_store):
    """Long h1 titles must ellipsize, not wrap into 3-4 lines."""
    page = mobile_page
    long_title = "A really really long session title that will absolutely push the context meter off the row"
    _select_session(page, e2e_session_store, title=long_title)

    h1 = page.locator(".console-thread-header h1").first
    style = h1.evaluate(
        "el => ({"
        " textOverflow: getComputedStyle(el).textOverflow,"
        " whiteSpace: getComputedStyle(el).whiteSpace,"
        " overflow: getComputedStyle(el).overflow,"
        "})"
    )
    assert style["textOverflow"] == "ellipsis", f"h1 must have text-overflow: ellipsis, got {style!r}"
    assert style["whiteSpace"] == "nowrap", f"h1 must have white-space: nowrap, got {style!r}"
    assert style["overflow"] in ("hidden", "clip"), f"h1 must clip overflow, got {style!r}"


def test_row4_metadata_chips_hidden_on_mobile(mobile_page, e2e_session_store):
    """Row 4 chips duplicate row 5 metadata; they must be hidden at <=640px."""
    page = mobile_page
    _select_session(
        page,
        e2e_session_store,
        metadata={"type": "interactive", "status_text": "in-progress"},
    )
    # Alpine needs a tick for the x-for chip template to render.
    page.wait_for_function(
        "(() => { const v = Alpine.$data(document.querySelector('[x-data*=conversationsView]'));"
        " return (v.metadataChips(v.selectedSessionMeta || {}, { excludeKeys: ['topic'] }) || []).length > 0; })()",
        timeout=2000,
    )
    chip_count = page.evaluate(
        "(() => { const v = Alpine.$data(document.querySelector('[x-data*=conversationsView]'));"
        " return (v.metadataChips(v.selectedSessionMeta || {}, { excludeKeys: ['topic'] }) || []).length; })()"
    )
    assert chip_count >= 1, "test setup: metadataChips should return at least one chip"

    # Check the computed display style on any rendered chip in the topic-row.
    # Even with display:none the elements remain in the DOM.
    chips = page.locator(".console-thread-header .row.topic-row span.chip")
    n = chips.count()
    assert n >= 1, "test setup: at least one chip <span> should be in the DOM"
    for i in range(n):
        display = chips.nth(i).evaluate("el => getComputedStyle(el).display")
        assert display == "none", f"row-4 chip {i} must have display:none on mobile, got {display!r}"


def test_model_select_clamped_on_mobile(mobile_page, e2e_session_store):
    """Model dropdown value must clamp at ~12em with ellipsis (was 30+ chars wide)."""
    page = mobile_page
    _select_session(page, e2e_session_store)

    model_btn = page.locator(".console-thread-header .meta-row .v-btn.model").first
    style = model_btn.evaluate(
        "el => ({"
        " maxWidth: getComputedStyle(el).maxWidth,"
        " textOverflow: getComputedStyle(el).textOverflow,"
        " whiteSpace: getComputedStyle(el).whiteSpace,"
        " overflow: getComputedStyle(el).overflow,"
        "})"
    )
    # 12em with font-size 11px ≈ 132px; assert clamp at a sensible threshold
    px = float(style["maxWidth"].replace("px", "")) if style["maxWidth"].endswith("px") else 9999
    assert px < 200, f"model v-btn must clamp on mobile, got max-width {style['maxWidth']}"
    assert style["textOverflow"] == "ellipsis", f"model v-btn must ellipsize, got {style!r}"

    title_attr = model_btn.get_attribute("title") or ""
    assert title_attr, "model v-btn must expose the full model name via title= for hover/longpress"


def test_desktop_layout_unchanged(desktop_page, e2e_session_store):
    """Regression guard: nothing in the mobile rules leaks to desktop."""
    page = desktop_page
    _select_session(page, e2e_session_store, title="A typical session title")

    rename_btn = page.locator('.console-thread-header .row.top button[title="rename"].inline-only').first
    assert rename_btn.is_visible(), "inline rename must remain visible on desktop"

    state_seg = page.locator(".console-thread-header .path .state-seg").first
    assert state_seg.is_visible(), "intermediate state segment must remain visible on desktop"

    # mobile-only popover rows should be hidden on desktop
    page.locator(".console-thread-header .row.top .more-actions .hbtn").first.click()
    pop = page.locator(".console-thread-header .row.top .more-actions .setting-pop").first
    pop.wait_for(state="visible", timeout=2000)
    mobile_only_rows = pop.locator(".setting-pop-row.mobile-only")
    n = mobile_only_rows.count()
    for i in range(n):
        assert not mobile_only_rows.nth(i).is_visible(), f"mobile-only popover row {i} must hide on desktop"
