"""Spaces: each space owns a whole multiplexer layout, switchable from the top bar.

The store has held a layout per space for a while (see stores/spaces.svelte.ts),
but nothing in the shell could create or switch one, so the model was
unreachable. These drive the top-bar switcher and assert the layouts survive a
round trip: a chat+note split in one space, a single fullscreen chat in another.
"""

from playwright.sync_api import expect

from .helpers import E2E_USER_ID, open_view, wait_for_authed

SPACE_BAR = '[data-testid="space-bar"]'
PANE = '[data-testid="mux-pane"]'


def _space(page, name: str):
    return page.locator(SPACE_BAR).get_by_role("button", name=name, exact=True)


def _split_active_pane(page) -> None:
    """Split the focused pane in two through the pane's own split affordance."""
    page.get_by_role("button", name="Split pane").first.click()
    expect(page.locator(PANE)).to_have_count(2)


def test_each_space_keeps_its_own_layout(authenticated_page, e2e_session_store, e2e_workspace):
    (e2e_workspace / "e2e_space_note.md").write_text("# Space note\n\nPinned beside the chat.\n")
    e2e_session_store.get_or_create_interactive(E2E_USER_ID, "test-agent")

    page = authenticated_page
    page.reload()
    page.wait_for_selector(SPACE_BAR)

    # The seeded space is the only one, so it offers no close control.
    expect(_space(page, "Main")).to_have_attribute("aria-pressed", "true")
    expect(page.locator(SPACE_BAR).get_by_role("button", name="Close Main")).to_have_count(0)

    # Main: a split holding the chat on the left and a markdown note on the right.
    _split_active_pane(page)
    open_view(page, "files")
    page.get_by_label("Search workspace").fill("e2e_space_note")
    page.locator('[data-testid="file-node-e2e_space_note.md"]').click()
    expect(page.locator(PANE).nth(1)).to_contain_text("Space note")
    expect(page.locator(PANE)).to_have_count(2)

    # A second space starts fresh on a single pane; Main's split is untouched.
    page.get_by_role("button", name="New space").click()
    expect(_space(page, "Space 2")).to_have_attribute("aria-pressed", "true")
    expect(page.locator(PANE)).to_have_count(1)

    # Back to Main: the split returns, note and all.
    _space(page, "Main").click()
    expect(_space(page, "Main")).to_have_attribute("aria-pressed", "true")
    expect(page.locator(PANE)).to_have_count(2)
    expect(page.locator(PANE).nth(1)).to_contain_text("Space note")

    # And forward again: the second space is still the single pane it was.
    _space(page, "Space 2").click()
    expect(page.locator(PANE)).to_have_count(1)


def test_a_space_can_be_renamed_and_closed(authenticated_page, e2e_session_store):
    e2e_session_store.get_or_create_interactive(E2E_USER_ID, "test-agent")

    page = authenticated_page
    page.reload()
    page.wait_for_selector(SPACE_BAR)

    page.get_by_role("button", name="New space").click()
    _space(page, "Space 2").dblclick()
    rename = page.get_by_label("Rename space")
    rename.fill("Planning")
    rename.press("Enter")
    expect(_space(page, "Planning")).to_be_visible()

    # Closing drops it and falls back to the neighbour, whose layout is intact.
    page.locator(SPACE_BAR).get_by_role("button", name="Close Planning").click()
    expect(_space(page, "Planning")).to_have_count(0)
    expect(_space(page, "Main")).to_have_attribute("aria-pressed", "true")
    expect(page.locator(PANE)).to_have_count(1)


def test_the_spaces_strip_survives_a_phone_width(authenticated_page):
    """The strip is the only shrinkable item in the appbar, so without a floor it
    absorbs every sibling's overflow and collapses to a few pixels."""
    page = authenticated_page
    page.set_viewport_size({"width": 1280, "height": 900})
    page.reload()
    wait_for_authed(page)
    page.get_by_role("button", name="New space").click()
    page.wait_for_timeout(300)
    try:
        page.set_viewport_size({"width": 380, "height": 800})
        page.wait_for_timeout(300)

        strip = page.locator(SPACE_BAR)
        box = strip.bounding_box()
        assert box is not None and box["width"] >= 90, f"strip collapsed to {box and box['width']}px"
        assert strip.get_by_role("button", name="Main", exact=True).is_visible()

        # A wider appbar than the phone is a horizontal page scroll, which this UI never has.
        overflow = page.evaluate("() => document.documentElement.scrollWidth - document.documentElement.clientWidth")
        assert overflow == 0, f"page scrolls horizontally by {overflow}px"
    finally:
        # The page and its localStorage outlive this test, so hand back a desktop
        # viewport and the single space the other tests start from.
        page.set_viewport_size({"width": 1280, "height": 900})
        page.wait_for_timeout(200)
        for name in ("Space 2", "Space 3"):
            close = page.get_by_role("button", name=f"Close {name}")
            if close.count():
                close.first.click()
                page.wait_for_timeout(200)
