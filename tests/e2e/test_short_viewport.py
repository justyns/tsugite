"""A short pane compacts its chrome and gives the transcript most of the height.

The pane's own height drives it. A wide-but-short window compacts, a
narrow-but-tall one does not, and the phone layout below 640px is untouched.
"""

from playwright.sync_api import expect

from .test_a11y import BLOCKING, _run_axe

CONVERSATION = '[data-testid="chat-conversation"]'
TRANSCRIPT = '[data-testid="chat-transcript"]'
TABLIST = '[data-testid="mux-pane"] [role="tablist"]'


def _chrome_height(page, width: int, height: int) -> float:
    """Window height not spent on the transcript."""
    page.set_viewport_size({"width": width, "height": height})
    expect(page.locator(TRANSCRIPT)).to_be_visible()
    return height - page.locator(TRANSCRIPT).bounding_box()["height"]


def test_a_wide_short_window_compacts_the_chat_chrome(chat_page):
    page = chat_page
    full = _chrome_height(page, 1152, 760)
    compact = _chrome_height(page, 1152, 693)
    expect(page.locator(CONVERSATION)).to_have_attribute("data-density", "short")
    expect(page.locator(TABLIST)).to_have_count(0)
    # The strip and the attach row are gone and the header is tighter.
    assert full - compact >= 60, (full, compact)
    assert page.locator(TRANSCRIPT).bounding_box()["height"] / 693 > 0.5

    # Hiding the strip must not leave the tab panel pointing at a tab that is gone.
    # axe files a dangling aria reference under `incomplete`, not `violations`.
    report = _run_axe(page)
    found = [v["id"] for v in report["violations"] if v.get("impact") in BLOCKING]
    found += [v["id"] for v in report["incomplete"] if v["id"].startswith("aria-")]
    assert found == [], found


def test_a_narrow_tall_window_keeps_the_full_chrome(chat_page):
    page = chat_page
    page.set_viewport_size({"width": 760, "height": 900})
    expect(page.locator(TABLIST)).to_have_count(1)
    expect(page.locator(CONVERSATION)).not_to_have_attribute("data-density", "short")


def test_a_portrait_phone_keeps_its_own_layout(chat_page):
    page = chat_page
    page.set_viewport_size({"width": 380, "height": 667})
    expect(page.locator(TABLIST)).to_have_count(1)
    expect(page.locator(CONVERSATION)).not_to_have_attribute("data-density", "short")
