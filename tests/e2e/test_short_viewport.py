"""A short pane or a phone-width viewport compacts the chat header, tab strip
and composer.

A phone also folds the header into a single row and moves the effort control
into the session menu. A narrow-but-tall window keeps the full layout.
"""

import re

from playwright.sync_api import expect

from .helpers import E2E_USER_ID, wait_for_authed
from .test_a11y import BLOCKING, _run_axe

CONVERSATION = '[data-testid="chat-conversation"]'
HEADER = f"{CONVERSATION} > header"
TRANSCRIPT = '[data-testid="chat-transcript"]'
TABLIST = '[data-testid="mux-pane"] [role="tablist"]'
RAIL = '[data-testid="chat-rail"]'
TOPBAR = '[data-testid="topbar"]'
MODEL_TRIGGER = '[data-testid="chat-model-trigger"]'
MODEL_POPOVER = '[data-testid="chat-model-popover"]'
MODEL_OPTION = '[data-testid^="chat-model-opt-"]'
EFFORT = '[data-testid="chat-effort-seg"]'
MENU_TRIGGER = '[data-testid="chat-session-menu-trigger"]'
MENU = '[data-testid="chat-session-menu"]'
PALETTE_TRIGGER = '[data-testid="palette-trigger"]'


def _surround_height(page, width: int, height: int) -> float:
    """Window height not spent on the transcript."""
    page.set_viewport_size({"width": width, "height": height})
    expect(page.locator(TRANSCRIPT)).to_be_visible()
    return height - page.locator(TRANSCRIPT).bounding_box()["height"]


def _assert_aria_intact(page) -> None:
    """Hiding the strip must not leave the tab panel pointing at a tab that is gone.

    axe files a dangling aria reference under `incomplete`, not `violations`.
    """
    report = _run_axe(page)
    found = [v["id"] for v in report["violations"] if v.get("impact") in BLOCKING]
    found += [v["id"] for v in report["incomplete"] if v["id"].startswith("aria-")]
    assert found == [], found


def test_a_wide_short_window_compacts_the_chat_header_and_composer(chat_page):
    page = chat_page
    full = _surround_height(page, 1152, 760)
    compact = _surround_height(page, 1152, 693)
    expect(page.locator(CONVERSATION)).to_have_attribute("data-density", "dense")
    expect(page.locator(TABLIST)).to_have_count(0)
    # The strip and the attach row are gone and the header is tighter.
    assert full - compact >= 60, (full, compact)
    assert page.locator(TRANSCRIPT).bounding_box()["height"] / 693 > 0.5

    _assert_aria_intact(page)


def test_a_narrow_tall_window_keeps_the_full_layout(chat_page):
    page = chat_page
    page.set_viewport_size({"width": 760, "height": 900})
    expect(page.locator(TABLIST)).to_have_count(1)
    expect(page.locator(CONVERSATION)).not_to_have_attribute("data-density", "dense")


def test_a_portrait_phone_folds_the_header_and_composer(chat_page, e2e_session_store):
    page = chat_page
    # Only a model that declares effort levels gets an effort control.
    session = e2e_session_store.get_or_create_interactive(E2E_USER_ID)
    e2e_session_store.set_model_override(session.id, "anthropic:claude-sonnet-4-5")
    page.reload()
    wait_for_authed(page)
    page.set_viewport_size({"width": 380, "height": 667})
    # A phone opens the sessions list first. Picking a session drills into it.
    page.locator(RAIL).get_by_role("button", name=re.compile(r"^Untitled session")).click()
    expect(page.locator(MODEL_TRIGGER)).to_be_visible()

    expect(page.locator(CONVERSATION)).to_have_attribute("data-density", "dense")
    expect(page.locator(TABLIST)).to_have_count(0)

    header = page.locator(HEADER).bounding_box()
    assert header["height"] < 44, header
    # Nothing between the app bar and the header, and nothing under the header
    # but the transcript.
    bar = page.locator(TOPBAR).bounding_box()
    transcript = page.locator(TRANSCRIPT).bounding_box()
    assert transcript["y"] <= bar["y"] + bar["height"] + header["height"] + 2, (
        bar,
        header,
        transcript,
    )
    assert transcript["height"] / 667 > 0.55, transcript

    # Effort is one tap deeper, inside the session menu.
    expect(page.locator(f"{HEADER} > {EFFORT}")).to_have_count(0)
    page.locator(MENU_TRIGGER).click()
    expect(page.locator(f"{MENU} {EFFORT}")).to_be_visible()
    page.keyboard.press("Escape")
    expect(page.locator(MENU)).to_have_count(0)

    # The model popover holds models, nothing else, and opens fully on-screen.
    page.locator(MODEL_TRIGGER).click()
    expect(page.locator(MODEL_POPOVER)).to_be_visible()
    expect(page.locator(f"{MODEL_POPOVER} {EFFORT}")).to_have_count(0)
    expect(page.locator(MODEL_OPTION).first).to_be_visible(timeout=15_000)
    pop = page.locator(MODEL_POPOVER).bounding_box()
    assert pop["x"] >= 0 and pop["x"] + pop["width"] <= 380, pop
    page.locator('[data-testid="chat-model-search"]').press("Escape")
    expect(page.locator(MODEL_POPOVER)).to_have_count(0)

    _assert_aria_intact(page)


def test_a_touch_device_drops_the_palette_keycap(browser, base_url, e2e_auth_token, session_runner_backend):
    ctx = browser.new_context(viewport={"width": 380, "height": 667}, has_touch=True)
    try:
        page = ctx.new_page()
        page.goto(base_url + "/api/health")
        page.evaluate(f"localStorage.setItem('tsugite_token', '{e2e_auth_token}')")
        page.goto(base_url)
        wait_for_authed(page)
        # The palette stays reachable. Only the keycap goes.
        expect(page.locator(PALETTE_TRIGGER)).to_be_visible()
        expect(page.locator(f"{PALETTE_TRIGGER} .t-kbd")).to_be_hidden()
    finally:
        ctx.close()
