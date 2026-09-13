"""Hovering a collapsed rail peeks it open without pinning it."""

from __future__ import annotations

import pytest
from playwright.sync_api import expect

from .helpers import wait_for_authed

RAIL_STRIP = '[data-testid="rail-expand"]'
RAIL_PEEK = '[data-testid="rail-peek"]'
CHAT_RAIL = '[data-testid="chat-rail"]'
NAV_RAIL = '[data-testid="nav-rail"]'
VIEW_HOST = '[data-testid="view-host"]'
WORK_MAIN = ".work-main"

COLLAPSED_RAIL = '{"chats": true, "terminals": false, "files": false}'


@pytest.fixture
def collapsed_page(authenticated_page):
    """Desktop page that loads with both rails collapsed."""
    page = authenticated_page
    page.set_viewport_size({"width": 1280, "height": 800})
    page.evaluate(f"localStorage.setItem('tsugite_rail_collapsed', '{COLLAPSED_RAIL}')")
    page.evaluate("localStorage.setItem('tsugite_nav_collapsed', '1')")
    page.reload()
    wait_for_authed(page)
    yield page


def _x(page, selector: str) -> float:
    box = page.locator(selector).bounding_box()
    assert box is not None, f"{selector} has no box"
    return box["x"]


def _away(page) -> None:
    """Park the pointer in the middle of the viewport, clear of both rails."""
    page.mouse.move(640, 400)


def test_hovering_the_expand_strip_peeks_the_sessions_rail_over_the_content(collapsed_page):
    page = collapsed_page
    before = _x(page, WORK_MAIN)

    page.locator(RAIL_STRIP).hover()

    peek = page.locator(RAIL_PEEK)
    expect(peek).to_be_visible()
    expect(peek.locator(CHAT_RAIL)).to_be_visible()
    # A peek is an overlay. The work area does not move and the collapsed flag stays set.
    assert _x(page, WORK_MAIN) == before
    assert page.evaluate("localStorage.getItem('tsugite_rail_collapsed')") == COLLAPSED_RAIL

    _away(page)
    expect(peek).to_have_count(0)


def test_escape_closes_a_peeked_sessions_rail(collapsed_page):
    page = collapsed_page
    page.locator(RAIL_STRIP).hover()
    expect(page.locator(RAIL_PEEK)).to_be_visible()

    page.keyboard.press("Escape")
    expect(page.locator(RAIL_PEEK)).to_have_count(0)


def test_hovering_the_collapsed_nav_rail_peeks_its_labels_open(collapsed_page):
    page = collapsed_page
    nav = page.locator(NAV_RAIL)
    before_host = _x(page, VIEW_HOST)
    before_nav = nav.bounding_box()
    assert before_nav is not None

    nav.hover()

    expect(nav).to_have_attribute("data-peeking", "")
    expect(page.locator(f'{NAV_RAIL} [data-testid="nav-chats"] .lb')).to_be_visible()
    # The nav keeps its 52px slot in the flex row. Only the overlay widens.
    after_nav = nav.bounding_box()
    assert after_nav is not None and after_nav["width"] == before_nav["width"]
    assert _x(page, VIEW_HOST) == before_host
    assert page.evaluate("localStorage.getItem('tsugite_nav_collapsed')") == "1"

    _away(page)
    expect(nav).not_to_have_attribute("data-peeking", "")
