"""Picking a chat in the command palette opens it in its own tab."""

from __future__ import annotations

import os
import re

import pytest
from playwright.sync_api import expect

from .helpers import open_view, seed_session, wait_for_authed

PALETTE_TRIGGER = '[data-testid="palette-trigger"]'
CHAT_RAIL = '[data-testid="chat-rail"]'


@pytest.fixture
def alpha_and_beta(authenticated_page, e2e_session_store):
    """Two seeded chats, Alpha picked from the rail into the only tab."""
    seed_session(e2e_session_store, "Alpha thread")
    seed_session(e2e_session_store, "Beta thread")

    page = authenticated_page
    page.reload()
    wait_for_authed(page)
    page.wait_for_selector('[data-testid="chat-session-menu-trigger"]', timeout=5000)

    rail = page.locator(CHAT_RAIL)
    rail.get_by_role("button", name=re.compile(re.escape("Alpha thread"))).click()
    expect(page.get_by_role("tab")).to_have_count(1)
    expect(page.get_by_role("tab").first).to_contain_text("Alpha thread")
    return page


def _pick(page, query: str, key: str = "Enter") -> None:
    page.locator(PALETTE_TRIGGER).click()
    palette = page.get_by_role("combobox", name="Command palette search")
    palette.fill(query)
    expect(page.get_by_role("option").first).to_contain_text(query)
    palette.press(key)


def test_palette_opens_the_picked_chat_beside_the_open_one(alpha_and_beta):
    page = alpha_and_beta
    _pick(page, "Beta thread")

    if shot := os.environ.get("TSU_SHOT"):
        page.screenshot(path=shot)

    tabs = page.get_by_role("tab")
    expect(tabs).to_have_count(2)
    expect(tabs.nth(0)).to_contain_text("Alpha thread")
    expect(tabs.nth(1)).to_contain_text("Beta thread")
    expect(tabs.nth(1)).to_have_attribute("aria-selected", "true")

    # Picking a chat that is already open focuses its tab and creates nothing.
    _pick(page, "Alpha thread")
    expect(tabs).to_have_count(2)
    expect(tabs.nth(0)).to_have_attribute("aria-selected", "true")


def test_shift_enter_in_the_palette_replaces_the_focused_tab(alpha_and_beta, e2e_workspace):
    page = alpha_and_beta
    (e2e_workspace / "e2e_palette_note.md").write_text("# Note\n\nOpen beside the chat.\n")

    # A file preview holds focus. Shift replaces that tab, not the chat tab.
    open_view(page, "files")
    page.get_by_label("Search workspace").fill("e2e_palette_note")
    page.locator('[data-testid="file-node-e2e_palette_note.md"]').click()
    tabs = page.get_by_role("tab")
    expect(tabs).to_have_count(2)
    expect(tabs.nth(1)).to_contain_text("e2e_palette_note.md")

    _pick(page, "Beta thread", "Shift+Enter")

    expect(tabs).to_have_count(2)
    expect(tabs.nth(0)).to_contain_text("Alpha thread")
    expect(tabs.nth(1)).to_contain_text("Beta thread")
    expect(tabs.nth(1)).to_have_attribute("aria-selected", "true")


def test_palette_reuses_the_default_spaces_paramless_chat_tab(authenticated_page, e2e_session_store):
    """The stock layout docks a chat tab with no sessionId, resolved to the primary session."""
    seed_session(e2e_session_store, "Alpha thread")
    seed_session(e2e_session_store, "Beta thread")

    page = authenticated_page
    page.reload()
    wait_for_authed(page)
    page.wait_for_selector('[data-testid="chat-session-menu-trigger"]', timeout=5000)

    tabs = page.get_by_role("tab")
    expect(tabs).to_have_count(1)
    on_screen = page.locator('[data-testid="chat-conversation"] .title-btn').inner_text()

    _pick(page, on_screen)

    expect(tabs).to_have_count(1)
    expect(page.locator('[data-testid="chat-conversation"] .title-btn')).to_have_text(on_screen)
