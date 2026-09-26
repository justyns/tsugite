"""Switching back to a space keeps pane focus and the rail highlight on the chat in use."""

import json

from playwright.sync_api import expect
from tsugite_daemon.session_store import Session, SessionSource

from tsugite.history import generate_session_id

from .helpers import E2E_USER_ID, wait_for_authed

SPACE_BAR = '[data-testid="space-bar"]'
PANE = '[data-testid="mux-pane"]'
RAIL_ACTIVE = '[data-testid="chat-rail"] [aria-current="true"]'


def _session(store, title: str) -> str:
    session = Session(
        id=generate_session_id("test-agent"), source=SessionSource.INTERACTIVE.value, user_id=E2E_USER_ID, title=title
    )
    store.create_session(session)
    return session.id


def _leaf(pane_id: str, tab_id: str, session_id: str) -> dict:
    tab = {"id": tab_id, "kind": "chat", "params": {"sessionId": session_id}, "title": "Chat"}
    return {"type": "leaf", "id": pane_id, "tabs": [tab], "activeTabId": tab_id}


def _focused_pane(page, space_id: str) -> str:
    return page.evaluate(
        "(id) => JSON.parse(localStorage.getItem('tsugite_spaces')).spaces.find((s) => s.id === id).layout.focusedPaneId",
        space_id,
    )


def _switch(page, name: str) -> None:
    page.locator(SPACE_BAR).get_by_role("button", name=name, exact=True).click()
    expect(page.locator(SPACE_BAR).get_by_role("button", name=name, exact=True)).to_have_attribute(
        "aria-pressed", "true"
    )


def test_space_switch_keeps_focus_on_the_chat_in_use(authenticated_page, e2e_session_store):
    page = authenticated_page
    left = _session(e2e_session_store, "Left chat")
    right = _session(e2e_session_store, "Right chat")
    other = _session(e2e_session_store, "Other chat")
    split = {
        "type": "split",
        "id": "split-a",
        "dir": "row",
        "sizes": [0.5, 0.5],
        "children": [_leaf("pane-left", "tab-left", left), _leaf("pane-right", "tab-right", right)],
    }
    spaces = {
        "version": 1,
        "activeSpaceId": "space-a",
        "spaces": [
            {"id": "space-a", "name": "A", "layout": {"version": 1, "root": split, "focusedPaneId": "pane-left"}},
            {
                "id": "space-b",
                "name": "B",
                "layout": {"version": 1, "root": _leaf("pane-b", "tab-b", other), "focusedPaneId": "pane-b"},
            },
        ],
    }
    page.evaluate("(v) => localStorage.setItem('tsugite_spaces', v)", json.dumps(spaces))
    page.reload()
    wait_for_authed(page)
    expect(page.locator(PANE)).to_have_count(2)

    page.locator(PANE).nth(0).locator("textarea").click()
    expect(page.locator(RAIL_ACTIVE)).to_contain_text("Left chat")

    _switch(page, "B")
    expect(page.locator(RAIL_ACTIVE)).to_contain_text("Other chat")
    _switch(page, "A")
    expect(page.locator(PANE)).to_have_count(2)
    # Both chats' session details load after the remount; give the autofocus a chance to land.
    expect(page.locator(PANE).nth(1).get_by_role("button", name="Right chat")).to_be_visible()
    page.wait_for_timeout(500)

    expect(page.locator(RAIL_ACTIVE)).to_contain_text("Left chat")
    page.wait_for_timeout(400)  # layout persistence is debounced
    assert _focused_pane(page, "space-a") == "pane-left"
