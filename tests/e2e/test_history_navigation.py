"""Browser history (back/forward) conversation switching.

Regression for the bug where hashchange updated store.viewSessionId but nothing
watched it, so the viewed conversation only changed on a view transition (or a
full reload) -- never on back/forward between two conversations.
"""

from tsugite_daemon.session_store import Session, SessionSource

from tsugite.history.storage import generate_session_id

from .helpers import CONV_VIEW, open_conversations, reload_conversations_view, select_session_in_view


def _make_session(store, user_id):
    sid = generate_session_id("test-agent")
    s = Session(id=sid, agent="test-agent", source=SessionSource.INTERACTIVE.value, user_id=user_id)
    store.create_session(s)
    return s


def _selected(page):
    return page.evaluate(f"Alpine.$data(document.querySelector({CONV_VIEW!r})).selectedSessionId")


def test_browser_back_forward_switches_conversation(authenticated_page, e2e_session_store, mock_chat):
    page = authenticated_page
    user_id = page.evaluate("Alpine.store('app').userId")
    s1 = _make_session(e2e_session_store, user_id)
    s2 = _make_session(e2e_session_store, user_id)

    open_conversations(page)
    reload_conversations_view(page)

    # Forward navigation builds two hash history entries (selectSession writes the hash).
    select_session_in_view(page, s1.id)
    select_session_in_view(page, s2.id)
    assert _selected(page) == s2.id

    # Back -> the viewed conversation must follow the URL back to s1.
    page.go_back()
    page.wait_for_function(
        f"Alpine.$data(document.querySelector({CONV_VIEW!r})).selectedSessionId === {s1.id!r}",
        timeout=3000,
    )
    assert _selected(page) == s1.id

    # Forward -> back to s2.
    page.go_forward()
    page.wait_for_function(
        f"Alpine.$data(document.querySelector({CONV_VIEW!r})).selectedSessionId === {s2.id!r}",
        timeout=3000,
    )
    assert _selected(page) == s2.id
