"""Draft input persistence across reloads and session switches."""

from tsugite_daemon.session_store import Session, SessionSource

from tsugite.history.storage import generate_session_id

from .helpers import CONV_VIEW


def _new_session(e2e_session_store, user_id):
    """Create a fresh interactive session."""
    sid = generate_session_id("test-agent")
    s = Session(id=sid, agent="test-agent", source=SessionSource.INTERACTIVE.value, user_id=user_id)
    e2e_session_store.create_session(s)
    return s


def _go_to_conversations(page):
    """Navigate to conversations and wait for session list to load."""
    page.locator(".console-tabs button.console-tab", has_text="Conversations").click()
    page.wait_for_function("Alpine.store('app').view === 'conversations'", timeout=3000)
    page.wait_for_selector(".console-session", timeout=5000)


def test_draft_survives_page_reload(authenticated_page, base_url, e2e_session_store):
    """Typed text should persist in localStorage and restore after reload."""
    page = authenticated_page
    user_id = page.evaluate("Alpine.store('app').userId")
    session = _new_session(e2e_session_store, user_id)

    # Navigate to conversations, then select the session via hash
    page.goto(base_url)
    page.wait_for_function("!Alpine.store('app').authRequired", timeout=5000)
    page.wait_for_function("Alpine.store('app').selectedAgent", timeout=5000)
    # Set the hash after agents are loaded so viewSessionId triggers properly
    page.evaluate(f"location.hash = 'conversations?session={session.id}'")
    page.wait_for_selector("#message-input", timeout=10000)

    page.locator("#message-input").fill("unsent draft message")
    draft_key = f"tsugite_draft_{session.id}"
    page.wait_for_function(
        f"localStorage.getItem('{draft_key}') === 'unsent draft message'",
        timeout=2000,
    )
    draft = page.evaluate(f"localStorage.getItem('{draft_key}')")
    assert draft == "unsent draft message"

    # Full reload - token and draft persist in localStorage
    page.goto(base_url)
    page.wait_for_function("!Alpine.store('app').authRequired", timeout=5000)
    page.wait_for_function("Alpine.store('app').selectedAgent", timeout=5000)
    page.evaluate(f"location.hash = 'conversations?session={session.id}'")
    page.wait_for_selector("#message-input", timeout=10000)

    restored = page.locator("#message-input").input_value()
    assert restored == "unsent draft message"


def test_draft_cleared_after_send(authenticated_page, base_url, e2e_session_store, mock_chat):
    """Sending a message should remove the draft from localStorage."""
    mock_chat("ok")

    page = authenticated_page
    user_id = page.evaluate("Alpine.store('app').userId")
    session = _new_session(e2e_session_store, user_id)

    page.goto(base_url)
    page.wait_for_function(
        "typeof Alpine !== 'undefined' && Alpine.store('app') && !Alpine.store('app').authRequired",
        timeout=5000,
    )
    page.wait_for_function("Alpine.store('app').selectedAgent", timeout=5000)
    # Set hash after Alpine is ready so viewSessionId triggers properly.
    page.evaluate(f"location.hash = 'conversations?session={session.id}'")
    page.wait_for_selector("#message-input", timeout=10000)

    page.locator("#message-input").fill("message to send")
    page.wait_for_function(
        f"localStorage.getItem('tsugite_draft_{session.id}') === 'message to send'",
        timeout=2000,
    )

    page.locator("#message-input").press("Enter")
    page.wait_for_selector(".console-turn.agent", timeout=15000)

    draft = page.evaluate(
        "localStorage.getItem(Object.keys(localStorage).find(k => k.startsWith('tsugite_draft_')) || 'missing')"
    )
    assert draft is None


def test_drafts_isolated_between_sessions(authenticated_page, e2e_session_store):
    """Each session should maintain its own independent draft."""
    page = authenticated_page
    user_id = page.evaluate("Alpine.store('app').userId")

    s1 = _new_session(e2e_session_store, user_id)
    s2 = _new_session(e2e_session_store, user_id)

    page.reload()
    page.wait_for_function(
        "typeof Alpine !== 'undefined' && Alpine.store('app') && !Alpine.store('app').authRequired",
        timeout=5000,
    )
    _go_to_conversations(page)

    items = page.locator(".console-session")
    assert items.count() >= 2

    def _select(sid):
        page.evaluate(
            f"Alpine.$data(document.querySelector({CONV_VIEW!r})).selectSessionById({sid!r}, {{follow: false}})"
        )
        page.wait_for_function(
            f"Alpine.$data(document.querySelector({CONV_VIEW!r})).selectedSessionId === {sid!r}",
            timeout=3000,
        )
        page.wait_for_selector("#message-input", timeout=5000)

    _select(s1.id)
    page.locator("#message-input").fill("draft for session 1")
    page.wait_for_function(
        f"localStorage.getItem('tsugite_draft_{s1.id}') === 'draft for session 1'",
        timeout=2000,
    )

    _select(s2.id)
    page.locator("#message-input").fill("draft for session 2")
    page.wait_for_function(
        f"localStorage.getItem('tsugite_draft_{s2.id}') === 'draft for session 2'",
        timeout=2000,
    )

    _select(s1.id)
    page.wait_for_function(
        "document.getElementById('message-input').value === 'draft for session 1'",
        timeout=2000,
    )
    assert page.locator("#message-input").input_value() == "draft for session 1"

    _select(s2.id)
    page.wait_for_function(
        "document.getElementById('message-input').value === 'draft for session 2'",
        timeout=2000,
    )
    assert page.locator("#message-input").input_value() == "draft for session 2"
