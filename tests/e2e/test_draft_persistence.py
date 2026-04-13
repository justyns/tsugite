"""Draft input persistence across reloads and session switches."""

from tsugite.daemon.session_store import Session, SessionSource
from tsugite.history.storage import generate_session_id


def _new_session(e2e_session_store, user_id):
    """Create a fresh interactive session."""
    sid = generate_session_id("test-agent")
    s = Session(id=sid, agent="test-agent", source=SessionSource.INTERACTIVE.value, user_id=user_id)
    e2e_session_store.create_session(s)
    return s


def _go_to_conversations(page):
    """Navigate to conversations and wait for session list to load."""
    page.locator("nav button", has_text="Conversations").click()
    page.wait_for_function("Alpine.store('app').view === 'conversations'", timeout=3000)
    page.wait_for_timeout(1500)


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
    page.wait_for_timeout(400)

    # Verify it was saved
    draft_key = f"tsugite_draft_{session.id}"
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

    page.goto(f"{base_url}#conversations?session={session.id}")
    page.wait_for_function("!Alpine.store('app').authRequired", timeout=5000)
    page.wait_for_selector("#message-input", timeout=10000)

    page.locator("#message-input").fill("message to send")
    page.wait_for_timeout(400)

    page.locator("#message-input").press("Enter")
    page.wait_for_selector(".msg.agent", timeout=15000)

    draft = page.evaluate(
        "localStorage.getItem(Object.keys(localStorage).find(k => k.startsWith('tsugite_draft_')) || 'missing')"
    )
    assert draft is None


def test_drafts_isolated_between_sessions(authenticated_page, e2e_session_store):
    """Each session should maintain its own independent draft."""
    page = authenticated_page
    user_id = page.evaluate("Alpine.store('app').userId")

    _new_session(e2e_session_store, user_id)
    _new_session(e2e_session_store, user_id)

    # Reload to pick up sessions
    page.reload()
    page.wait_for_function("!Alpine.store('app').authRequired", timeout=5000)
    _go_to_conversations(page)

    items = page.locator(".turn-item[role='button']")
    assert items.count() >= 2

    # Type draft in first session
    items.first.click()
    page.wait_for_selector("#message-input", timeout=5000)
    page.locator("#message-input").fill("draft for session 1")
    page.wait_for_timeout(400)

    # Switch to second session and type different draft
    items.nth(1).click()
    page.wait_for_timeout(500)
    page.locator("#message-input").fill("draft for session 2")
    page.wait_for_timeout(400)

    # Switch back to first - should restore its draft
    items.first.click()
    page.wait_for_timeout(500)
    assert page.locator("#message-input").input_value() == "draft for session 1"

    # Switch to second again - should restore its draft
    items.nth(1).click()
    page.wait_for_timeout(500)
    assert page.locator("#message-input").input_value() == "draft for session 2"
