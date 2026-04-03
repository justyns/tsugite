"""Session management tests."""


def test_create_and_switch_sessions(authenticated_page, e2e_session_store, mock_chat):
    """Creating sessions and switching between them keeps messages isolated."""
    mock_chat("Response for session A")

    page = authenticated_page
    user_id = page.evaluate("Alpine.store('app').userId")

    # Create two sessions
    s1 = e2e_session_store.get_or_create_interactive(user_id, "test-agent")
    from tsugite.daemon.session_store import Session, SessionSource
    from tsugite.history.storage import generate_session_id

    s2_id = generate_session_id("test-agent")
    s2 = Session(id=s2_id, agent="test-agent", source=SessionSource.INTERACTIVE.value, user_id=user_id)
    e2e_session_store.create_session(s2)

    page.locator("nav button", has_text="Conversations").click()
    page.wait_for_function("Alpine.store('app').view === 'conversations'", timeout=3000)

    # Wait for session list to load and show both sessions
    page.wait_for_selector(".turn-item", timeout=5000)

    # Select first session and send a message
    page.locator(".turn-item").first.click()
    page.wait_for_selector("textarea", timeout=3000)

    textarea = page.locator("textarea").first
    textarea.fill("Message in session 1")
    textarea.press("Enter")
    page.wait_for_selector(".msg.agent", timeout=15000)

    # Switch to second session — messages from session 1 should not appear
    mock_chat("Response for session B")
    page.locator(".turn-item").nth(1).click()
    page.wait_for_timeout(500)

    user_msgs = page.locator(".msg.user")
    assert user_msgs.count() == 0
