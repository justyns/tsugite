"""Session management tests."""


def test_create_and_switch_sessions(authenticated_page, e2e_session_store, mock_chat):
    """Creating sessions and switching between them keeps messages isolated."""
    mock_chat("Response for session A")

    page = authenticated_page
    user_id = page.evaluate("Alpine.store('app').userId")

    # Create two sessions
    e2e_session_store.get_or_create_interactive(user_id, "test-agent")
    from tsugite_daemon.session_store import Session, SessionSource

    from tsugite.history.storage import generate_session_id

    s2_id = generate_session_id("test-agent")
    s2 = Session(id=s2_id, agent="test-agent", source=SessionSource.INTERACTIVE.value, user_id=user_id)
    e2e_session_store.create_session(s2)

    page.locator(".console-tabs button.console-tab", has_text="Conversations").click()
    page.wait_for_function("Alpine.store('app').view === 'conversations'", timeout=3000)

    # Reload the session list to pick up the just-created sessions.
    page.evaluate("Alpine.$data(document.querySelector('[x-data*=conversationsView]')).reload()")
    page.wait_for_function(
        f"(() => {{ const v = Alpine.$data(document.querySelector('[x-data*=conversationsView]')); "
        f"return v && v.allSessions && v.allSessions.some(s => s.id === {s2_id!r}); }})()",
        timeout=5000,
    )
    page.wait_for_selector(".console-session", timeout=5000)

    # Select first session and send a message
    page.locator(".console-session").first.click()
    page.wait_for_selector("textarea#message-input", timeout=3000)

    textarea = page.locator("textarea#message-input")
    textarea.fill("Message in session 1")
    textarea.press("Enter")
    page.wait_for_selector(".console-turn.agent", timeout=15000)

    # Switch to second session — messages from session 1 should not appear
    mock_chat("Response for session B")
    page.locator(".console-session").nth(1).click()
    page.wait_for_function(
        "Alpine.$data(document.querySelector('[x-data*=conversationsView]')).messages.length === 0",
        timeout=3000,
    )

    user_msgs = page.locator(".console-turn.user")
    assert user_msgs.count() == 0
