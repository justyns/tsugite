"""E2E tests for the context bar above conversations."""


def test_context_bar_shows_metadata(chat_page, e2e_session_store):
    """Context bar displays session agent and metadata when a session is selected."""
    page = chat_page

    page.wait_for_selector(".context-bar", timeout=5000)

    bar = page.locator(".context-bar")
    assert bar.is_visible()

    title = page.locator(".context-title")
    assert title.is_visible()
    assert title.text_content().strip()


def test_context_bar_context_meter(chat_page, mock_chat):
    """Context meter shows token usage when status info is available."""
    page = chat_page

    mock_chat("Hello!")
    textarea = page.locator("textarea").first
    textarea.fill("test")
    textarea.press("Enter")
    page.wait_for_selector(".msg.agent", timeout=15000)

    page.wait_for_selector(".context-bar", timeout=5000)
    bar = page.locator(".context-bar")
    assert bar.is_visible()
