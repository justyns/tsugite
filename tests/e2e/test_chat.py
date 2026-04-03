"""Chat interaction tests."""


def test_send_message_shows_response(chat_page, mock_chat):
    mock_chat("I can help with that!")

    page = chat_page
    textarea = page.locator("textarea").first
    textarea.fill("Hello agent")
    textarea.press("Enter")

    page.wait_for_selector(".msg.user", timeout=5000)
    page.wait_for_selector(".msg.agent", timeout=15000)
    assert "I can help with that!" in page.locator(".msg.agent").last.text_content()


def test_reaction_emoji_appears(chat_page, mock_chat):
    mock_chat("Done!", events=[("reaction", {"emoji": "👍"})])

    page = chat_page
    textarea = page.locator("textarea").first
    textarea.fill("React to this")
    textarea.press("Enter")

    page.wait_for_selector(".msg.agent", timeout=15000)
    page.wait_for_selector(".reaction-badge", timeout=3000)
    assert page.locator(".reaction-badge").first.text_content().strip() == "👍"
