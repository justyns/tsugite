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


def test_tool_call_progress_display(chat_page, mock_chat):
    """Tool calls during chat should render as expandable progress steps."""
    mock_chat(
        "Found 3 files.",
        events=[
            ("tool_result", {"tool": "list_files", "output": "file1.txt\nfile2.txt\nfile3.txt", "success": True}),
        ],
    )

    page = chat_page
    textarea = page.locator("textarea").first
    textarea.fill("List the files")
    textarea.press("Enter")

    page.wait_for_selector(".msg.agent", timeout=15000)
    # Progress section should exist with the tool step
    progress = page.locator(".step-row, details")
    assert progress.count() > 0
    assert "list_files" in page.locator(".step-row, details").first.text_content()


def test_error_displayed_on_failure(chat_page, e2e_adapter):
    """When the agent raises, the error message should appear in chat."""
    from unittest.mock import AsyncMock

    async def failing_handle(user_id, message, channel_context, custom_logger=None):
        raise RuntimeError("Agent crashed: out of memory")

    e2e_adapter.handle_message = AsyncMock(side_effect=failing_handle)

    page = chat_page
    textarea = page.locator("textarea").first
    textarea.fill("Do something")
    textarea.press("Enter")

    # Error appears inside collapsed progress details — open it, then check
    page.wait_for_selector(".msg.progress", timeout=10000)
    page.locator(".msg.progress details summary").first.click()
    page.wait_for_selector(".err", timeout=3000)
    assert "out of memory" in page.locator(".err").first.text_content()
