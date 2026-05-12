"""Chat interaction tests."""


def test_send_message_shows_response(chat_page, mock_chat):
    mock_chat("I can help with that!")

    page = chat_page
    textarea = page.locator("textarea#message-input")
    textarea.fill("Hello agent")
    textarea.press("Enter")

    page.wait_for_selector(".console-turn.user", timeout=5000)
    page.wait_for_selector(".console-turn.agent", timeout=15000)
    assert "I can help with that!" in page.locator(".console-turn.agent").last.text_content()


def test_reaction_emoji_appears(chat_page, mock_chat):
    mock_chat("Done!", events=[("reaction", {"emoji": "👍"})])

    page = chat_page
    textarea = page.locator("textarea#message-input")
    textarea.fill("React to this")
    textarea.press("Enter")

    page.wait_for_selector(".console-turn.agent", timeout=15000)
    # Reactions attach to the user_input bubble as inline emoji spans.
    user_bubble = page.locator(".console-turn.user .console-turn-bubble").first
    page.wait_for_function(
        "document.querySelector('.console-turn.user .console-turn-bubble')?.textContent?.includes('👍')",
        timeout=3000,
    )
    assert "👍" in user_bubble.text_content()


def test_tool_call_progress_display(chat_page, mock_chat):
    """Tool calls during chat should render as expandable progress steps."""
    mock_chat(
        "Found 3 files.",
        events=[
            ("tool_result", {"tool": "list_files", "output": "file1.txt\nfile2.txt\nfile3.txt", "success": True}),
        ],
    )

    page = chat_page
    textarea = page.locator("textarea#message-input")
    textarea.fill("List the files")
    textarea.press("Enter")

    page.wait_for_selector(".console-turn.agent", timeout=15000)
    steps = page.locator(".console-codeblock .tool-step")
    assert steps.count() > 0
    assert "list_files" in steps.first.text_content()


def test_error_displayed_on_failure(chat_page, e2e_adapter):
    """When the agent raises, the error message should appear in the progress block."""
    from unittest.mock import AsyncMock

    async def failing_handle(user_id, message, channel_context, custom_logger=None):
        raise RuntimeError("Agent crashed: out of memory")

    e2e_adapter.handle_message = AsyncMock(side_effect=failing_handle)

    page = chat_page
    textarea = page.locator("textarea#message-input")
    textarea.fill("Do something")
    textarea.press("Enter")

    # Error event becomes a tool-step with an .err span inside the progress block.
    page.wait_for_selector(".console-codeblock .tool-row .err", timeout=10000)
    err_text = page.locator(".console-codeblock .tool-row").first.text_content()
    assert "out of memory" in err_text
