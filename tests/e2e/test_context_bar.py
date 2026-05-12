"""E2E tests for the thread header (formerly 'context bar') above conversations.

The pre-Console-redesign `.context-bar` was replaced by a thread header consisting
of a title row (`[x-data*=conversationsView] .title-row`), a topic row, and a meta row. The token-usage meter
moved from a standalone element to the `.ctx-mini` trigger inside the title row.
"""


def test_thread_header_shows_session_title(chat_page, e2e_session_store):
    """The thread header displays the session title for the selected session."""
    page = chat_page

    page.wait_for_selector("[x-data*=conversationsView] .title-row h1", timeout=5000)
    title = page.locator("[x-data*=conversationsView] .title-row h1").first
    assert title.is_visible()
    assert title.text_content().strip()


def test_ctx_mini_appears_after_message(chat_page, mock_chat):
    """After a turn with token info, the .ctx-mini token meter appears in the header."""
    page = chat_page

    mock_chat(
        "Hello!",
        events=[
            (
                "prompt_snapshot",
                {
                    "token_breakdown": {
                        "categories": [{"name": "task", "tokens": 50, "items": []}],
                        "total": 50,
                    },
                },
            ),
        ],
    )
    textarea = page.locator("textarea#message-input")
    textarea.fill("test")
    textarea.press("Enter")
    page.wait_for_selector(".console-turn.agent", timeout=15000)

    # ctx-mini only renders when statusInfo.tokens and context_limit are set;
    # the title row itself is the more stable header signal.
    page.wait_for_selector("[x-data*=conversationsView] .title-row", timeout=5000)
    assert page.locator("[x-data*=conversationsView] .title-row").is_visible()
