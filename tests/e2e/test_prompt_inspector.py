"""Prompt inspector E2E tests."""

_SAMPLE_SNAPSHOT = (
    "prompt_snapshot",
    {
        "token_breakdown": {"system": 5, "context": 0, "history": 0, "task": 2, "steps": 0, "total": 7},
    },
)


def test_context_link_opens_inspector(chat_page, mock_chat):
    """Clicking the Context stat in the status bar opens the prompt inspector modal."""
    mock_chat("Done!", events=[_SAMPLE_SNAPSHOT])

    page = chat_page
    textarea = page.locator("textarea").first
    textarea.fill("test task")
    textarea.press("Enter")

    page.wait_for_selector(".msg.agent", timeout=15000)

    context_link = page.locator("#status-bar .att-link", has_text="k /")
    context_link.click()

    page.wait_for_selector(".prompt-inspector", timeout=3000)
    assert page.locator(".token-bar").is_visible()
    assert page.locator(".token-legend").is_visible()

    # Close modal
    page.locator(".prompt-inspector .btn-cancel").click()
    page.wait_for_timeout(300)
    assert not page.locator(".prompt-inspector").is_visible()
