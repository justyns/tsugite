"""Prompt inspector E2E tests."""

_SAMPLE_SNAPSHOT = (
    "prompt_snapshot",
    {
        "token_breakdown": {
            "categories": [
                {"name": "instructions", "tokens": 100, "items": []},
                {
                    "name": "tools",
                    "tokens": 200,
                    "items": [{"name": "read_file", "tokens": 120}, {"name": "write_file", "tokens": 80}],
                },
                {"name": "attachments", "tokens": 0, "items": []},
                {"name": "skills", "tokens": 0, "items": []},
                {"name": "history", "tokens": 0, "items": []},
                {"name": "task", "tokens": 50, "items": []},
                {"name": "steps", "tokens": 0, "items": []},
            ],
            "total": 350,
        },
    },
)


def test_context_link_opens_inspector(chat_page, mock_chat):
    """Clicking the context meter in the context bar opens the prompt inspector modal."""
    mock_chat("Done!", events=[_SAMPLE_SNAPSHOT])

    page = chat_page
    textarea = page.locator("textarea").first
    textarea.fill("test task")
    textarea.press("Enter")

    page.wait_for_selector(".msg.agent", timeout=15000)

    context_meter = page.locator(".context-meter")
    if context_meter.is_visible():
        context_meter.click()
    else:
        page.locator("#status-bar .att-link", has_text="k /").click()

    page.wait_for_selector(".prompt-inspector", timeout=3000)
    assert page.locator(".token-bar").is_visible()
    assert page.locator(".pi-categories").is_visible()

    # Should show categories with items
    assert page.locator(".pi-cat").count() >= 2

    # Close modal
    page.locator(".prompt-inspector .btn-cancel").click()
    page.wait_for_timeout(300)
    assert not page.locator(".prompt-inspector").is_visible()
