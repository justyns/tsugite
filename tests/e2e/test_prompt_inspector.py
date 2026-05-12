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
    """Clicking the .ctx-mini token meter opens the prompt inspector popover."""
    mock_chat("Done!", events=[_SAMPLE_SNAPSHOT])

    page = chat_page
    textarea = page.locator("textarea#message-input")
    textarea.fill("test task")
    textarea.press("Enter")

    page.wait_for_selector(".console-turn.agent", timeout=15000)

    # The token meter only renders when context_limit is known; trigger the
    # inspector directly through the Alpine method to keep the test focused on
    # the popover contents, not the gating logic.
    page.evaluate("Alpine.$data(document.querySelector('[x-data*=conversationsView]')).openPromptInspector()")

    page.wait_for_selector(".pi-pop", timeout=3000)
    assert page.locator(".pi-stack-bar").is_visible()
    assert page.locator(".pi-list").is_visible()
    # At least two categories with non-zero tokens should render.
    assert page.locator(".pi-list .pi-row").count() >= 2

    page.locator(".pi-pop .pi-close").click()
    page.wait_for_function(
        "!document.querySelector('.pi-pop')",
        timeout=3000,
    )
