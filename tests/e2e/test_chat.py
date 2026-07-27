"""Chat send/stream flow against the mocked adapter.

`mock_chat` swaps in a fast fake `handle_message`; the second test needs the
"in flight" window to actually be observable, so it overrides
`e2e_adapter.handle_message` directly with a slow fake (same pattern the old
suite used for its error-path test) after still calling `mock_chat(...)`
first so its fixture-level teardown restores the real (tripwired)
`handle_message` afterwards.
"""

import asyncio
from unittest.mock import AsyncMock

from playwright.sync_api import expect


def test_send_message_shows_streamed_response(chat_page, mock_chat):
    mock_chat("I can help with that!")

    page = chat_page
    textarea = page.get_by_role("textbox", name="Message", exact=True)
    textarea.fill("Hello agent")
    textarea.press("Enter")

    expect(page.locator(".t-msg--user").last).to_contain_text("Hello agent")
    expect(page.locator(".t-msg--ai").last).to_contain_text("I can help with that!", timeout=15000)


def test_stop_button_flips_while_streaming(chat_page, mock_chat, e2e_adapter):
    mock_chat("placeholder")

    async def slow_handle(user_id, message, channel_context, custom_logger=None):
        await asyncio.sleep(0.6)
        return "Done after a beat"

    e2e_adapter.handle_message = AsyncMock(side_effect=slow_handle)

    page = chat_page
    textarea = page.get_by_role("textbox", name="Message", exact=True)
    textarea.fill("Take your time")
    textarea.press("Enter")

    send_button = page.locator('[data-act="send"]')
    stop_button = page.locator('[data-act="stop"]')
    expect(stop_button).to_be_visible(timeout=3000)
    expect(send_button).to_have_count(0)

    expect(page.locator(".t-msg--ai").last).to_contain_text("Done after a beat", timeout=15000)
    expect(send_button).to_be_visible()
    expect(stop_button).to_have_count(0)
