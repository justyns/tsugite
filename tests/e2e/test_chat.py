"""Chat send/stream flow against the mocked adapter.

`mock_chat` swaps in a fast fake `handle_message`; `delay=` holds the turn in
flight so a test can observe mid-turn state.
"""

from playwright.sync_api import expect


def test_send_message_shows_streamed_response(chat_page, mock_chat):
    mock_chat("I can help with that!")

    page = chat_page
    textarea = page.get_by_role("textbox", name="Message", exact=True)
    textarea.fill("Hello agent")
    textarea.press("Enter")

    expect(page.locator(".t-msg--user").last).to_contain_text("Hello agent")
    expect(page.locator(".t-msg--ai").last).to_contain_text("I can help with that!", timeout=15000)


def test_stop_button_flips_while_streaming(chat_page, mock_chat):
    mock_chat("Done after a beat", delay=0.6)

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


def test_waiting_line_shows_the_agent_turn_budget(chat_page, mock_chat):
    """The in-flight Work line reports how far through its turns the loop is."""
    mock_chat("Done after a beat", events=[("turn_start", {"turn": 3, "max_turns": 20})], delay=0.6)

    page = chat_page
    textarea = page.get_by_role("textbox", name="Message", exact=True)
    textarea.fill("Take your time")
    textarea.press("Enter")

    work = page.locator(".t-work")
    expect(work).to_be_visible(timeout=5000)
    expect(work).to_contain_text("turn 3 / 20")


def test_soft_line_breaks_render_hard_in_the_persons_own_message_only(chat_page, mock_chat):
    """The agent's reply keeps CommonMark: a model that hard-wraps its prose
    must not gain a break mid-sentence."""
    mock_chat("first half of a thought\nsecond half of the same thought")

    page = chat_page
    textarea = page.get_by_role("textbox", name="Message", exact=True)
    textarea.fill("https://example.test/a\nhttps://example.test/b\nhttps://example.test/c")
    textarea.press("Enter")

    user_bubble = page.locator(".t-msg--user").last
    expect(user_bubble).to_contain_text("example.test/c")
    expect(user_bubble.locator("br")).to_have_count(2)

    ai_bubble = page.locator(".t-msg--ai").last
    expect(ai_bubble).to_contain_text("second half", timeout=15000)
    expect(ai_bubble.locator("br")).to_have_count(0)
