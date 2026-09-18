"""The mid-turn message queue lives on the session, so two tabs share it.

`mock_chat` swaps in a fake `handle_message`; `delay=` holds the turn in flight
so a send made during it is queued by the daemon rather than run.
"""

from playwright.sync_api import expect

from .helpers import wait_for_authed

CHIP = ".queuedrow .t-chip"


def _compose(page, text):
    textarea = page.get_by_role("textbox", name="Message", exact=True)
    textarea.fill(text)
    textarea.press("Enter")


def test_a_message_queued_in_one_tab_shows_in_another_and_either_can_remove_it(chat_page, mock_chat, base_url):
    mock_chat("Done after a beat", delay=5)

    tab_a = chat_page
    _compose(tab_a, "read the logs")
    expect(tab_a.locator('[data-act="stop"]')).to_be_visible(timeout=5000)

    _compose(tab_a, "and also check the disk")
    expect(tab_a.locator(CHIP)).to_contain_text("and also check the disk", timeout=5000)

    tab_b = tab_a.context.new_page()
    tab_b.goto(base_url)
    wait_for_authed(tab_b)

    # Read from the sessions payload on a cold load, not from anything tab A holds.
    expect(tab_b.locator(CHIP)).to_contain_text("and also check the disk", timeout=10000)

    tab_b.get_by_role("button", name="Remove queued message 1").click()

    # Tab A never reloads: the removal reaches it over the session broadcast.
    expect(tab_a.locator(CHIP)).to_have_count(0, timeout=10000)
    expect(tab_b.locator(CHIP)).to_have_count(0)
    tab_b.close()
