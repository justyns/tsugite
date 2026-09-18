"""The mid-turn message queue lives on the session, so two tabs share it.

`mock_chat` swaps in a fake `handle_message`. `delay=` holds the turn in flight, so
a send made during it is queued by the daemon.
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

    # Tab A never reloads. The session broadcast repaints it.
    expect(tab_a.locator(CHIP)).to_have_count(0, timeout=10000)
    expect(tab_b.locator(CHIP)).to_have_count(0)
    tab_b.close()


def test_a_queued_message_runs_at_turn_end_with_no_client_action(chat_page, mock_chat):
    """The daemon flushes the queue itself, so the chip clears and the second
    turn renders live in a tab that did nothing but watch."""
    mock_chat("Done after a beat", delay=2)

    page = chat_page
    _compose(page, "read the logs")
    expect(page.locator('[data-act="stop"]')).to_be_visible(timeout=5000)

    _compose(page, "and also check the disk")
    expect(page.locator(CHIP)).to_contain_text("and also check the disk", timeout=5000)

    expect(page.locator(CHIP)).to_have_count(0, timeout=20000)
    expect(page.locator(".t-msg--ai")).to_have_count(2, timeout=20000)
