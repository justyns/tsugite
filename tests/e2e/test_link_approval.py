"""Link-paste approval prompt, driven through the real daemon and a real browser.

The fake `handle_message` here runs the genuine tsugite-web URL detector on a
worker thread, exactly as the production send path does, so the options the
browser renders are the ones `request_approval` built and the button the test
clicks is what `_may_fetch` acts on.
"""

import asyncio
import re
from unittest.mock import AsyncMock

import pytest
import yaml
from playwright.sync_api import expect
from tsugite_web import context as web_ctx

from tsugite.config import get_xdg_write_path

ASK_APPROVAL = '[data-testid="ask-approval"]'


@pytest.fixture
def link_detector_chat(e2e_adapter):
    """Swap in a turn that gates a pasted link through the real approval path.

    Yields the dict the detector's return value lands in.
    """
    original = e2e_adapter.handle_message
    result: dict = {}

    async def fake_handle(user_id, message, channel_context, custom_logger=None):
        result["items"] = await asyncio.to_thread(web_ctx.detect_urls, message, {})
        return "done"

    e2e_adapter.handle_message = AsyncMock(side_effect=fake_handle)
    yield result
    e2e_adapter.handle_message = original


def _send(page, message: str):
    textarea = page.get_by_role("textbox", name="Message", exact=True)
    textarea.fill(message)
    textarea.press("Enter")


def test_always_deny_persists_the_host_and_silences_later_pastes(chat_page, link_detector_chat):
    page = chat_page
    _send(page, "read https://blocked.example/post for me")

    page.wait_for_selector(ASK_APPROVAL, timeout=15000)
    expect(page.locator(f"{ASK_APPROVAL} button")).to_have_text(
        [re.compile(r"Approve$"), re.compile(r"Deny$"), re.compile(r"Always allow$"), re.compile(r"Always deny$")]
    )

    page.locator(ASK_APPROVAL).get_by_role("button", name="Always deny").click()

    expect(page.locator(".t-msg--ai").last).to_contain_text("done", timeout=15000)
    expect(page.locator('[data-act="send"]')).to_be_visible()
    assert link_detector_chat["items"] == []

    stored = yaml.safe_load(get_xdg_write_path("permissions.yaml").read_text())
    assert stored["web"]["fetch_denylist"] == ["blocked.example"]

    _send(page, "and https://blocked.example/other too")

    expect(page.locator(".t-msg--ai")).to_have_count(2, timeout=15000)
    assert link_detector_chat["items"] == []
    assert page.locator(ASK_APPROVAL).count() == 0
