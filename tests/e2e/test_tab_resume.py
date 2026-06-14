"""Tab-resume catch-up.

A conversation must refetch history when the tab becomes visible again, so a reply
made on another device shows up without a manual reload. The SSE reconnect after
resume suppresses its 'reconnect' catch-up event (fresh closure => everConnected
starts false) and missed events aren't replayed, so the live stream can't be relied
on — an explicit refetch on resume is required.
"""

from datetime import datetime, timezone
from unittest.mock import patch

from tsugite.history.models import Event
from tsugite.history.storage import SessionStorage

from .test_scroll_behavior import _open_session, _seed_long_session, _wait_at_bottom


def test_tab_resume_refetches_open_conversation(authenticated_page, e2e_adapter, e2e_tmp):
    page = authenticated_page
    history_dir, user_id, session_id = _seed_long_session(e2e_adapter, e2e_tmp, "resume", turns=3)

    with patch("tsugite.daemon.adapters.http.get_history_dir", return_value=history_dir):
        _open_session(page, user_id, session_id)
        assert "CROSSDEVICE" not in (page.locator("#messages").text_content() or "")

        # A reply lands on another device while this tab is backgrounded: appended to
        # the session JSONL but NOT delivered as a live SSE event to this tab.
        now = datetime.now(timezone.utc)
        SessionStorage.load(history_dir / f"{session_id}.jsonl").record_many(
            [
                Event(type="user_input", ts=now, data={"text": "CROSSDEVICE question"}),
                Event(type="final_result", ts=now, data={"result": "CROSSDEVICE answer"}),
            ]
        )

        # Tab becomes visible again -> the view must refetch and show the new reply.
        page.evaluate("document.dispatchEvent(new Event('visibilitychange'))")
        page.wait_for_function(
            "() => document.getElementById('messages').textContent.includes('CROSSDEVICE answer')",
            timeout=3000,
        )


def test_tab_resume_does_not_yank_when_scrolled_up(authenticated_page, e2e_adapter, e2e_tmp):
    """If the user has scrolled up to read history, returning to the tab must NOT yank
    them to the bottom — the resume refetch is gated on isAtBottom (loadHistory
    force-scrolls, so refetching while scrolled up would be disruptive)."""
    page = authenticated_page
    history_dir, user_id, session_id = _seed_long_session(e2e_adapter, e2e_tmp, "resume-up", turns=14)

    with patch("tsugite.daemon.adapters.http.get_history_dir", return_value=history_dir):
        _open_session(page, user_id, session_id)
        _wait_at_bottom(page)

        # Scroll up to read history.
        page.evaluate(
            "(() => { const el = document.getElementById('messages');"
            " el.scrollTop = 0; el.dispatchEvent(new Event('scroll')); })()"
        )
        before = page.evaluate("document.getElementById('messages').scrollTop")
        assert before == 0

        # Refocus the tab. The refetch must be skipped, leaving scroll position alone.
        page.evaluate("document.dispatchEvent(new Event('visibilitychange'))")
        page.wait_for_timeout(400)  # cover the 200ms debounce
        after = page.evaluate("document.getElementById('messages').scrollTop")
        assert after == before, "resume must not yank a scrolled-up reader to the bottom"
