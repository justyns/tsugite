"""E2E tests for sidebar unread badge clearing while user is actively viewing a session.

Bug: `_markSessionViewed()` only fires on selectSession (switch) and visibilitychange
(refocus). When a message arrives in the already-selected session in a focused tab,
the sidebar badge stays unread until the user clicks away and back.
"""

from .helpers import CONV_VIEW


def _force_unread(page, sid):
    page.evaluate(
        f"""
        const view = Alpine.$data(document.querySelector({CONV_VIEW!r}));
        const s = view.allSessions.find(x => x.id === {sid!r});
        s.unread = true;
        """
    )


def _is_unread(page, sid):
    return page.evaluate(
        f"(() => {{ const view = Alpine.$data(document.querySelector({CONV_VIEW!r})); "
        f"const s = view.allSessions.find(x => x.id === {sid!r}); return !!(s && s.unread); }})()"
    )


def _wait_for_send_complete(page):
    page.wait_for_function(
        f"!Alpine.$data(document.querySelector({CONV_VIEW!r})).sending",
        timeout=3000,
    )


def _send_and_await_reply(page):
    """Type "Hello agent", submit, wait for the agent turn + post-send settle.

    Used by the issue-#313 regression tests that need to drive a full
    user-message-then-mock-reply cycle from an already-marked-unread session.
    """
    textarea = page.locator("textarea#message-input")
    textarea.fill("Hello agent")
    textarea.press("Enter")
    page.wait_for_selector(".console-turn.agent", timeout=10000)
    _wait_for_send_complete(page)


def test_unread_badge_clears_when_stream_ends_in_active_focused_session(chat_page, mock_chat):
    """When user receives a reply in the currently-selected session and the tab is visible,
    the sidebar unread badge should clear at stream end - no need for a refocus or switch."""
    mock_chat("Here is your reply.")

    page = chat_page
    sid = page.evaluate(f"Alpine.$data(document.querySelector({CONV_VIEW!r})).selectedSessionId")
    assert sid, "chat_page should have a selected session"

    # Simulate the bug condition: server returned unread=true (user's own message
    # bumped last_active past last_viewed_at). selectSession's _markSessionViewed
    # already fired at chat_page setup so we have to re-set this manually.
    _force_unread(page, sid)
    page.wait_for_selector(".console-session .title.unread", timeout=2000)

    textarea = page.locator("textarea#message-input")
    textarea.fill("Hello agent")
    textarea.press("Enter")

    page.wait_for_selector(".console-turn.agent", timeout=10000)
    _wait_for_send_complete(page)
    page.screenshot(path="/tmp/tsugite-issue-state.png", full_page=True)

    assert not _is_unread(page, sid)
    assert page.locator(".console-session .title.unread").count() == 0


def test_unread_badge_stays_clear_after_metadata_update_post_stream_end(chat_page, mock_chat, e2e_session_store):
    """Issue #313: after the stream-end mark-viewed clear, an agent-side
    `session_metadata` call (and the broadcast `metadata_updated` SSE) must
    not flip the sidebar bold-unread back on.

    Pre-fc85c54 the metadata write bumped `last_active`, the next
    `/api/sessions` reload re-derived `unread = last_active > last_viewed_at`
    as True, and the client overwrote the post-`mark-viewed` clear ~200 ms
    later. The fix dropped the bump in `set_metadata_bulk` /
    `delete_metadata`; this test catches a regression that puts it back.
    """
    mock_chat("Here is your reply.")

    page = chat_page
    sid = page.evaluate(f"Alpine.$data(document.querySelector({CONV_VIEW!r})).selectedSessionId")
    assert sid

    _force_unread(page, sid)
    page.wait_for_selector(".console-session .title.unread", timeout=2000)
    _send_and_await_reply(page)
    assert not _is_unread(page, sid), "stream-end should have cleared the unread flag"

    # The e2e_server fixture doesn't wire `session_runner`, so the client's
    # mark-viewed POST silently 503s and `last_viewed_at` never lands on the
    # server. Simulate the successful POST so the unread derivation has
    # something to compare against.
    e2e_session_store.mark_viewed(sid)

    # Simulate the agent's housekeeping `session_metadata(key='status_text', value='idle')`
    # at the server level - the regression in #313 fires here if `set_metadata_bulk`
    # bumps `last_active` past the just-set `last_viewed_at`.
    e2e_session_store.set_metadata(sid, "status_text", "idle")

    # Force the sidebar reload that the `metadata_updated` SSE would normally
    # debounce. If `last_active` was bumped, /api/sessions returns unread=true
    # and the client overwrites the cleared flag.
    page.evaluate(f"Alpine.$data(document.querySelector({CONV_VIEW!r})).loadSessions()")
    page.wait_for_function(
        f"!Alpine.$data(document.querySelector({CONV_VIEW!r})).loading",
        timeout=3000,
    )
    page.screenshot(path="/tmp/tsugite-issue-313-state.png", full_page=True)

    assert not _is_unread(page, sid), "metadata update revived unread flag after mark-viewed - fc85c54 regression"
    assert page.locator(".console-session .title.unread").count() == 0


def test_unread_badge_stays_when_tab_hidden_during_stream_end(chat_page, mock_chat):
    """The fix must respect document.visibilityState: if the tab is hidden when the
    stream ends, the badge stays unread (user has not actually seen the reply yet -
    the next visibilitychange handler will clear it on refocus)."""
    mock_chat("Quiet background reply.")

    page = chat_page
    sid = page.evaluate(f"Alpine.$data(document.querySelector({CONV_VIEW!r})).selectedSessionId")
    assert sid

    _force_unread(page, sid)

    # Override the visibility API for the rest of the page lifetime.
    page.evaluate("Object.defineProperty(document, 'visibilityState', { configurable: true, get: () => 'hidden' });")

    textarea = page.locator("textarea#message-input")
    textarea.fill("Hello agent")
    textarea.press("Enter")

    page.wait_for_selector(".console-turn.agent", timeout=10000)
    _wait_for_send_complete(page)

    assert _is_unread(page, sid), "unread should stay set when the tab is hidden"
