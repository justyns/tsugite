"""The context-usage bar must not vanish when a partial session_info event
arrives: updateStatusFromEvent replaced statusInfo wholesale, so an event
without tokens/context_limit wiped previously-good data and blanked the bar
until a page reload (confirmed live: reload restored the bar)."""

from .helpers import CONV_VIEW


def test_partial_session_info_does_not_blank_context_bar(chat_page):
    page = chat_page

    result = page.evaluate(
        """(sel) => {
            const v = Alpine.$data(document.querySelector(sel));
            const sid = v.selectedSessionId;
            // Authoritative /status data has landed:
            v._sessionState(sid).statusInfo = {
                model: 'claude_code:opus', tokens: 381990, context_limit: 1000000,
                threshold: 0.8, message_count: 12, attachments: [],
            };
            // A partial session_info event (interrupted turn / resume path)
            // arrives with no token data:
            v.updateStatusFromEvent({ model: 'claude_code:opus' }, sid);
            const si = v._sessionState(sid).statusInfo;
            return { tokens: si.tokens ?? null, limit: si.context_limit ?? null };
        }""",
        CONV_VIEW,
    )
    page.screenshot(path="/tmp/tsugite-issue-430-state.png", full_page=True)

    assert result["tokens"] == 381990, f"partial event wiped tokens (got {result['tokens']!r})"
    assert result["limit"] == 1000000, f"partial event wiped context_limit (got {result['limit']!r})"

    bar = page.locator('[x-show="statusInfo?.tokens != null && statusInfo?.context_limit"]').first
    assert bar.is_visible(), "context bar must stay visible after a partial session_info event"


def test_full_session_info_event_still_updates(chat_page):
    page = chat_page

    result = page.evaluate(
        """(sel) => {
            const v = Alpine.$data(document.querySelector(sel));
            const sid = v.selectedSessionId;
            v._sessionState(sid).statusInfo = {
                model: 'old-model', tokens: 100, context_limit: 1000,
                threshold: 0.8, message_count: 1, attachments: [],
            };
            v.updateStatusFromEvent({
                model: 'new-model', tokens: 500, context_limit: 2000,
                threshold: 0.9, message_count: 2, attachments: [],
            }, sid);
            return v._sessionState(sid).statusInfo;
        }""",
        CONV_VIEW,
    )
    assert result["tokens"] == 500
    assert result["context_limit"] == 2000
    assert result["model"] == "new-model"
