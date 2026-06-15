"""Live-progress bubble lifecycle: no duplicate "Working..." and no phantom progress-done.

Opening a session that another client/schedule is mid-run on (loadStatus injects a
"Working..." bubble) must not end up with two progress bubbles when the live SSE event
arrives; and a session that ends with no progress shown in this tab must not render an
empty `progress-done` (`code ✓/✗`) bubble.
"""

from tsugite.daemon.session_store import Session, SessionSource
from tsugite.history.storage import generate_session_id

from .helpers import CONV_VIEW, open_conversations, reload_conversations_view


def _make_session(store, user_id):
    sid = generate_session_id("test-agent")
    s = Session(id=sid, agent="test-agent", source=SessionSource.INTERACTIVE.value, user_id=user_id)
    store.create_session(s)
    return s


def _select(page, sid):
    page.evaluate(f"Alpine.$data(document.querySelector({CONV_VIEW!r})).selectSessionById({sid!r}, {{follow: false}})")
    page.wait_for_function(
        f"Alpine.$data(document.querySelector({CONV_VIEW!r})).selectedSessionId === {sid!r}",
        timeout=3000,
    )


def test_session_end_without_progress_makes_no_phantom_bubble(authenticated_page, e2e_session_store):
    page = authenticated_page
    user_id = page.evaluate("Alpine.store('app').userId")
    s = _make_session(e2e_session_store, user_id)
    open_conversations(page)
    reload_conversations_view(page)
    _select(page, s.id)

    result = page.evaluate(
        f"""() => {{
            const v = Alpine.$data(document.querySelector({CONV_VIEW!r}));
            v._sessionProgress = null;
            const before = v.messages.filter(m => m.type === 'progress' || m.type === 'progress-done').length;
            v._handleSessionEvent({{ session_id: v.selectedSessionId, event_type: 'session_complete' }});
            const after = v.messages.filter(m => m.type === 'progress' || m.type === 'progress-done').length;
            return {{ before, after }};
        }}"""
    )
    assert result["after"] == result["before"], "session_complete with no in-flight progress must not add a bubble"


def test_busy_session_open_does_not_duplicate_progress_bubble(authenticated_page, e2e_session_store):
    page = authenticated_page
    user_id = page.evaluate("Alpine.store('app').userId")
    s = _make_session(e2e_session_store, user_id)

    import json

    page.route(
        "**/api/agents/*/status*",
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({"busy": True, "pending_message": "do the thing"}),
        ),
    )
    open_conversations(page)
    reload_conversations_view(page)
    _select(page, s.id)

    count = page.evaluate(
        f"""async () => {{
            const v = Alpine.$data(document.querySelector({CONV_VIEW!r}));
            v.messages.length = 0;          // clean slate (a select-time loadStatus may have run)
            v._sessionProgress = null;
            await v.loadStatus();           // injects the "Working..." bubble for the busy session
            v._handleSessionEvent({{ session_id: v.selectedSessionId, event_type: 'turn_start', turn: 1 }});
            return v.messages.filter(m => m.type === 'progress' || m.type === 'progress-done').length;
        }}"""
    )
    assert count == 1, f"busy-session open + a live event must keep exactly one progress bubble, got {count}"
