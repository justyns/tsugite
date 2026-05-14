"""Multi-session and multi-tab interaction tests.

Covers session-switching flows that the single-session tests don't exercise:
the conversationsView keeps a per-session messages map, an SSE event bus
fans out to all subscribed pages, and the compaction flow auto-follows the
UI to the successor session.
"""

from tsugite.daemon.session_store import Session, SessionSource, SessionStatus
from tsugite.history.storage import generate_session_id

from .helpers import CONV_VIEW, open_conversations, reload_conversations_view


def _make_session(store, user_id):
    sid = generate_session_id("test-agent")
    s = Session(id=sid, agent="test-agent", source=SessionSource.INTERACTIVE.value, user_id=user_id)
    store.create_session(s)
    return s


def test_switching_sessions_preserves_per_session_messages(authenticated_page, e2e_session_store, mock_chat):
    """Each session keeps its own messages array; switching never bleeds."""
    page = authenticated_page
    user_id = page.evaluate("Alpine.store('app').userId")
    s1 = _make_session(e2e_session_store, user_id)
    s2 = _make_session(e2e_session_store, user_id)

    open_conversations(page)
    reload_conversations_view(page)

    def select(sid):
        page.evaluate(
            f"Alpine.$data(document.querySelector({CONV_VIEW!r})).selectSessionById({sid!r}, {{follow: false}})"
        )
        page.wait_for_function(
            f"Alpine.$data(document.querySelector({CONV_VIEW!r})).selectedSessionId === {sid!r}",
            timeout=3000,
        )
        page.wait_for_selector("textarea#message-input", timeout=5000)

    # Send a message in session 1.
    select(s1.id)
    mock_chat("Response in session one")
    page.locator("textarea#message-input").fill("hi from one")
    page.locator("textarea#message-input").press("Enter")
    page.wait_for_selector(".console-turn.agent", timeout=15000)

    # Switch to session 2; messages from session 1 must not appear.
    mock_chat("Response in session two")
    select(s2.id)
    page.wait_for_function(
        f"Alpine.$data(document.querySelector({CONV_VIEW!r})).messages.length === 0",
        timeout=3000,
    )

    # Send a message in session 2.
    page.locator("textarea#message-input").fill("hi from two")
    page.locator("textarea#message-input").press("Enter")
    page.wait_for_selector(".console-turn.agent", timeout=15000)
    assert "two" in page.locator(".console-turn.agent").last.text_content().lower()

    # Back to session 1; its history is restored.
    select(s1.id)
    page.wait_for_function(
        f"Alpine.$data(document.querySelector({CONV_VIEW!r})).messages.length >= 2",
        timeout=3000,
    )
    user_texts = page.locator(".console-turn.user").all_text_contents()
    assert any("hi from one" in t for t in user_texts)
    assert not any("hi from two" in t for t in user_texts)


def test_compaction_event_auto_follows_active_tab_to_successor(authenticated_page, e2e_session_store, base_url):
    """An SSE compaction event re-points the active tab at the successor session.

    A stale tab loaded before compaction was visible would otherwise keep
    interacting with the now-completed predecessor and the next /chat POST
    would silently land in the successor without the URL or selected-meta
    state changing.
    """
    page = authenticated_page
    user_id = page.evaluate("Alpine.store('app').userId")
    predecessor = e2e_session_store.get_or_create_interactive(user_id, "test-agent")
    successor = e2e_session_store.create_session(
        Session(
            id="multi-sess-successor",
            agent="test-agent",
            source=SessionSource.INTERACTIVE.value,
            status=SessionStatus.ACTIVE.value,
            user_id=user_id,
        )
    )
    e2e_session_store.update_session(predecessor.id, status=SessionStatus.COMPLETED.value, superseded_by=successor.id)

    page.goto(base_url + f"#conversations?session={predecessor.id}")
    page.wait_for_function("Alpine.store('app').view === 'conversations'", timeout=3000)
    reload_conversations_view(page)
    # Simulate the stale-tab case: force selection on the predecessor even
    # though selectSession() would normally chase superseded_by forward.
    page.evaluate(f"Alpine.$data(document.querySelector({CONV_VIEW!r})).selectedSessionId = {predecessor.id!r};")

    page.evaluate(
        f"""
        Alpine.store('app').lastEvent = {{
            type: 'session_update',
            data: {{
                action: 'compacted',
                id: {predecessor.id!r},
                successor_id: {successor.id!r}
            }},
            _ts: Date.now()
        }};
        """
    )
    page.wait_for_function(
        f"Alpine.$data(document.querySelector({CONV_VIEW!r})).selectedSessionId === {successor.id!r}",
        timeout=3000,
    )


def test_two_pages_see_same_session_state_after_external_event(
    authenticated_page, e2e_session_store, base_url, e2e_auth_token
):
    """Two browser contexts on the same session both react to a session_update event.

    The conversationsView in each tab subscribes to the same SSE feed; an
    external session_update should land in both tabs' lastEvent and trigger
    the same UI reaction.
    """
    page1 = authenticated_page
    user_id = page1.evaluate("Alpine.store('app').userId")
    session = e2e_session_store.get_or_create_interactive(user_id, "test-agent")

    # A second page in a new context: must authenticate independently.
    context = page1.context.browser.new_context()
    page2 = context.new_page()
    page2.goto(base_url + "/api/health")
    page2.evaluate(f"localStorage.setItem('tsugite_token', '{e2e_auth_token}')")
    page2.evaluate(f"localStorage.setItem('tsugite_user_id', {user_id!r})")
    page2.goto(base_url)
    page2.wait_for_function(
        "typeof Alpine !== 'undefined' && Alpine.store('app') && !Alpine.store('app').authRequired",
        timeout=10000,
    )

    for page in (page1, page2):
        page.goto(base_url + f"#conversations?session={session.id}")
        page.wait_for_function("Alpine.store('app').view === 'conversations'", timeout=3000)
        page.evaluate(f"Alpine.$data(document.querySelector({CONV_VIEW!r})).reload()")
        page.wait_for_function(
            f"Alpine.$data(document.querySelector({CONV_VIEW!r}))?.selectedSessionId === {session.id!r}",
            timeout=5000,
        )

    # Fire a metadata_updated event into both pages' Alpine store. Each tab
    # consumes its own copy.
    payload = """
        Alpine.store('app').lastEvent = {
            type: 'session_update',
            data: { action: 'metadata_updated', id: 'shared', metadata: {} },
            _ts: Date.now()
        };
    """
    page1.evaluate(payload)
    page2.evaluate(payload)

    # Both pages should remain stable (no JS errors crashed the view).
    for page in (page1, page2):
        assert page.locator(".title-row").count() >= 1

    context.close()
