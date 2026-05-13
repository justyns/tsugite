"""Sidebar shows a useful fallback (date / prompt / id) when a session has no title.

Reproduces: when a session has no title, the UI showed the bare username
(e.g. "justyn") as fallback. That isn't distinguishable when the same user
has multiple sessions. Better fallback priority: date > truncated prompt >
session ID > username.
"""

from tsugite.daemon.session_store import Session, SessionSource, SessionStatus

from .helpers import open_conversations, reload_conversations_view, wait_for_session_in_list


def _title_for(page, session_id: str) -> str:
    view_sel = "[x-data*=conversationsView]"
    return page.evaluate(
        f"(() => {{ const v = Alpine.$data(document.querySelector({view_sel!r})); "
        f"const s = v.allSessions.find(x => x.id === {session_id!r}); "
        f"return v.sessionLabel(s); }})()"
    )


def test_titleless_session_falls_back_to_date_not_username(authenticated_page, e2e_session_store):
    """Untitled background session should show its date, not the bare username."""
    page = authenticated_page

    s = Session(
        id="bg-no-title",
        agent="test-agent",
        source=SessionSource.BACKGROUND.value,
        status=SessionStatus.COMPLETED.value,
        user_id="justyn",
        prompt="",
        title=None,
        created_at="2026-04-07T15:44:00+00:00",
        last_active="2026-04-07T15:44:00+00:00",
    )
    e2e_session_store.create_session(s)

    open_conversations(page)
    reload_conversations_view(page)
    wait_for_session_in_list(page, s.id)
    page.wait_for_selector(".console-session", timeout=5000)
    page.screenshot(path="/tmp/tsugite-issue-state.png", full_page=True)

    label = _title_for(page, s.id)
    assert label != "justyn", f"label fell back to bare username: {label!r}"
    # Expect a month-name date fragment ("Apr") since created_at is in April.
    assert "Apr" in label, f"expected date-style label, got {label!r}"


def test_titleless_interactive_session_falls_back_to_date(authenticated_page, e2e_session_store):
    """My own untitled interactive session shouldn't show the bare username either."""
    page = authenticated_page
    user_id = page.evaluate("Alpine.store('app').userId")

    s = Session(
        id="int-no-title",
        agent="test-agent",
        source=SessionSource.INTERACTIVE.value,
        status=SessionStatus.ACTIVE.value,
        user_id=user_id,
        prompt="",
        title=None,
        created_at="2026-04-07T15:44:00+00:00",
        last_active="2026-04-07T15:44:00+00:00",
    )
    e2e_session_store.create_session(s)

    open_conversations(page)
    reload_conversations_view(page)
    wait_for_session_in_list(page, s.id)
    page.wait_for_selector(".console-session", timeout=5000)

    label = _title_for(page, s.id)
    assert label != user_id, f"interactive label fell back to bare username: {label!r}"
    assert "Apr" in label, f"expected date-style label, got {label!r}"


def test_titleless_session_with_prompt_prefers_prompt_when_date_missing(
    authenticated_page, e2e_session_store
):
    """When a prompt exists but date doesn't, use truncated prompt over username."""
    page = authenticated_page

    long_prompt = "Investigate the daemon's reconnection loop after PWA cold restart"
    s = Session(
        id="bg-prompt-only",
        agent="test-agent",
        source=SessionSource.BACKGROUND.value,
        status=SessionStatus.COMPLETED.value,
        user_id="justyn",
        prompt=long_prompt,
        title=None,
    )
    e2e_session_store.create_session(s)
    # Clear the created_at the dataclass auto-fills so we can exercise the
    # date-missing path: we need to test that prompt wins over username.
    with e2e_session_store._lock:
        e2e_session_store._sessions[s.id].created_at = ""
        e2e_session_store._sessions[s.id].last_active = ""

    open_conversations(page)
    reload_conversations_view(page)
    wait_for_session_in_list(page, s.id)
    page.wait_for_selector(".console-session", timeout=5000)

    label = _title_for(page, s.id)
    assert label != "justyn", f"label fell back to bare username: {label!r}"
    # First word of the prompt should appear in the label.
    assert "Investigate" in label, f"expected prompt-based label, got {label!r}"
