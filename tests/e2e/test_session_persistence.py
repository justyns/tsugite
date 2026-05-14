"""Persisted-selection tests for the conversations view.

Mobile PWA standalone reopens at the manifest `start_url: "/"` after iOS
evicts it from memory, dropping the URL hash that selectSession writes. With
no localStorage persistence of `selectedSessionId`, `reload()` falls through
to `autoSelectInteractive()` and lands the user on whatever happens to be
"most recently active" - rarely the conversation they had open.
"""

from tsugite.daemon.session_store import Session, SessionSource, SessionStatus
from tsugite.history.storage import generate_session_id

from .helpers import (
    CONV_VIEW,
    open_conversations,
    reload_conversations_view,
    select_session_in_view,
    wait_for_alpine_ready,
    wait_for_session_in_list,
)


def _make_session(store, user_id, last_active):
    sid = generate_session_id("test-agent")
    s = Session(
        id=sid,
        agent="test-agent",
        source=SessionSource.INTERACTIVE.value,
        user_id=user_id,
        last_active=last_active,
    )
    store.create_session(s)
    return s


def test_selected_session_restored_after_pwa_cold_restart(
    authenticated_page, e2e_session_store, base_url
):
    """A user-selected session survives a PWA cold restart at start_url="/".

    Repro: select a session that is NOT the auto-pick, then navigate to bare
    base_url with no hash (simulating iOS evicting the PWA from memory and
    reopening at the manifest's start_url). Without persistence the page
    auto-picks the most-recently-active session instead of restoring.
    """
    page = authenticated_page
    user_id = page.evaluate("Alpine.store('app').userId")

    # auto_pick is most recently active so autoSelectInteractive() lands here by default.
    auto_pick = _make_session(e2e_session_store, user_id, last_active="2026-05-14T12:00:00+00:00")
    user_pick = _make_session(e2e_session_store, user_id, last_active="2026-05-14T08:00:00+00:00")

    open_conversations(page)
    reload_conversations_view(page)
    wait_for_session_in_list(page, user_pick.id)

    select_session_in_view(page, user_pick.id)

    # Cold restart: navigate to start_url with no hash. The PWA manifest pins
    # this to "/" so iOS reopens here when the app gets evicted.
    page.goto(base_url + "/")
    wait_for_alpine_ready(page)
    open_conversations(page)
    page.wait_for_selector(CONV_VIEW, timeout=5000)
    page.wait_for_function(
        f"!!Alpine.$data(document.querySelector({CONV_VIEW!r}))?.selectedSessionId",
        timeout=5000,
    )

    page.screenshot(path="/tmp/tsugite-issue-state.png", full_page=True)

    selected = page.evaluate(
        f"Alpine.$data(document.querySelector({CONV_VIEW!r})).selectedSessionId"
    )
    assert selected == user_pick.id, (
        f"expected user-selected session {user_pick.id} to be restored after cold restart, "
        f"got {selected} (likely auto_pick={auto_pick.id})"
    )


def test_persisted_session_id_for_missing_session_falls_through_to_auto_select(
    authenticated_page, e2e_session_store, base_url
):
    """Stale persisted ID for a session that no longer exists shouldn't block auto-select."""
    page = authenticated_page
    user_id = page.evaluate("Alpine.store('app').userId")

    auto_pick = _make_session(e2e_session_store, user_id, last_active="2026-05-14T12:00:00+00:00")

    page.evaluate("localStorage.setItem('tsugite_selected_session_test-agent', 'sess-does-not-exist')")

    page.goto(base_url + "/")
    wait_for_alpine_ready(page)
    open_conversations(page)
    page.wait_for_function(
        f"Alpine.$data(document.querySelector({CONV_VIEW!r}))?.selectedSessionId === {auto_pick.id!r}",
        timeout=5000,
    )


def test_persisted_session_id_for_superseded_session_chases_to_successor(
    authenticated_page, e2e_session_store, base_url
):
    """Persisted ID points at a now-compacted session; restoration follows superseded_by."""
    page = authenticated_page
    user_id = page.evaluate("Alpine.store('app').userId")

    successor = _make_session(e2e_session_store, user_id, last_active="2026-05-14T12:00:00+00:00")
    predecessor_id = generate_session_id("test-agent")
    predecessor = Session(
        id=predecessor_id,
        agent="test-agent",
        source=SessionSource.INTERACTIVE.value,
        user_id=user_id,
        last_active="2026-05-14T08:00:00+00:00",
        status=SessionStatus.COMPLETED.value,
        superseded_by=successor.id,
    )
    e2e_session_store.create_session(predecessor)

    page.evaluate(
        f"localStorage.setItem('tsugite_selected_session_test-agent', {predecessor_id!r})"
    )

    page.goto(base_url + "/")
    wait_for_alpine_ready(page)
    open_conversations(page)
    page.wait_for_function(
        f"Alpine.$data(document.querySelector({CONV_VIEW!r}))?.selectedSessionId === {successor.id!r}",
        timeout=5000,
    )
