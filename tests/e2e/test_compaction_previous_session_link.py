"""A session that was compacted from a predecessor must expose an always-visible
"previous session" link in the conversation header (driven by the compaction
event's `source_session_id`), and clicking it must navigate to that predecessor.

Before this, the only back-link was an 11px button buried in the compaction
separator at the top of history — easy to miss and often scrolled off, so users
couldn't find their way back to the pre-compaction session.
"""

from unittest.mock import patch

from tsugite_daemon.session_store import Session, SessionSource

from tsugite.history.storage import SessionStorage, generate_session_id

from .helpers import CONV_VIEW, open_conversations, reload_conversations_view, select_session_in_view


def _make_session(store, user_id):
    sid = generate_session_id("test-agent")
    s = Session(id=sid, agent="test-agent", source=SessionSource.INTERACTIVE.value, user_id=user_id)
    store.create_session(s)
    return s


def _fresh_session_storage(e2e_tmp, new_session_id):
    """Create a clean JSONL storage for a session, returning (history_dir, storage)."""
    history_dir = e2e_tmp / "history"
    history_dir.mkdir(exist_ok=True)
    path = history_dir / f"{new_session_id}.jsonl"
    if path.exists():
        path.unlink()
    return history_dir, SessionStorage.create("test-agent", model="test", session_path=path)


def _seed_new_with_compaction(e2e_tmp, new_session_id, source_id):
    history_dir, storage = _fresh_session_storage(e2e_tmp, new_session_id)
    storage.record(
        "compaction",
        summary="prior work on widgets",
        reason="scheduled",
        source_session_id=source_id,
        replaced_count=2,
        retained_count=1,
    )
    storage.record("user_input", text="continue please")
    storage.record("model_response", provider="test", model="test", raw_content="sure, continuing")
    return history_dir


def _seed_new_with_two_compactions(e2e_tmp, new_session_id, source_id):
    """Two compaction events in one session: an early one (distinct summary) and
    a later one whose summary drives the top-of-thread banner."""
    history_dir, storage = _fresh_session_storage(e2e_tmp, new_session_id)
    storage.record(
        "compaction",
        summary="early summary about widgets",
        reason="scheduled",
        source_session_id=source_id,
        replaced_count=2,
        retained_count=1,
    )
    storage.record("user_input", text="first follow-up")
    storage.record("model_response", provider="test", model="test", raw_content="ok one")
    storage.record(
        "compaction",
        summary="later summary about gadgets",
        reason="token_threshold",
        replaced_count=3,
        retained_count=1,
    )
    storage.record("user_input", text="second follow-up")
    storage.record("model_response", provider="test", model="test", raw_content="ok two")
    return history_dir


def test_previous_session_header_link_renders_and_navigates(authenticated_page, e2e_session_store, e2e_tmp):
    page = authenticated_page
    open_conversations(page)
    user_id = page.evaluate("Alpine.store('app').userId")

    old = _make_session(e2e_session_store, user_id)
    new = _make_session(e2e_session_store, user_id)
    new.parent_session = old.id
    e2e_session_store._sessions[old.id].superseded_by = new.id

    history_dir = _seed_new_with_compaction(e2e_tmp, new.id, old.id)

    with patch("tsugite_daemon.adapters.http.get_history_dir", return_value=history_dir):
        reload_conversations_view(page)
        select_session_in_view(page, new.id)

        # The header affordance is wired from the compaction event's source_session_id.
        page.wait_for_function(
            f"Alpine.$data(document.querySelector({CONV_VIEW!r})).compactionSourceId === {old.id!r}",
            timeout=5000,
        )

        # The link lives in the always-visible conversation header (not the
        # scrolled-off inline compaction separator).
        header_link = page.locator(".console-thread-header").get_by_role("button", name="previous session")
        header_link.wait_for(state="visible", timeout=3000)
        header_link.click()

        # Clicking it lands on the predecessor (follow:false — does not auto-forward back).
        page.wait_for_function(
            f"Alpine.$data(document.querySelector({CONV_VIEW!r})).selectedSessionId === {old.id!r}",
            timeout=3000,
        )


def test_compaction_summary_not_duplicated(authenticated_page, e2e_session_store, e2e_tmp):
    """The carried-forward summary renders exactly once for a successor session.

    The top-of-thread banner already shows the last compaction's summary, so the
    inline compaction event must not re-render the same summary as a nested
    `.console-compaction-banner` inside its <details>. Before the fix the summary
    appeared twice (top banner + nested inline block), which looked noisy and
    confusing for compacted sessions.
    """
    page = authenticated_page
    open_conversations(page)
    user_id = page.evaluate("Alpine.store('app').userId")

    old = _make_session(e2e_session_store, user_id)
    new = _make_session(e2e_session_store, user_id)
    new.parent_session = old.id
    e2e_session_store._sessions[old.id].superseded_by = new.id

    history_dir = _seed_new_with_compaction(e2e_tmp, new.id, old.id)

    with patch("tsugite_daemon.adapters.http.get_history_dir", return_value=history_dir):
        reload_conversations_view(page)
        select_session_in_view(page, new.id)

        # Top banner is wired from the last compaction's summary.
        page.wait_for_selector(".console-thread .console-compaction-banner", timeout=5000)
        page.screenshot(path="/tmp/tsugite-issue-state.png", full_page=True)

        # Exactly one carried-forward summary card inside the thread (top banner only).
        assert page.locator(".console-thread .console-compaction-banner").count() == 1

        # The inline compaction separator must not nest a duplicate summary banner.
        assert page.locator(".console-history-sep details .console-compaction-banner").count() == 0

        # Previous-session navigation is still available in the header.
        header_link = page.locator(".console-thread-header").get_by_role("button", name="previous session")
        header_link.wait_for(state="visible", timeout=3000)


def test_multi_compaction_only_top_summary_deduped(authenticated_page, e2e_session_store, e2e_tmp):
    """Dedup is surgical: in a multi-compaction chain only the inline event whose
    summary matches the top banner is collapsed; an earlier compaction with a
    *different* summary still expands its own inline summary."""
    page = authenticated_page
    open_conversations(page)
    user_id = page.evaluate("Alpine.store('app').userId")

    old = _make_session(e2e_session_store, user_id)
    new = _make_session(e2e_session_store, user_id)
    new.parent_session = old.id
    e2e_session_store._sessions[old.id].superseded_by = new.id

    history_dir = _seed_new_with_two_compactions(e2e_tmp, new.id, old.id)

    with patch("tsugite_daemon.adapters.http.get_history_dir", return_value=history_dir):
        reload_conversations_view(page)
        select_session_in_view(page, new.id)

        # The top banner is wired from the LAST compaction's summary.
        page.wait_for_function(
            f"Alpine.$data(document.querySelector({CONV_VIEW!r})).compactionSummary === 'later summary about gadgets'",
            timeout=5000,
        )

        # Exactly one inline compaction event still expands — the earlier, distinct one.
        page.wait_for_function(
            "document.querySelectorAll('.console-history-sep details').length === 1",
            timeout=3000,
        )
        nested = page.locator(".console-history-sep details .console-compaction-banner")
        assert nested.count() == 1
        assert "early summary about widgets" in nested.first.text_content()

        # Top banner + the single expanded inline event = two cards; the later
        # summary is not re-rendered inline.
        assert page.locator(".console-thread .console-compaction-banner").count() == 2
        assert "later summary about gadgets" not in page.locator(".console-history-sep details").first.text_content()
