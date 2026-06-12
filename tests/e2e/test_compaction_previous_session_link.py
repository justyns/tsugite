"""A session that was compacted from a predecessor must expose an always-visible
"previous session" link in the conversation header (driven by the compaction
event's `source_session_id`), and clicking it must navigate to that predecessor.

Before this, the only back-link was an 11px button buried in the compaction
separator at the top of history — easy to miss and often scrolled off, so users
couldn't find their way back to the pre-compaction session.
"""

from unittest.mock import patch

from tsugite.daemon.session_store import Session, SessionSource
from tsugite.history.storage import SessionStorage, generate_session_id

from .helpers import CONV_VIEW, open_conversations, reload_conversations_view, select_session_in_view


def _make_session(store, user_id):
    sid = generate_session_id("test-agent")
    s = Session(id=sid, agent="test-agent", source=SessionSource.INTERACTIVE.value, user_id=user_id)
    store.create_session(s)
    return s


def _seed_new_with_compaction(e2e_tmp, new_session_id, source_id):
    history_dir = e2e_tmp / "history"
    history_dir.mkdir(exist_ok=True)
    path = history_dir / f"{new_session_id}.jsonl"
    if path.exists():
        path.unlink()
    storage = SessionStorage.create("test-agent", model="test", session_path=path)
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


def test_previous_session_header_link_renders_and_navigates(authenticated_page, e2e_session_store, e2e_tmp):
    page = authenticated_page
    open_conversations(page)
    user_id = page.evaluate("Alpine.store('app').userId")

    old = _make_session(e2e_session_store, user_id)
    new = _make_session(e2e_session_store, user_id)
    new.parent_session = old.id
    e2e_session_store._sessions[old.id].superseded_by = new.id

    history_dir = _seed_new_with_compaction(e2e_tmp, new.id, old.id)

    with patch("tsugite.daemon.adapters.http.get_history_dir", return_value=history_dir):
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
