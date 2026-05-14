"""History loading and reaction persistence tests."""

import json
from unittest.mock import patch

from tsugite.history.storage import SessionStorage


def _seed_history(e2e_adapter, e2e_tmp, user_id, turns, reactions=None):
    """Seed user_input + model_response events as a session JSONL.

    Args:
        turns: list of (user_msg, assistant_msg) tuples — each becomes a
            user_input event followed by a model_response event.
        reactions: list of emoji strings, attached to the latest user_input
            via the reaction event type.
    Returns:
        (history_dir, session)
    """
    session = e2e_adapter.session_store.get_or_create_interactive(user_id, "test-agent")
    history_dir = e2e_tmp / "history"
    history_dir.mkdir(exist_ok=True)

    session_path = history_dir / f"{session.id}.jsonl"
    if session_path.exists():
        session_path.unlink()
    storage = SessionStorage.create("test-agent", model="test", session_path=session_path)

    for user_msg, assistant_msg in turns:
        storage.record("user_input", text=user_msg)
        storage.record(
            "model_response",
            provider="test",
            model="test",
            raw_content=assistant_msg,
        )

    for emoji in reactions or []:
        storage.record("reaction", emoji=emoji)

    return history_dir, session


def test_history_reactions_persist_across_reload(authenticated_page, mock_chat, e2e_adapter, e2e_tmp):
    """Reactions seeded into the JSONL render after a fresh page load."""
    page = authenticated_page
    page.locator(".console-tabs button.console-tab", has_text="Conversations").click()
    page.wait_for_function("Alpine.store('app').view === 'conversations'", timeout=3000)

    user_id = page.evaluate("Alpine.store('app').userId")
    history_dir, _ = _seed_history(
        e2e_adapter,
        e2e_tmp,
        user_id,
        turns=[("seeded message", "seeded response")],
        reactions=["✅"],
    )

    with patch("tsugite.daemon.adapters.http.get_history_dir", return_value=history_dir):
        page.reload()
        page.wait_for_function("!Alpine.store('app').authRequired", timeout=5000)
        page.locator(".console-tabs button.console-tab", has_text="Conversations").click()
        page.wait_for_function("Alpine.store('app').view === 'conversations'", timeout=3000)
        page.wait_for_selector(".console-turn.user", timeout=5000)

        user_bubble = page.locator(".console-turn.user .console-turn-bubble").first
        assert "✅" in user_bubble.text_content()


def test_history_pagination_load_more(authenticated_page, e2e_adapter, e2e_tmp):
    """With many turns, only recent messages show initially; 'load more' fetches earlier ones."""
    page = authenticated_page
    user_id = page.evaluate("Alpine.store('app').userId")

    turns = [(f"user msg {i}", f"assistant msg {i}") for i in range(30)]
    history_dir, session = _seed_history(e2e_adapter, e2e_tmp, user_id, turns=turns)

    with patch("tsugite.daemon.adapters.http.get_history_dir", return_value=history_dir):
        page.locator(".console-tabs button.console-tab", has_text="Conversations").click()
        page.wait_for_function("Alpine.store('app').view === 'conversations'", timeout=3000)
        page.evaluate(f"Alpine.store('app').viewSessionId = {session.id!r}")
        page.evaluate("Alpine.$data(document.querySelector('[x-data*=conversationsView]')).reload()")
        page.wait_for_function(
            f"Alpine.$data(document.querySelector('[x-data*=conversationsView]')).selectedSessionId === {session.id!r}",
            timeout=5000,
        )
        page.wait_for_selector(".console-turn.user", timeout=5000)
        # Pagination separator is only rendered when total bubbles > HISTORY_PAGE_SIZE.
        page.wait_for_function(
            "document.querySelectorAll('.console-history-sep button').length > 0",
            timeout=3000,
        )

        load_more = page.locator(".console-history-sep button")
        assert "earlier" in load_more.first.text_content().lower()

        initial = page.locator(".console-turn.user").count()
        load_more.first.click()
        page.wait_for_function(
            f"document.querySelectorAll('.console-turn.user').length > {initial}",
            timeout=5000,
        )


def test_compaction_banner_displayed(authenticated_page, e2e_adapter, e2e_tmp):
    """A compaction event in the JSONL surfaces as a compaction banner."""
    page = authenticated_page
    user_id = page.evaluate("Alpine.store('app').userId")

    history_dir, session = _seed_history(e2e_adapter, e2e_tmp, user_id, turns=[("after compaction", "response")])

    actual_path = history_dir / f"{session.id}.jsonl"
    with open(actual_path, "r") as f:
        lines = f.readlines()
    compaction_event = json.dumps(
        {
            "type": "compaction",
            "ts": "2025-01-01T00:00:00+00:00",
            "data": {
                "summary": "Previously discussed project architecture and deployment strategy.",
                "replaced_count": 5,
                "retained_count": 0,
                "reason": "token_threshold",
            },
        }
    )
    lines.insert(1, compaction_event + "\n")
    with open(actual_path, "w") as f:
        f.writelines(lines)

    with patch("tsugite.daemon.adapters.http.get_history_dir", return_value=history_dir):
        page.locator(".console-tabs button.console-tab", has_text="Conversations").click()
        page.wait_for_function("Alpine.store('app').view === 'conversations'", timeout=3000)
        page.evaluate("Alpine.$data(document.querySelector('[x-data*=conversationsView]')).reload()")
        page.wait_for_selector(".console-turn.user", timeout=5000)

        banner = page.locator(".console-compaction-banner")
        assert banner.count() > 0
        text = banner.first.text_content().lower()
        assert "summary" in text or "architecture" in text
