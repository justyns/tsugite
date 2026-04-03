"""History loading and reaction persistence tests."""

from unittest.mock import patch

from tsugite.history.models import Turn
from tsugite.history.storage import SessionStorage


def _seed_history(e2e_adapter, e2e_tmp, user_id, turns, reactions=None):
    """Helper to seed turns and optional reactions into history + event log.

    Args:
        turns: list of (user_msg, assistant_msg) tuples
        reactions: list of emoji strings to attach to the last turn
    Returns:
        (history_dir, session)
    """
    session = e2e_adapter.session_store.get_or_create_interactive(user_id, "test-agent")
    history_dir = e2e_tmp / "history"
    history_dir.mkdir(exist_ok=True)

    session_path = history_dir / f"{session.id}.jsonl"
    storage = SessionStorage.create("test-agent", model="test", session_path=session_path)

    for user_msg, assistant_msg in turns:
        storage.record_turn(
            messages=[
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": assistant_msg},
            ],
            final_answer=assistant_msg,
        )

    # Rename to match session ID (create() generates its own ID)
    actual_path = history_dir / f"{session.id}.jsonl"
    if session_path != actual_path and session_path.exists():
        session_path.rename(actual_path)

    if reactions:
        records = SessionStorage.load(actual_path).load_records()
        last_turn = [r for r in records if isinstance(r, Turn)][-1]
        reaction_ts = last_turn.timestamp.isoformat().replace("+00:00", "") + ".100000+00:00"
        for emoji in reactions:
            e2e_adapter.session_store.append_event(
                session.id,
                {
                    "type": "reaction",
                    "emoji": emoji,
                    "message_id": None,
                    "timestamp": reaction_ts,
                },
            )

    return history_dir, session


def test_history_reactions_persist_across_reload(authenticated_page, mock_chat, e2e_adapter, e2e_tmp):
    """Reactions should survive a page reload via session event log."""
    page = authenticated_page
    page.locator("nav button", has_text="Conversations").click()
    page.wait_for_function("Alpine.store('app').view === 'conversations'", timeout=3000)

    user_id = page.evaluate("Alpine.store('app').userId")
    history_dir, _ = _seed_history(
        e2e_adapter, e2e_tmp, user_id, turns=[("seeded message", "seeded response")], reactions=["✅"]
    )

    with patch("tsugite.daemon.adapters.http.get_history_dir", return_value=history_dir):
        page.reload()
        page.wait_for_function("!Alpine.store('app').authRequired", timeout=5000)
        page.locator("nav button", has_text="Conversations").click()
        page.wait_for_function("Alpine.store('app').view === 'conversations'", timeout=3000)

        page.wait_for_selector(".msg.user", timeout=5000)
        assert page.locator(".reaction-badge").count() > 0
        assert "✅" in page.locator(".reaction-badge").first.text_content()


def test_history_pagination_load_more(authenticated_page, e2e_adapter, e2e_tmp):
    """With many turns, only recent messages show initially; 'load more' fetches earlier ones."""
    page = authenticated_page
    user_id = page.evaluate("Alpine.store('app').userId")

    # Seed 30 turns (HISTORY_PAGE_SIZE is 20, so this triggers pagination)
    turns = [(f"user msg {i}", f"assistant msg {i}") for i in range(30)]
    history_dir, _ = _seed_history(e2e_adapter, e2e_tmp, user_id, turns=turns)

    with patch("tsugite.daemon.adapters.http.get_history_dir", return_value=history_dir):
        page.locator("nav button", has_text="Conversations").click()
        page.wait_for_function("Alpine.store('app').view === 'conversations'", timeout=3000)
        page.wait_for_selector(".msg.user", timeout=5000)

        # Should see a "load more" separator at the top
        load_more = page.locator(".history-sep button")
        assert load_more.count() > 0
        assert "earlier messages" in load_more.first.text_content()

        initial_user_msgs = page.locator(".msg.user").count()

        # Click load more
        load_more.first.click()
        page.wait_for_timeout(500)

        # Should now have more user messages visible
        assert page.locator(".msg.user").count() > initial_user_msgs


def test_compaction_banner_displayed(authenticated_page, e2e_adapter, e2e_tmp):
    """When a session has a compaction summary, a banner should appear."""
    page = authenticated_page
    user_id = page.evaluate("Alpine.store('app').userId")

    history_dir, session = _seed_history(e2e_adapter, e2e_tmp, user_id, turns=[("after compaction", "response")])

    # Write a compaction summary record into the history file
    actual_path = history_dir / f"{session.id}.jsonl"
    import json

    with open(actual_path, "r") as f:
        lines = f.readlines()
    compaction_record = json.dumps(
        {
            "type": "compaction_summary",
            "summary": "Previously discussed project architecture and deployment strategy.",
            "previous_turns": 5,
            "retained_turns": 0,
            "compaction_reason": "token_threshold",
        }
    )
    lines.insert(1, compaction_record + "\n")
    with open(actual_path, "w") as f:
        f.writelines(lines)

    # Also set compacted_from on the session meta by rewriting the first line
    meta = json.loads(lines[0])
    meta["compacted_from"] = "old-session-id"
    lines[0] = json.dumps(meta) + "\n"
    with open(actual_path, "w") as f:
        f.writelines(lines)

    with patch("tsugite.daemon.adapters.http.get_history_dir", return_value=history_dir):
        page.locator("nav button", has_text="Conversations").click()
        page.wait_for_function("Alpine.store('app').view === 'conversations'", timeout=3000)
        page.wait_for_selector(".msg.user", timeout=5000)

        banner = page.locator(".compaction-banner")
        assert banner.count() > 0
        assert "summary" in banner.first.text_content().lower() or "architecture" in banner.first.text_content().lower()
