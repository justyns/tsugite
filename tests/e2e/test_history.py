"""History loading and reaction persistence tests."""

from unittest.mock import patch


def test_history_reactions_persist_across_reload(authenticated_page, mock_chat, e2e_adapter, e2e_tmp):
    """Reactions should survive a page reload via session event log."""
    from tsugite.history.storage import SessionStorage

    page = authenticated_page
    page.locator("nav button", has_text="Conversations").click()
    page.wait_for_function("Alpine.store('app').view === 'conversations'", timeout=3000)

    # Get the user ID and session the UI is using
    user_id = page.evaluate("Alpine.store('app').userId")
    session = e2e_adapter.session_store.get_or_create_interactive(user_id, "test-agent")

    # Seed a turn in history storage
    history_dir = e2e_tmp / "history"
    history_dir.mkdir(exist_ok=True)
    session_path = history_dir / f"{session.id}.jsonl"

    storage = SessionStorage.create("test-agent", model="test", session_path=session_path)
    storage.record_turn(
        messages=[
            {"role": "user", "content": "seeded message"},
            {"role": "assistant", "content": "seeded response"},
        ],
        final_answer="seeded response",
    )
    # Rename to match the session ID the adapter resolves
    actual_path = history_dir / f"{session.id}.jsonl"
    if session_path != actual_path and session_path.exists():
        session_path.rename(actual_path)

    # Seed a reaction event in the session event log
    from tsugite.history.models import Turn

    records = SessionStorage.load(actual_path).load_records()
    turn = next(r for r in records if isinstance(r, Turn))
    reaction_ts = turn.timestamp.isoformat().replace("+00:00", "") + ".100000+00:00"

    e2e_adapter.session_store.append_event(session.id, {
        "type": "reaction",
        "emoji": "✅",
        "message_id": None,
        "timestamp": reaction_ts,
    })

    # Reload the page — reactions should come back from history
    with patch("tsugite.daemon.adapters.http.get_history_dir", return_value=history_dir):
        page.reload()
        page.wait_for_function("!Alpine.store('app').authRequired", timeout=5000)
        page.locator("nav button", has_text="Conversations").click()
        page.wait_for_function("Alpine.store('app').view === 'conversations'", timeout=3000)

        page.wait_for_selector(".msg.user", timeout=5000)
        assert page.locator(".reaction-badge").count() > 0
        assert "✅" in page.locator(".reaction-badge").first.text_content()
