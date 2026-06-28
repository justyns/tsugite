"""Persisted `info` events from `send_message()` must render as info bubbles on history replay."""

from unittest.mock import patch

from tsugite.history.storage import SessionStorage


def _seed_history_with_info(e2e_adapter, e2e_tmp, user_id):
    session = e2e_adapter.session_store.get_or_create_interactive(user_id, "test-agent")
    history_dir = e2e_tmp / "history"
    history_dir.mkdir(exist_ok=True)
    session_path = history_dir / f"{session.id}.jsonl"
    if session_path.exists():
        session_path.unlink()

    storage = SessionStorage.create("test-agent", model="test", session_path=session_path)
    storage.record("user_input", text="Hi")
    storage.record("info", message="test info bubble")
    storage.record("model_response", provider="test", model="test", raw_content="All done.")
    return history_dir, session


def test_info_event_renders_as_bubble_on_history_replay(authenticated_page, e2e_adapter, e2e_tmp):
    page = authenticated_page
    page.locator(".console-tabs button.console-tab", has_text="Conversations").click()
    page.wait_for_function("Alpine.store('app').view === 'conversations'", timeout=3000)

    user_id = page.evaluate("Alpine.store('app').userId")
    history_dir, _ = _seed_history_with_info(e2e_adapter, e2e_tmp, user_id)

    with patch("tsugite.history.storage.get_history_dir", return_value=history_dir):
        page.reload()
        page.wait_for_selector(".console-turn.user", timeout=5000)
        page.wait_for_selector(".console-turn.agent", timeout=5000)
        page.screenshot(path="/tmp/tsugite-issue-state.png", full_page=True)

        info_bubble = page.locator(".console-turn.info .console-turn-bubble")
        assert info_bubble.count() > 0, (
            "info events in the persisted JSONL must render as .console-turn.info bubbles "
            "via eventsToBubbles, but the replay produced no info element"
        )
        assert "test info bubble" in info_bubble.first.text_content()
