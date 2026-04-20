"""E2E tests for the smart scroll-to-bottom behavior.

The `scrollMessages()` helper now respects an `isAtBottom` flag so streaming
events no longer yank the user down when they have scrolled up to read
history. A floating `.scroll-fab` button appears while `isAtBottom` is false.
"""

from unittest.mock import patch

from tsugite.history.storage import SessionStorage


def _seed_long_session(e2e_adapter, e2e_tmp, label, turns):
    unique_user = f"scroll-user-{label}"
    session = e2e_adapter.session_store.get_or_create_interactive(unique_user, "test-agent")
    history_dir = e2e_tmp / f"history-scroll-{label}"
    history_dir.mkdir(exist_ok=True)
    session_path = history_dir / f"{session.id}.jsonl"
    storage = SessionStorage.create("test-agent", model="test", session_path=session_path)
    for i in range(turns):
        storage.record_turn(
            messages=[{"role": "user", "content": f"ask {i}"}],
            final_answer=f"answer {i}\n\n" + ("filler line\n" * 20),
        )
    return history_dir, unique_user, session.id


def _open_session(page, user_id, session_id):
    page.evaluate(f"localStorage.setItem('tsugite_user_id', {user_id!r})")
    page.goto(page.url.split("#")[0] + f"#conversations?session={session_id}")
    page.reload()
    page.wait_for_function("!Alpine.store('app').authRequired", timeout=5000)
    page.wait_for_function(f"Alpine.store('app').userId === {user_id!r}", timeout=3000)
    page.wait_for_selector(".msg.agent", timeout=5000)


def test_scroll_fab_hidden_when_at_bottom(authenticated_page, e2e_adapter, e2e_tmp):
    page = authenticated_page
    history_dir, user_id, session_id = _seed_long_session(e2e_adapter, e2e_tmp, "atbottom", turns=8)

    with patch("tsugite.daemon.adapters.http.get_history_dir", return_value=history_dir):
        _open_session(page, user_id, session_id)

        # History load forces scroll-to-bottom; FAB must be hidden.
        page.wait_for_function(
            "(() => { const b = document.querySelector('.scroll-fab'); "
            "return !b || getComputedStyle(b).display === 'none'; })()",
            timeout=2000,
        )


def test_scroll_fab_appears_when_scrolled_up_and_returns_on_click(authenticated_page, e2e_adapter, e2e_tmp):
    page = authenticated_page
    history_dir, user_id, session_id = _seed_long_session(e2e_adapter, e2e_tmp, "scrollup", turns=12)

    with patch("tsugite.daemon.adapters.http.get_history_dir", return_value=history_dir):
        _open_session(page, user_id, session_id)

        page.evaluate("document.getElementById('messages').scrollTop = 0")

        fab = page.locator(".scroll-fab")
        page.wait_for_function(
            "(() => { const b = document.querySelector('.scroll-fab'); "
            "return b && getComputedStyle(b).display !== 'none'; })()",
            timeout=2000,
        )
        assert fab.is_visible()

        fab.click()

        page.wait_for_function(
            "(() => { const el = document.getElementById('messages'); "
            "return el.scrollHeight - el.scrollTop - el.clientHeight < 40; })()",
            timeout=2000,
        )


def test_streaming_does_not_yank_user_when_scrolled_up(authenticated_page, e2e_adapter, e2e_tmp):
    """If the user is reading earlier output, newly-pushed messages must not auto-scroll them down."""
    page = authenticated_page
    history_dir, user_id, session_id = _seed_long_session(e2e_adapter, e2e_tmp, "nopin", turns=10)

    with patch("tsugite.daemon.adapters.http.get_history_dir", return_value=history_dir):
        _open_session(page, user_id, session_id)

        page.evaluate(
            """
            const el = document.getElementById('messages');
            el.scrollTop = 0;
            // Fire the scroll listener explicitly so isAtBottom updates synchronously
            // (passive scroll events can otherwise be coalesced in headless).
            el.dispatchEvent(new Event('scroll'));
            """
        )

        # isAtBottom is false now; pushing a message via the conversations view
        # should hit the guarded path in scrollMessages() and leave scrollTop alone.
        before = page.evaluate("document.getElementById('messages').scrollTop")
        assert before == 0

        page.evaluate(
            """
            const root = document.querySelector('[x-data*=conversations]') || document.querySelector('#app');
            const view = Alpine.$data(document.getElementById('messages'));
            view.messages.push({ type: 'agent', text: 'late arrival from stream' });
            view.scrollMessages();
            """
        )

        # Give $nextTick + microtasks a moment.
        page.wait_for_timeout(100)
        after = page.evaluate("document.getElementById('messages').scrollTop")
        assert after == before, "scroll position changed while user was reading history"


def test_streaming_follows_when_user_is_at_bottom(authenticated_page, e2e_adapter, e2e_tmp):
    page = authenticated_page
    history_dir, user_id, session_id = _seed_long_session(e2e_adapter, e2e_tmp, "pinfollow", turns=10)

    with patch("tsugite.daemon.adapters.http.get_history_dir", return_value=history_dir):
        _open_session(page, user_id, session_id)

        # Pinned to bottom after the history open.
        page.wait_for_function(
            "(() => { const el = document.getElementById('messages'); "
            "return el.scrollHeight - el.scrollTop - el.clientHeight < 40; })()",
            timeout=2000,
        )

        page.evaluate(
            """
            const view = Alpine.$data(document.getElementById('messages'));
            view.messages.push({ type: 'agent', text: 'tail one' });
            view.messages.push({ type: 'agent', text: 'tail two' });
            view.scrollMessages();
            """
        )

        page.wait_for_function(
            "(() => { const el = document.getElementById('messages'); "
            "return el.scrollHeight - el.scrollTop - el.clientHeight < 40; })()",
            timeout=2000,
        )
