"""Regression: agent's final-reply bubble must not render twice when the per-chat
streaming push races the history_update reload."""

from unittest.mock import patch

from tsugite.history.storage import SessionStorage

from .helpers import CONV_VIEW


def _seed_completed_turn(history_dir, session_id, agent_name, user_msg, response_text):
    session_path = history_dir / f"{session_id}.jsonl"
    storage = SessionStorage.create(agent_name, model="test", session_path=session_path)
    storage.record("user_input", text=user_msg)
    storage.record("model_response", provider="test", model="test", raw_content=response_text)


def _setup_session(page, sid, *, sending, user_msg=None):
    page.evaluate(
        f"""({{sid, userMsg, sending}}) => {{
            const v = Alpine.$data(document.querySelector({CONV_VIEW!r}));
            v.selectedSessionId = sid;
            const s = v._sessionState(sid);
            s.messages.length = 0;
            s.sending = !!sending;
            if (userMsg) s.messages.push({{type: 'user', text: userMsg}});
        }}""",
        {"sid": sid, "userMsg": user_msg, "sending": sending},
    )


def _fire_history_update(page, agent="test-agent"):
    page.evaluate(
        """(agent) => {
            Alpine.store('app').lastEvent = {
                type: 'history_update',
                data: {agent},
                _ts: Date.now(),
            };
        }""",
        agent,
    )


def _snapshot_messages(page, sid):
    return page.evaluate(
        f"""(sid) => {{
            const v = Alpine.$data(document.querySelector({CONV_VIEW!r}));
            return (v.sessionsState[sid]?.messages || []).map(m => ({{type: m.type, text: (m.text || '').slice(0, 60)}}));
        }}""",
        sid,
    )


def _count_agent_bubbles(snapshot, text):
    return sum(1 for m in snapshot if m["type"] == "agent" and text in m["text"])


# Debounce in _debouncedLoadHistory is 200ms; 350ms covers debounce + fetch + rebuild.
_RELOAD_WAIT_MS = 350


def test_history_update_during_send_does_not_duplicate_streamed_bubble(chat_page, e2e_adapter, e2e_tmp):
    page = chat_page
    user_id = page.evaluate("Alpine.store('app').userId")
    session = e2e_adapter.session_store.get_or_create_interactive(user_id, "test-agent")

    history_dir = e2e_tmp / "history"
    history_dir.mkdir(exist_ok=True)
    response_text = "Final reply text"
    _seed_completed_turn(history_dir, session.id, "test-agent", "the prompt", response_text)

    with patch("tsugite_daemon.adapters.http.get_history_dir", return_value=history_dir):
        _setup_session(page, session.id, sending=True, user_msg="the prompt")
        _fire_history_update(page)
        page.wait_for_timeout(_RELOAD_WAIT_MS)

        # Stream's final_result lands after the reload would have rebuilt the array.
        page.evaluate(
            f"""({{sid, response}}) => {{
                const v = Alpine.$data(document.querySelector({CONV_VIEW!r}));
                v._sessionState(sid).messages.push({{type: 'agent', text: response}});
            }}""",
            {"sid": session.id, "response": response_text},
        )

        snapshot = _snapshot_messages(page, session.id)

    assert _count_agent_bubbles(snapshot, response_text) == 1, snapshot


def test_history_update_when_idle_still_reloads(chat_page, e2e_adapter, e2e_tmp):
    page = chat_page
    user_id = page.evaluate("Alpine.store('app').userId")
    session = e2e_adapter.session_store.get_or_create_interactive(user_id, "test-agent")

    history_dir = e2e_tmp / "history"
    history_dir.mkdir(exist_ok=True)
    response_text = "Adapter-driven reply"
    _seed_completed_turn(history_dir, session.id, "test-agent", "a prompt", response_text)

    with patch("tsugite_daemon.adapters.http.get_history_dir", return_value=history_dir):
        _setup_session(page, session.id, sending=False)
        _fire_history_update(page)
        page.wait_for_timeout(_RELOAD_WAIT_MS)
        snapshot = _snapshot_messages(page, session.id)

    assert _count_agent_bubbles(snapshot, response_text) == 1, snapshot


def test_history_update_after_send_completes_reloads(chat_page, e2e_adapter, e2e_tmp):
    page = chat_page
    user_id = page.evaluate("Alpine.store('app').userId")
    session = e2e_adapter.session_store.get_or_create_interactive(user_id, "test-agent")

    history_dir = e2e_tmp / "history"
    history_dir.mkdir(exist_ok=True)
    response_text = "Reconciled reply"
    _seed_completed_turn(history_dir, session.id, "test-agent", "a prompt", response_text)

    with patch("tsugite_daemon.adapters.http.get_history_dir", return_value=history_dir):
        _setup_session(page, session.id, sending=True)
        _fire_history_update(page)
        page.wait_for_timeout(_RELOAD_WAIT_MS)

        page.evaluate(
            f"""(sid) => {{
                const v = Alpine.$data(document.querySelector({CONV_VIEW!r}));
                v._sessionState(sid).sending = false;
            }}""",
            session.id,
        )
        _fire_history_update(page)
        page.wait_for_timeout(_RELOAD_WAIT_MS)
        snapshot = _snapshot_messages(page, session.id)

    assert _count_agent_bubbles(snapshot, response_text) == 1, snapshot
