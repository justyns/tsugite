"""Reconnect catch-up must be coalesced and non-destructive.

A mobile PWA wake produces a burst of SSE `reconnect` events (flaky radio:
connect -> fail -> backoff -> reconnect). Each one used to run the destructive
`reload()` (resetHistory + sessionsState wipe + reselect), so the thread
blanked and repainted once per event — the reported wake flashing.
"""

from unittest.mock import patch

from tsugite_daemon.session_store import Session, SessionSource

from tsugite.history.storage import SessionStorage, generate_session_id

from .helpers import (
    CONV_VIEW,
    open_conversations,
    reload_conversations_view,
    select_session_in_view,
    wait_for_session_in_list,
)


def _make_and_seed(e2e_session_store, e2e_tmp, user_id, turns=2):
    sid = generate_session_id("test-agent")
    e2e_session_store.create_session(
        Session(id=sid, agent="test-agent", source=SessionSource.INTERACTIVE.value, user_id=user_id)
    )
    history_dir = e2e_tmp / "history"
    history_dir.mkdir(exist_ok=True)
    path = history_dir / f"{sid}.jsonl"
    if path.exists():
        path.unlink()
    storage = SessionStorage.create("test-agent", model="test", session_path=path)
    for i in range(turns):
        storage.record("user_input", text=f"user message {i}")
        storage.record("model_response", provider="test", model="test", raw_content=f"assistant reply {i}")
    storage.record("session_end", status="success")
    return sid, storage


def _fire_reconnect(page, times=3, spacing_ms=60):
    page.evaluate(
        f"""(async () => {{
            for (let i = 0; i < {times}; i++) {{
                Alpine.store('app').lastEvent = {{ type: 'reconnect', _ts: Date.now() + i }};
                await new Promise(r => setTimeout(r, {spacing_ms}));
            }}
        }})()"""
    )


def test_reconnect_burst_coalesces_and_preserves_state(authenticated_page, e2e_session_store, e2e_tmp):
    page = authenticated_page
    open_conversations(page)
    user_id = page.evaluate("Alpine.store('app').userId")
    sid, _ = _make_and_seed(e2e_session_store, e2e_tmp, user_id)
    history_dir = e2e_tmp / "history"

    with patch("tsugite.history.storage.get_history_dir", return_value=history_dir):
        reload_conversations_view(page)
        wait_for_session_in_list(page, sid)
        select_session_in_view(page, sid)
        page.wait_for_selector(".console-turn.user", timeout=5000)

        # Marker in the per-session cache: reload()'s `sessionsState = {}` wipes
        # it, a non-destructive reconcile keeps it.
        page.evaluate(f"Alpine.$data(document.querySelector({CONV_VIEW!r})).sessionsState[{sid!r}]._testMarker = 42")

        session_fetches = []
        page.on(
            "request",
            lambda req: session_fetches.append(req.url) if "/sessions?" in req.url else None,
        )

        _fire_reconnect(page, times=3)
        page.screenshot(path="/tmp/tsugite-issue-state.png", full_page=True)
        page.wait_for_timeout(1200)

        marker = page.evaluate(
            f"Alpine.$data(document.querySelector({CONV_VIEW!r})).sessionsState[{sid!r}]?._testMarker"
        )
        selected = page.evaluate(f"Alpine.$data(document.querySelector({CONV_VIEW!r})).selectedSessionId")

    assert selected == sid, f"selection changed during reconnect burst: {selected}"
    assert marker == 42, "reconnect burst wiped sessionsState (destructive reload ran)"
    assert len(session_fetches) <= 1, f"reconnect burst was not coalesced: {len(session_fetches)} session-list fetches"


def test_reconnect_still_reconciles_missed_messages(authenticated_page, e2e_session_store, e2e_tmp):
    """Correctness preserved: a message that arrived while disconnected shows up after reconnect."""
    page = authenticated_page
    open_conversations(page)
    user_id = page.evaluate("Alpine.store('app').userId")
    sid, storage = _make_and_seed(e2e_session_store, e2e_tmp, user_id)
    history_dir = e2e_tmp / "history"

    with patch("tsugite.history.storage.get_history_dir", return_value=history_dir):
        reload_conversations_view(page)
        wait_for_session_in_list(page, sid)
        select_session_in_view(page, sid)
        page.wait_for_selector(".console-turn.user", timeout=5000)

        storage.record("user_input", text="missed while asleep")
        storage.record("model_response", provider="test", model="test", raw_content="missed reply landed")

        _fire_reconnect(page, times=1)
        page.wait_for_function(
            "document.body.innerText.includes('missed reply landed')",
            timeout=5000,
        )
