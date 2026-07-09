"""Client-side reconnect reconciliation: events missed while the stream was
down replay on reconnect and apply to the UI (no manual reload), and a daemon
restart (epoch change) triggers the full-resync path."""

from .helpers import CONV_VIEW, open_conversations, reload_conversations_view


def test_missed_event_replays_after_reconnect(authenticated_page, e2e_session_store, e2e_server):
    page = authenticated_page
    open_conversations(page)
    user_id = page.evaluate("Alpine.store('app').userId")
    _url, server = e2e_server

    from tsugite_daemon.session_store import Session, SessionSource

    sid = "20260708_000000_ssereplay"
    e2e_session_store.create_session(
        Session(id=sid, agent="test-agent", source=SessionSource.INTERACTIVE.value, user_id=user_id, title="before")
    )
    reload_conversations_view(page)
    page.wait_for_function("!!window.__tsugiteEventStream", timeout=5000)

    # Sever the stream (sleep/wake analogue), let an event happen while
    # disconnected, then reconnect - the replay must deliver it.
    page.evaluate("window.__tsugiteEventStream.pause()")
    page.wait_for_selector(".console-conn-lost", timeout=5000)  # visible disconnected indicator
    server.event_bus.emit("session_update", {"action": "titled", "id": sid, "title": "renamed while offline"})
    page.evaluate("window.__tsugiteEventStream.resume()")

    page.wait_for_function(
        f"""(() => {{
            const v = Alpine.$data(document.querySelector({CONV_VIEW!r}));
            const s = v.allSessions.find(x => x.id === '{sid}');
            return s && s.title === 'renamed while offline';
        }})()""",
        timeout=8000,
    )
    page.wait_for_selector(".console-conn-lost", state="detached", timeout=5000)
    page.screenshot(path="/tmp/tsugite-issue-447-replay.png", full_page=True)


def test_epoch_change_triggers_full_resync(authenticated_page, e2e_server):
    """A daemon restart hands out a new epoch; the client must detect it on
    reconnect and fire the full-reload path instead of trusting a delta."""
    page = authenticated_page
    open_conversations(page)
    _url, server = e2e_server
    page.wait_for_function("!!window.__tsugiteEventStream", timeout=5000)

    original_epoch = server.event_bus.epoch
    try:
        page.evaluate("window.__tsugiteEventStream.pause()")
        server.event_bus.epoch = "restarted-epoch"
        page.evaluate("Alpine.store('app').lastEvent = null")
        page.evaluate("window.__tsugiteEventStream.resume()")
        page.wait_for_function(
            "Alpine.store('app').lastEvent && Alpine.store('app').lastEvent.type === 'reconnect'",
            timeout=8000,
        )
    finally:
        server.event_bus.epoch = original_epoch
