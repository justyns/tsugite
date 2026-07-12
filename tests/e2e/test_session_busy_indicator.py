"""The UI must render busy state from the server's authoritative flag, never
infer it from cached progress labels.

The failure this pins (seen live on mobile): a long turn runs server-side, the
client reconnects / PWA-resumes with no fresh progress events, the sidebar
shows the session as idle - and sends get rejected with 409 "a turn is already
running" with no visible explanation."""

from .helpers import CONV_VIEW, open_conversations, reload_conversations_view


def test_busy_session_shows_running_despite_stale_progress(authenticated_page, e2e_session_store):
    """A session whose payload says busy=true must render as running even when
    this client has no progress events for it at all."""
    page = authenticated_page
    open_conversations(page)
    user_id = page.evaluate("Alpine.store('app').userId")

    from tsugite_daemon.session_store import Session, SessionSource

    sid = "20260706_000000_busytest_aa1"
    e2e_session_store.create_session(
        Session(id=sid, agent="test-agent", source=SessionSource.INTERACTIVE.value, user_id=user_id, title="busy one")
    )
    e2e_session_store.begin_turn(sid)
    reload_conversations_view(page)

    label = page.evaluate(
        """([sel, sid]) => {
            const v = Alpine.$data(document.querySelector(sel));
            const s = v.allSessions.find(x => x.id === sid);
            const fresh = { busy: s?.busy ?? null, label: v.sessionProgressLabel(s), dot: v.dotClassNames(s) };
            // The disappearing-label shape: a stale progress cache that looks
            // "between turns" (events seen, none live) while the server says a
            // turn is in flight. This is exactly a reconnect/PWA resume that
            // missed the turn's broadcasts.
            v._sessionState(sid).progress = { turnCount: 0, toolCount: 0, statusText: '', lastEventTime: '2026-07-06T00:00:00+00:00' };
            const stale = { label: v.sessionProgressLabel(s), dot: v.dotClassNames(s) };
            return { fresh, stale };
        }""",
        [CONV_VIEW, sid],
    )
    page.screenshot(path="/tmp/tsugite-busy-indicator.png", full_page=True)
    assert label["fresh"]["busy"] is True, "payload must carry busy"
    assert label["fresh"]["label"] != "", "busy session must never look idle"
    assert "pulse" in label["fresh"]["dot"], "busy session dot must show the running pulse"
    assert label["stale"]["label"] == "Working...", (
        f"stale between-turns cache must not blank a busy session's label; got {label['stale']['label']!r}"
    )
    assert "pulse" in label["stale"]["dot"]

    e2e_session_store.end_turn(sid)


def test_idle_active_session_does_not_pulse(authenticated_page, e2e_session_store):
    """An active session with NO turn in flight and no cached progress must show
    a steady dot. The old optimistic fallback pulsed every active/running
    session the client had never streamed - i.e. most of the sidebar."""
    page = authenticated_page
    open_conversations(page)
    user_id = page.evaluate("Alpine.store('app').userId")

    from tsugite_daemon.session_store import Session, SessionSource

    sid = "20260711_000000_idletest_bb2"
    e2e_session_store.create_session(
        Session(id=sid, agent="test-agent", source=SessionSource.INTERACTIVE.value, user_id=user_id, title="idle one")
    )
    # No begin_turn: the session is persistent-active but idle.
    reload_conversations_view(page)

    result = page.evaluate(
        """([sel, sid]) => {
            const v = Alpine.$data(document.querySelector(sel));
            const s = v.allSessions.find(x => x.id === sid);
            // The sidebar-at-large case: an active session this client has no
            // progress cache for at all (never streamed it this page-load, or
            // the between-turns reconcile nulled it).
            v._sessionState(sid).progress = null;
            return { busy: s?.busy ?? null, state: s?.state, dot: v.dotClassNames(s), inFlight: v.sessionTurnInFlight(s) };
        }""",
        [CONV_VIEW, sid],
    )
    assert not result["busy"], f"precondition: no turn in flight, got busy={result['busy']}"
    assert result["state"] == "active", f"precondition: persistent-active session, got {result['state']!r}"
    assert "pulse" not in result["dot"], f"an idle {result['state']} session must not pulse, got {result['dot']!r}"
    assert result["inFlight"] is False


def test_mid_turn_progress_still_pulses_without_busy_flag(authenticated_page, e2e_session_store):
    """The narrow case the optimistic fallback was for: live progress events are
    arriving but the busy broadcast hasn't landed yet. Fresh mid-turn statusText
    must still pulse."""
    page = authenticated_page
    open_conversations(page)
    user_id = page.evaluate("Alpine.store('app').userId")

    from tsugite_daemon.session_store import Session, SessionSource

    sid = "20260711_000000_midturn_cc3"
    e2e_session_store.create_session(
        Session(id=sid, agent="test-agent", source=SessionSource.INTERACTIVE.value, user_id=user_id, title="mid turn")
    )
    reload_conversations_view(page)

    result = page.evaluate(
        """([sel, sid]) => {
            const v = Alpine.$data(document.querySelector(sel));
            const s = v.allSessions.find(x => x.id === sid);
            v._updateProgressCache({ session_id: sid, event_type: 'tool_start', tool: 'bash' });
            return { dot: v.dotClassNames(s), inFlight: v.sessionTurnInFlight(s) };
        }""",
        [CONV_VIEW, sid],
    )
    assert result["inFlight"] is True, "fresh mid-turn progress must count as in flight even before busy lands"
    assert "pulse" in result["dot"]


def test_409_send_restores_draft_and_marks_busy(chat_page):
    """A 409 turn-conflict is not a 'Connection error': the optimistic user
    bubble is withdrawn, the draft comes back, and the session renders busy."""
    page = chat_page

    page.route(
        "**/api/agents/*/chat",
        lambda route: route.fulfill(
            status=409,
            content_type="application/json",
            body='{"error": "a turn is already running for this session", "code": "turn_in_flight"}',
        ),
    )

    result = page.evaluate(
        """(sel) => {
            const v = Alpine.$data(document.querySelector(sel));
            v.messageText = 'hello while busy';
            return v.sendMessage().then(() => {
                const msgs = v.messages.map(m => ({ type: m.type, text: (m.text || '').slice(0, 90) }));
                const s = v.allSessions.find(x => (x.conversation_id || x.id) === v.selectedSessionId);
                return { msgs, draft: v.messageText, busy: s?.busy ?? null };
            });
        }""",
        CONV_VIEW,
    )
    page.screenshot(path="/tmp/tsugite-busy-409.png", full_page=True)

    assert result["draft"] == "hello while busy", "the unsent draft must be restored"
    assert not any(m["type"] == "user" and m["text"] == "hello while busy" for m in result["msgs"]), (
        "the optimistic user bubble must be withdrawn - the message was never accepted"
    )
    assert not any(m["type"] == "error" for m in result["msgs"]), "a turn conflict is not a connection error"
    info = [m for m in result["msgs"] if m["type"] == "info"]
    assert info and "already running" in info[-1]["text"]
    assert result["busy"] is True, "the 409 must flip the cached session to busy"


def test_409_session_finished_shows_error_not_busy(chat_page):
    """The finished-session 409 must NOT mark the session busy - it needs the
    machine-readable code to distinguish it from a turn conflict."""
    page = chat_page

    page.route(
        "**/api/agents/*/chat",
        lambda route: route.fulfill(
            status=409,
            content_type="application/json",
            body='{"error": "Session is completed. Start a new session to continue.", "code": "session_finished"}',
        ),
    )

    result = page.evaluate(
        """(sel) => {
            const v = Alpine.$data(document.querySelector(sel));
            v.messageText = 'hello finished';
            return v.sendMessage().then(() => {
                const msgs = v.messages.map(m => ({ type: m.type, text: (m.text || '').slice(0, 90) }));
                const s = v.allSessions.find(x => (x.conversation_id || x.id) === v.selectedSessionId);
                return { msgs, draft: v.messageText, busy: s?.busy ?? null };
            });
        }""",
        CONV_VIEW,
    )
    assert result["draft"] == "hello finished", "draft must still be restored"
    assert result["busy"] is not True, "a finished-session conflict must not render the session as busy"
    errors = [m for m in result["msgs"] if m["type"] == "error"]
    assert errors and "Start a new session" in errors[-1]["text"]
