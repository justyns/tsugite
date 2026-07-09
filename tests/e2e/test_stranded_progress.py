"""A live "Working..." progress bubble must never outlive the turn it
represents. Three strand paths, all real user reports:

1. Revisit: reselecting a session whose turn already ended, with a stale
   progress cache saying "working", kept the stale bubble because the
   ended-scan only ran on first visit.
2. Passive turn-end: `_handleSessionEvent` ignored final_result/error/cancelled
   entirely, deferring finalize to a conditional history reload that may never
   fire (non-interactive sessions being watched, dropped history_update).
3. Missed terminal event (disconnect/sleep): nothing arrives at all - a
   watchdog must notice the server is not busy and finalize."""

from unittest.mock import patch

from tsugite.history.storage import SessionStorage

from .helpers import CONV_VIEW, open_conversations, reload_conversations_view


def _mk_ended_session(e2e_session_store, user_id, sid, e2e_tmp=None):
    from tsugite_daemon.session_store import Session, SessionSource

    e2e_session_store.create_session(
        Session(id=sid, agent="test-agent", source=SessionSource.INTERACTIVE.value, user_id=user_id, title="ended")
    )
    e2e_session_store.append_event(sid, {"type": "user_input", "data": {"content": "do the thing"}})
    e2e_session_store.append_event(sid, {"type": "final_result", "data": {"result": "done thing"}})
    if e2e_tmp is not None:
        # The events above drive the replay/ended-scan; the history backend is
        # what loadHistory renders - seed it too so the settled turn has content.
        history_dir = e2e_tmp / "history"
        history_dir.mkdir(exist_ok=True)
        path = history_dir / f"{sid}.jsonl"
        if path.exists():
            path.unlink()
        storage = SessionStorage.create("test-agent", model="claude_code:sonnet", session_path=path)
        storage.record("user_input", text="do the thing")
        storage.record("turn_start", turn=1)
        storage.record("final_result", result="done thing", turns=1)
        storage.record("session_end", status="success")


def test_revisit_settles_stale_working_bubble(authenticated_page, e2e_session_store, e2e_tmp):
    page = authenticated_page
    open_conversations(page)
    user_id = page.evaluate("Alpine.store('app').userId")

    sid = "20260708_000000_strand_rev1"
    other = "20260708_000000_strand_other"
    _mk_ended_session(e2e_session_store, user_id, sid, e2e_tmp)
    _mk_ended_session(e2e_session_store, user_id, other, e2e_tmp)
    reload_conversations_view(page)

    with patch("tsugite.history.storage.get_history_dir", return_value=e2e_tmp / "history"):
        result = page.evaluate(
            """async ([sel, sid, other]) => {
                const v = Alpine.$data(document.querySelector(sel));
                await v.selectSessionById(sid);
                await new Promise(r => setTimeout(r, 200));
                // Simulate the stranded state a dropped terminal event leaves behind:
                // a live bubble in this session's messages + a stale "working" cache.
                const bubble = { type: 'progress', steps: [], statusText: 'Working...', turnCount: 1, toolCount: 0 };
                v._sessionProgress = bubble;
                v.messages.push(bubble);
                v._sessionState(sid).progress = { turnCount: 1, toolCount: 0, statusText: 'Working...', lastEventTime: new Date().toISOString() };

                // Navigate away and back - the revisit path must reconcile against
                // the event log (which says the turn ended) and settle the bubble.
                const orig = v._rehydrateProgressFromEvents.bind(v);
                let reconciled = false;
                v._rehydrateProgressFromEvents = (...a) => { reconciled = true; return orig(...a); };
                await v.selectSessionById(other);
                await v.selectSessionById(sid);
                await new Promise(r => setTimeout(r, 400));
                return {
                    reconciled,
                    liveBubbles: v.messages.filter(m => m.type === 'progress').length,
                    hasFinal: v.messages.some(m => (m.text || '').includes('done thing')),
                };
            }""",
            [CONV_VIEW, sid, other],
        )
    page.screenshot(path="/tmp/tsugite-issue-450-revisit.png", full_page=True)
    assert result["reconciled"], (
        "revisit with a live-looking turn must reconcile against the event log (was first-visit only)"
    )
    assert result["liveBubbles"] == 0, "revisit must settle a stale Working... bubble when the log says the turn ended"
    assert result["hasFinal"], "the settled turn's real content should render instead"


def test_passive_turn_end_finalizes_bubble(chat_page):
    page = chat_page
    result = page.evaluate(
        """(sel) => {
            const v = Alpine.$data(document.querySelector(sel));
            const sid = v.selectedSessionId;
            v._handleSessionEvent({ session_id: sid, event_type: 'tool_call', tool: 'fetch_text' });
            const midTurn = v._sessionProgress ? v._sessionProgress.type : null;
            v._handleSessionEvent({ session_id: sid, event_type: 'final_result', result: 'ok' });
            const after = v.messages.filter(m => m.type === 'progress').length;
            return { midTurn, liveBubblesAfter: after, cleared: v._sessionProgress === null };
        }""",
        CONV_VIEW,
    )
    assert result["midTurn"] == "progress", "passive events must create the live bubble (precondition)"
    assert result["liveBubblesAfter"] == 0, "a turn-end event must finalize the passive bubble directly"
    assert result["cleared"], "_sessionProgress must be released at turn end"


def test_watchdog_finalizes_when_server_not_busy(chat_page):
    page = chat_page
    page.route(
        "**/api/agents/*/status**",
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body='{"busy": false, "model": "m", "tokens": 0, "message_count": 0, "metadata": {}, "attachments": []}',
        ),
    )
    result = page.evaluate(
        """async (sel) => {
            const v = Alpine.$data(document.querySelector(sel));
            const sid = v.selectedSessionId;
            const bubble = { type: 'progress', steps: [], statusText: 'Working...', turnCount: 1, toolCount: 0 };
            v._sessionProgress = bubble;
            v.messages.push(bubble);
            // Stale: last event a minute ago, server reports not busy.
            v._sessionState(sid).progress = { turnCount: 1, toolCount: 0, statusText: 'Working...', lastEventTime: new Date(Date.now() - 60000).toISOString() };
            await v._reconcileStaleProgress();
            return {
                liveBubbles: v.messages.filter(m => m.type === 'progress').length,
                cacheStatus: v._sessionState(sid).progress.statusText,
            };
        }""",
        CONV_VIEW,
    )
    page.screenshot(path="/tmp/tsugite-issue-450-watchdog.png", full_page=True)
    assert result["liveBubbles"] == 0, "watchdog must finalize a quiet bubble once the server reports not busy"
    assert result["cacheStatus"] == "", "the stale cached statusText must clear so the sidebar stops showing working"
