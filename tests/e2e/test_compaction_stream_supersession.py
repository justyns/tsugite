"""When auto-compaction rotates the session ID mid-message, the in-flight
stream state (sending flag, live progress, messages) must not be stranded on
the old session's state object. The fix uses a supersession alias so any
closure-captured OLD session id transparently routes to NEW's state.

Pre-fix symptom: user lands on NEW after the `compacted` SSE follows them
forward, but their `sending`/progress reads of `sessionsState[NEW]` return
defaults because everything mid-turn was written under `sessionsState[OLD]`.
"""

from tsugite_daemon.session_store import Session, SessionSource

from tsugite.history.storage import generate_session_id

from .helpers import CONV_VIEW, open_conversations, reload_conversations_view


def _make_session(store, user_id):
    sid = generate_session_id("test-agent")
    s = Session(id=sid, agent="test-agent", source=SessionSource.INTERACTIVE.value, user_id=user_id)
    store.create_session(s)
    return s


def _select(page, sid: str) -> None:
    page.evaluate(f"Alpine.$data(document.querySelector({CONV_VIEW!r})).selectSessionById({sid!r}, {{follow: false}})")
    page.wait_for_function(
        f"Alpine.$data(document.querySelector({CONV_VIEW!r})).selectedSessionId === {sid!r}",
        timeout=3000,
    )


def test_inflight_stream_state_follows_compaction_to_successor(authenticated_page, e2e_session_store):
    """Stream state captured on OLD via the streaming.js closure must surface
    on NEW after `compacted` fires; the alias makes `_sessionState(OLD)`
    return NEW's state object so future writes land where the user is looking.
    """
    page = authenticated_page
    user_id = page.evaluate("Alpine.store('app').userId")
    old = _make_session(e2e_session_store, user_id)
    new = _make_session(e2e_session_store, user_id)
    new.parent_session = old.id
    e2e_session_store._sessions[old.id].superseded_by = new.id

    open_conversations(page)
    reload_conversations_view(page)

    _select(page, old.id)

    page.evaluate(
        f"""
        // Simulate streaming.js mid-message state: closure captured sendSessionId = OLD,
        // so the sending flag, progress bubble, and one pushed user message are all
        // currently sitting on sessionsState[OLD].
        const v = Alpine.$data(document.querySelector({CONV_VIEW!r}));
        const st = v._sessionState({old.id!r});
        st.sending = true;
        st.messages = [{{ type: 'user', text: 'pre-compaction message' }}];
        st.liveProgress = {{ type: 'progress', steps: [], statusText: 'Working...', turnCount: 1, toolCount: 0 }};
        st.loadedSkills = [{{ name: 'pre-compact-skill', description: 'x' }}];
        """
    )

    page.evaluate(
        f"""
        Alpine.store('app').lastEvent = {{
            type: 'session_update',
            data: {{ action: 'compacted', id: {old.id!r}, successor_id: {new.id!r} }},
            _ts: Date.now()
        }};
        """
    )

    page.wait_for_function(
        f"Alpine.$data(document.querySelector({CONV_VIEW!r})).selectedSessionId === {new.id!r}",
        timeout=3000,
    )

    page.screenshot(path="/tmp/tsugite-issue-state.png", full_page=True)

    snap = page.evaluate(
        f"""({{
            sending: Alpine.$data(document.querySelector({CONV_VIEW!r})).sending,
            messages: Alpine.$data(document.querySelector({CONV_VIEW!r})).messages,
            liveProgress: Alpine.$data(document.querySelector({CONV_VIEW!r}))._sessionProgress,
            loadedSkills: Alpine.$data(document.querySelector({CONV_VIEW!r})).loadedSkills,
            // Critical: the streaming.js closure still does _sessionState(OLD).foo = ...
            // That must reach NEW's state through the supersession alias.
            redirected: Alpine.$data(document.querySelector({CONV_VIEW!r}))._sessionState({old.id!r}) ===
                         Alpine.$data(document.querySelector({CONV_VIEW!r}))._sessionState({new.id!r}),
        }})"""
    )

    assert snap["redirected"], (
        "_sessionState(OLD) must route to NEW's state object after compaction so closure-captured "
        "sendSessionId writes land where the user is looking"
    )
    assert snap["sending"] is True, "in-flight sending flag did not surface on NEW after compaction"
    assert len(snap["messages"]) >= 1 and any(m.get("text") == "pre-compaction message" for m in snap["messages"]), (
        f"pre-compaction message was not preserved on NEW: {snap['messages']!r}"
    )
    assert snap["liveProgress"] is not None, "in-flight live progress bubble was lost on compaction"
    assert any(s.get("name") == "pre-compact-skill" for s in snap["loadedSkills"]), (
        f"loadedSkills did not carry over from OLD to NEW: {snap['loadedSkills']!r}"
    )


def test_subsequent_writes_via_old_id_still_reach_successor(authenticated_page, e2e_session_store):
    """After compaction, the streaming.js closure keeps writing through
    `_sessionState(sendSessionId=OLD)`. Each such write must mutate NEW's
    state, not a stranded OLD object the user can't see.
    """
    page = authenticated_page
    user_id = page.evaluate("Alpine.store('app').userId")
    old = _make_session(e2e_session_store, user_id)
    new = _make_session(e2e_session_store, user_id)
    new.parent_session = old.id
    e2e_session_store._sessions[old.id].superseded_by = new.id

    open_conversations(page)
    reload_conversations_view(page)
    _select(page, old.id)

    page.evaluate(
        f"""
        Alpine.store('app').lastEvent = {{
            type: 'session_update',
            data: {{ action: 'compacted', id: {old.id!r}, successor_id: {new.id!r} }},
            _ts: Date.now()
        }};
        """
    )
    page.wait_for_function(
        f"Alpine.$data(document.querySelector({CONV_VIEW!r})).selectedSessionId === {new.id!r}",
        timeout=3000,
    )
    # selectSession's loadHistory races writes by clobbering NEW.messages in-place;
    # wait for it to settle so the writes-via-OLD-id assertion isn't fighting it.
    page.wait_for_function(
        f"!Alpine.$data(document.querySelector({CONV_VIEW!r})).historyLoading",
        timeout=3000,
    )

    page.evaluate(
        f"""
        // Reproduce streaming.js's mid-message writes via the stale OLD id.
        const v = Alpine.$data(document.querySelector({CONV_VIEW!r}));
        v._sessionState({old.id!r}).sending = true;
        v._sessionState({old.id!r}).messages.push({{ type: 'agent', text: 'post-compaction reply' }});
        """
    )

    snap = page.evaluate(
        f"""({{
            sending: Alpine.$data(document.querySelector({CONV_VIEW!r})).sending,
            messages: Alpine.$data(document.querySelector({CONV_VIEW!r})).messages,
        }})"""
    )
    assert snap["sending"] is True, "writes via OLD id did not reach NEW; closure-captured sends stay stranded"
    assert any(m.get("text") == "post-compaction reply" for m in snap["messages"]), (
        f"messages pushed via OLD id did not surface on NEW: {snap['messages']!r}"
    )
