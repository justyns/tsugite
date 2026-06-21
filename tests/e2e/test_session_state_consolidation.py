"""Per-session UI state is consolidated under `sessionsState[sid]` (umbrella #298).

Every field that's conceptually per-session — statusInfo, sessionEffort,
sessionModel, compactionSummary, compactedIntoEvent, liveProgress, plus the
compaction/loadedSkills fields covered by their own tests — lives in one
`sessionsState[sid]` object. The orchestrator exposes scalar getters that
read from `sessionsState[selectedSessionId]` so templates don't need to change.

These tests assert that setting a field on session A and switching to B
leaves B's view at the field's default.
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


def _seed_session_state_js(sid: str) -> str:
    """JS that seeds every consolidated field on `sid`. Returned as a string so
    the seed and the readback can run in ONE page.evaluate - no event-loop tick
    (and thus no late selectSession network callback) can interleave and clobber
    the seed, which was the source of this test's flake under parallel load."""
    return f"""
        const v = Alpine.$data(document.querySelector({CONV_VIEW!r}));
        const s = v._sessionState({sid!r});
        s.statusInfo = {{ message_count: 42, tokens: 1234, model: 'gpt-a-only' }};
        s.effort = 'high';
        s.model = 'gpt-a-only';
        s.compactionSummary = 'summary visible only on A';
        s.compactedIntoEvent = {{ ts: '2026-05-15T00:00:00Z', data: {{ replaced_count: 5, retained_count: 2 }} }};
        s.liveProgress = {{ type: 'progress', steps: [], statusText: 'A in progress', turnCount: 1, toolCount: 0 }};
        s.loadedSkills = [{{ name: 'skill-from-a', description: '' }}];
        s.compacting = true;
    """


def test_per_session_state_does_not_bleed_across_sessions(authenticated_page, e2e_session_store):
    """Every consolidated per-session field reads default on the other session."""
    page = authenticated_page
    user_id = page.evaluate("Alpine.store('app').userId")
    a = _make_session(e2e_session_store, user_id)
    b = _make_session(e2e_session_store, user_id)

    open_conversations(page)
    reload_conversations_view(page)

    _select(page, a.id)

    # Seed A's state and read it back ATOMICALLY in one evaluate so no async
    # selectSession response can land between the write and the read.
    a_view = page.evaluate(
        f"""(() => {{
            {_seed_session_state_js(a.id)}
            return {{
                statusInfo: v.statusInfo,
                effort: v.sessionEffort,
                model: v.sessionModel,
                summary: v.compactionSummary,
                compactedInto: v.compactedIntoEvent,
                liveProgress: v._sessionProgress,
                loadedSkills: v.loadedSkills,
                compacting: v.compacting,
            }};
        }})()"""
    )
    assert a_view["statusInfo"].get("message_count") == 42, "A should see its own statusInfo"
    assert a_view["effort"] == "high"
    assert a_view["model"] == "gpt-a-only"
    assert a_view["summary"] == "summary visible only on A"
    assert a_view["compactedInto"]["data"]["replaced_count"] == 5
    assert a_view["liveProgress"]["statusText"] == "A in progress"
    assert any(s.get("name") == "skill-from-a" for s in a_view["loadedSkills"])
    assert a_view["compacting"] is True

    _select(page, b.id)

    b_view = page.evaluate(
        f"""({{
            statusInfo: Alpine.$data(document.querySelector({CONV_VIEW!r})).statusInfo,
            effort: Alpine.$data(document.querySelector({CONV_VIEW!r})).sessionEffort,
            model: Alpine.$data(document.querySelector({CONV_VIEW!r})).sessionModel,
            summary: Alpine.$data(document.querySelector({CONV_VIEW!r})).compactionSummary,
            compactedInto: Alpine.$data(document.querySelector({CONV_VIEW!r})).compactedIntoEvent,
            liveProgress: Alpine.$data(document.querySelector({CONV_VIEW!r}))._sessionProgress,
            loadedSkills: Alpine.$data(document.querySelector({CONV_VIEW!r})).loadedSkills,
            compacting: Alpine.$data(document.querySelector({CONV_VIEW!r})).compacting,
        }})"""
    )
    # B's loadStatus call on select repopulates statusInfo from the API; the assertion
    # is that A's specific marker value didn't bleed across (A had 42 messages).
    assert b_view["statusInfo"].get("message_count") != 42, f"B saw A's statusInfo: {b_view['statusInfo']}"
    assert b_view["statusInfo"].get("model") != "gpt-a-only", "B saw A's model in statusInfo"
    assert b_view["effort"] == "", f"B saw A's effort: {b_view['effort']!r}"
    assert b_view["model"] == "", f"B saw A's model: {b_view['model']!r}"
    assert b_view["summary"] is None, f"B saw A's compactionSummary: {b_view['summary']!r}"
    assert b_view["compactedInto"] is None, "B saw A's compactedIntoEvent"
    assert b_view["liveProgress"] is None, "B saw A's liveProgress"
    assert b_view["loadedSkills"] == [], f"B saw A's loadedSkills: {b_view['loadedSkills']}"
    assert b_view["compacting"] is False, "B saw A's compacting flag"


def test_session_state_is_single_object_not_parallel_maps(authenticated_page, e2e_session_store):
    """The orchestrator exposes one `sessionsState` map, not many `*BySession` parallel maps.

    Regression guard: keeps future contributors from re-introducing the
    parallel-map pattern that #298's audit was about retiring.
    """
    page = authenticated_page
    user_id = page.evaluate("Alpine.store('app').userId")
    a = _make_session(e2e_session_store, user_id)
    open_conversations(page)
    reload_conversations_view(page)
    _select(page, a.id)

    # These names previously named per-session parallel maps. After consolidation
    # none of them should exist as a top-level field on the orchestrator — all
    # per-session state lives inside `sessionsState[sid]` and is read via getters.
    forbidden_names = [
        "compactingBySession",
        "compactingCountsBySession",
        "compactingPhaseBySession",
        "loadedSkillsBySession",
        "statusInfoBySession",
        "sessionEffortBySession",
        "sessionModelBySession",
        "compactionSummaryBySession",
        "compactedIntoEventBySession",
        "messagesBySession",
        "sendingBySession",
        "_activeReadersBySession",
        "historyEventsCache",
        "historyLoadingBySession",
        "progressCache",
        "_prefetchInFlight",
    ]
    leaked = page.evaluate(
        f"""(() => {{
            const v = Alpine.$data(document.querySelector({CONV_VIEW!r}));
            return {forbidden_names!r}.filter(n => v[n] !== undefined);
        }})()"""
    )
    assert leaked == [], f"Per-session state still exposed as parallel BySession maps: {leaked}"
    has_state = page.evaluate(f"typeof Alpine.$data(document.querySelector({CONV_VIEW!r})).sessionsState === 'object'")
    assert has_state, "Consolidated sessionsState container is missing"
