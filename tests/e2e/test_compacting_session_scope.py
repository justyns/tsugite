"""Per-session scoping of the 'compacting…' indicator.

Bug 297: When session A is compacting, switching to session B in the same
agent erroneously also shows the indicator because the client tracks compaction
state as a single global flag and the server-side SSE event doesn't carry
session_id.

The fix threads session_id through the SSE payload and routes per-session
state through the consolidated `sessionsState[sid]` container.
"""

from tsugite.daemon.session_store import Session, SessionSource
from tsugite.history.storage import generate_session_id

from .helpers import CONV_VIEW, open_conversations, reload_conversations_view


def _make_session(store, user_id):
    sid = generate_session_id("test-agent")
    s = Session(id=sid, agent="test-agent", source=SessionSource.INTERACTIVE.value, user_id=user_id)
    store.create_session(s)
    return s


def _select(page, sid: str) -> None:
    page.evaluate(
        f"Alpine.$data(document.querySelector({CONV_VIEW!r})).selectSessionById({sid!r}, {{follow: false}})"
    )
    page.wait_for_function(
        f"Alpine.$data(document.querySelector({CONV_VIEW!r})).selectedSessionId === {sid!r}",
        timeout=3000,
    )


def _fire_compaction_started(page, agent: str, session_id: str) -> None:
    """Inject an SSE compaction_started event with session_id.

    The fix requires the server to include session_id in the payload; this
    test asserts the client routes the spinner correctly when it does.
    """
    page.evaluate(
        f"""
        Alpine.store('app').lastEvent = {{
            type: 'compaction_started',
            data: {{ agent: {agent!r}, session_id: {session_id!r} }},
            _ts: Date.now()
        }};
        """
    )


def test_compacting_indicator_does_not_bleed_across_sessions(authenticated_page, e2e_session_store):
    """A compaction_started event for session A must not light up session B's spinner.

    User is viewing B; server fires compaction_started for A. Pre-fix, the
    client SSE handler set a global `compacting` flag and B's templates
    rendered the indicator. Post-fix, the per-session map keeps B clean.
    """
    page = authenticated_page
    user_id = page.evaluate("Alpine.store('app').userId")
    a = _make_session(e2e_session_store, user_id)
    b = _make_session(e2e_session_store, user_id)

    open_conversations(page)
    reload_conversations_view(page)

    _select(page, b.id)

    _fire_compaction_started(page, "test-agent", a.id)
    page.wait_for_timeout(150)

    page.screenshot(path="/tmp/tsugite-issue-state.png", full_page=True)

    is_compacting_b = page.evaluate(
        f"Alpine.$data(document.querySelector({CONV_VIEW!r})).compacting"
    )
    assert is_compacting_b is False, (
        "session B saw 'compacting' flag flip even though session A is the one compacting"
    )

    banner_count = page.locator(".console-composer .console-compaction-banner").count()
    assert banner_count == 0, "composer banner appeared on a non-compacting session"


def test_compaction_lifecycle_for_other_session_does_not_touch_viewer(authenticated_page, e2e_session_store):
    """Full compaction_started → progress → finished flow for session A
    must not flip any per-session state for session B (the one the user is viewing).

    Pre-fix, the global flag (and counts/phase scalars) flipped on with any
    `compaction_started` event for the agent and stayed wrong until the next
    full reload.
    """
    page = authenticated_page
    user_id = page.evaluate("Alpine.store('app').userId")
    a = _make_session(e2e_session_store, user_id)
    b = _make_session(e2e_session_store, user_id)

    open_conversations(page)
    reload_conversations_view(page)

    _select(page, b.id)

    _fire_compaction_started(page, "test-agent", a.id)
    page.evaluate(
        f"""
        Alpine.store('app').lastEvent = {{
            type: 'compaction_progress',
            data: {{ agent: 'test-agent', session_id: {a.id!r}, phase: 'starting',
                     replaced_count: 5, retained_count: 2 }},
            _ts: Date.now()
        }};
        """
    )
    page.wait_for_timeout(100)

    state_during_a = page.evaluate(
        f"""({{
            compacting: Alpine.$data(document.querySelector({CONV_VIEW!r})).compacting,
            counts: Alpine.$data(document.querySelector({CONV_VIEW!r})).compactingCounts,
            phase: Alpine.$data(document.querySelector({CONV_VIEW!r})).compactingPhase,
            aCompacting: Alpine.$data(document.querySelector({CONV_VIEW!r})).sessionsState[{a.id!r}]?.compacting || false,
        }})"""
    )
    assert state_during_a["compacting"] is False, (
        "B's compacting flag flipped while A was being compacted"
    )
    assert state_during_a["counts"] is None, "B's compactingCounts is non-null during A's compaction"
    assert state_during_a["phase"] is None, "B's compactingPhase is non-null during A's compaction"
    assert state_during_a["aCompacting"] is True, "A's compacting flag was not set"

    page.evaluate(
        f"""
        Alpine.store('app').lastEvent = {{
            type: 'compaction_finished',
            data: {{ agent: 'test-agent', session_id: {a.id!r} }},
            _ts: Date.now()
        }};
        """
    )
    page.wait_for_timeout(100)

    state_after = page.evaluate(
        f"""({{
            compacting: Alpine.$data(document.querySelector({CONV_VIEW!r})).compacting,
            aCompacting: Alpine.$data(document.querySelector({CONV_VIEW!r})).sessionsState[{a.id!r}]?.compacting || false,
        }})"""
    )
    assert state_after["aCompacting"] is False, "A's compacting flag was not cleared after compaction_finished"
    assert state_after["compacting"] is False, "B's view shows compacting after A's compaction finished"
