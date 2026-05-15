"""Per-session scoping of the `loadedSkills` array (umbrella #298 high-risk).

`skill_loaded` SSE events are emitted from the per-chat stream
(`tsugite/daemon/web/js/views/conversation/streaming.js`) and push into
the global `loadedSkills` array. If the user is viewing session B while
session A's chat stream pushes `skill_loaded`, B's status chip shows
"Skills: N loaded" for A's skills — and `removeLoadedSkill` would target
the wrong session.
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


def test_loaded_skills_do_not_bleed_across_sessions(authenticated_page, e2e_session_store):
    """A skill loaded in session A's stream must not appear in B's loadedSkills."""
    page = authenticated_page
    user_id = page.evaluate("Alpine.store('app').userId")
    a = _make_session(e2e_session_store, user_id)
    b = _make_session(e2e_session_store, user_id)

    open_conversations(page)
    reload_conversations_view(page)

    _select(page, a.id)

    page.evaluate(
        f"""
        const v = Alpine.$data(document.querySelector({CONV_VIEW!r}));
        v._sessionState({a.id!r}).loadedSkills = [
            {{ name: 'skill-a-only', description: 'streamed in session A' }}
        ];
        """
    )

    a_skills = page.evaluate(
        f"Alpine.$data(document.querySelector({CONV_VIEW!r})).loadedSkills"
    )
    assert any(s.get("name") == "skill-a-only" for s in a_skills), (
        "Session A should see its own loaded skill"
    )

    _select(page, b.id)

    b_skills = page.evaluate(
        f"Alpine.$data(document.querySelector({CONV_VIEW!r})).loadedSkills"
    )
    assert not any(s.get("name") == "skill-a-only" for s in b_skills), (
        f"Session B saw session A's loaded skill: {b_skills}"
    )
