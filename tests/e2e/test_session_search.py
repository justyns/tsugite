"""The sidebar search box must find sessions outside the loaded recency
window and match topic metadata - a session that exists must never produce
'no results' (real case: searching "design" missed "Incremental Game Design"
because it was recency rank 120 of 1005)."""

from .helpers import CONV_VIEW, open_conversations, reload_conversations_view


def test_search_finds_session_outside_loaded_window(authenticated_page, e2e_session_store):
    page = authenticated_page
    open_conversations(page)
    user_id = page.evaluate("Alpine.store('app').userId")

    from tsugite_daemon.session_store import Session, SessionSource

    for i in range(110):
        sid = f"20260708_1{i:04d}_filler"
        e2e_session_store.create_session(
            Session(
                id=sid, agent="test-agent", source=SessionSource.INTERACTIVE.value, user_id=user_id, title=f"filler {i}"
            )
        )
        e2e_session_store._sessions[sid].last_active = f"2026-07-08T{10 + i // 60:02d}:{i % 60:02d}:00+00:00"
    old_sid = "20260601_000000_design"
    e2e_session_store.create_session(
        Session(
            id=old_sid,
            agent="test-agent",
            source=SessionSource.INTERACTIVE.value,
            user_id=user_id,
            title="Incremental Game Design",
            metadata={"topic": "Idle infra-provider game design (homelab to interplanetary)"},
        )
    )
    e2e_session_store._sessions[old_sid].last_active = "2026-06-01T00:00:00+00:00"

    reload_conversations_view(page)

    loaded = page.evaluate(
        f"""(sel) => {{
            const v = Alpine.$data(document.querySelector(sel));
            return v.allSessions.some(s => s.id === '{old_sid}');
        }}""",
        CONV_VIEW,
    )
    assert not loaded, "precondition: the target session must be outside the loaded window"

    page.fill('input[aria-label="Filter sessions"]', "design")
    row = page.locator(".console-session-item, [class*=session]", has_text="Incremental Game Design").first
    row.wait_for(state="visible", timeout=5000)
    page.screenshot(path="/tmp/tsugite-issue-445-found.png", full_page=True)

    found = page.evaluate(
        f"""(sel) => {{
            const v = Alpine.$data(document.querySelector(sel));
            const s = v.allSessions.find(x => x.id === '{old_sid}');
            return {{ merged: !!s, matches: s ? v._matchesFilters(s) : false }};
        }}""",
        CONV_VIEW,
    )
    assert found["merged"] and found["matches"]


def test_topic_metadata_matches_client_side(authenticated_page, e2e_session_store):
    """A loaded session whose match lives only in metadata.topic must match the
    client-side filter (title/label/id alone used to be the searchable text)."""
    page = authenticated_page
    open_conversations(page)
    user_id = page.evaluate("Alpine.store('app').userId")

    from tsugite_daemon.session_store import Session, SessionSource

    sid = "20260708_000000_topiconly"
    e2e_session_store.create_session(
        Session(
            id=sid,
            agent="test-agent",
            source=SessionSource.INTERACTIVE.value,
            user_id=user_id,
            title="Untitled chat",
            metadata={"topic": "kerbal-style rocket telemetry"},
        )
    )
    reload_conversations_view(page)

    result = page.evaluate(
        f"""(sel) => {{
            const v = Alpine.$data(document.querySelector(sel));
            v.sessionFilter = 'telemetry';
            const s = v.allSessions.find(x => x.id === '{sid}');
            return s ? v._matchesFilters(s) : null;
        }}""",
        CONV_VIEW,
    )
    assert result is True, "topic metadata must be part of the searchable text"
