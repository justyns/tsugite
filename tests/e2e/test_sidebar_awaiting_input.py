"""The sidebar must flag a session that is blocked waiting on the user's reply.

When an agent calls `ask_user` / `ask_user_batch`, the session sits idle until the
user answers. Without a sidebar marker it's easy to miss that a session needs
attention (especially one you aren't currently viewing). The `ask_user` event is
broadcast to every client, so `_updateProgressCache` can flag the session for the
sidebar regardless of which session is selected; the next event (the agent resumes
once answered) clears it.
"""

VIEW = "[x-data*=conversationsView]"


def _view_call(page, expr):
    return page.evaluate(f"(() => {{ const view = Alpine.$data(document.querySelector({VIEW!r})); return {expr}; }})()")


def _emit_session_event(page, session_id, event_type, extra=""):
    page.evaluate(
        f"""([sid, et]) => {{
            const view = Alpine.$data(document.querySelector({VIEW!r}));
            view._updateProgressCache({{ session_id: sid, event_type: et{(", " + extra) if extra else ""} }});
        }}""",
        [session_id, event_type],
    )


def _load_session_into_sidebar(page, base_url, session_id):
    page.goto(base_url + "#conversations")
    page.wait_for_function("Alpine.store('app').view === 'conversations'", timeout=3000)
    page.evaluate(f"Alpine.$data(document.querySelector({VIEW!r})).reload()")
    page.wait_for_function(
        f"(() => {{ const v = Alpine.$data(document.querySelector({VIEW!r})); "
        f"return v && v.allSessions && v.allSessions.some(s => s.id === {session_id!r}); }})()",
        timeout=5000,
    )


def test_sidebar_marks_session_awaiting_user_reply(authenticated_page, e2e_session_store, base_url):
    page = authenticated_page
    user_id = page.evaluate("Alpine.store('app').userId")
    session = e2e_session_store.get_or_create_interactive(user_id, "test-agent")
    _load_session_into_sidebar(page, base_url, session.id)

    # Agent blocks on ask_user -> broadcast reaches _updateProgressCache.
    _emit_session_event(page, session.id, "ask_user")

    label = _view_call(page, f"view.sessionProgressLabel(view.allSessions.find(s => s.id === {session.id!r}))")
    dot = _view_call(page, f"view.dotClassNames(view.allSessions.find(s => s.id === {session.id!r}))")
    page.screenshot(path="/tmp/tsugite-issue-state.png", full_page=True)

    assert "waiting" in label.lower(), f"sidebar label should flag the wait, got {label!r}"
    assert "awaiting" in dot, f"status dot should carry an 'awaiting' marker, got {dot!r}"

    # The rendered sidebar row must actually show it (distinct from running/working).
    waiting_label = page.locator(".console-session .row2 .live-label.awaiting")
    assert waiting_label.count() > 0, "no awaiting live-label rendered in the sidebar"
    assert "waiting" in waiting_label.first.text_content().lower()


def test_sidebar_awaiting_marker_clears_when_session_resumes(authenticated_page, e2e_session_store, base_url):
    page = authenticated_page
    user_id = page.evaluate("Alpine.store('app').userId")
    session = e2e_session_store.get_or_create_interactive(user_id, "test-agent")
    _load_session_into_sidebar(page, base_url, session.id)

    _emit_session_event(page, session.id, "ask_user")
    assert (
        "waiting"
        in _view_call(page, f"view.sessionProgressLabel(view.allSessions.find(s => s.id === {session.id!r}))").lower()
    )

    # The user answers; the agent resumes and emits its next event.
    _emit_session_event(page, session.id, "turn_start", extra="turn: 2")

    label = _view_call(page, f"view.sessionProgressLabel(view.allSessions.find(s => s.id === {session.id!r}))")
    dot = _view_call(page, f"view.dotClassNames(view.allSessions.find(s => s.id === {session.id!r}))")
    assert "waiting" not in label.lower(), f"awaiting marker should clear once resumed, got {label!r}"
    assert "awaiting" not in dot, f"dot 'awaiting' marker should clear once resumed, got {dot!r}"


def _job_update(page, session_id, job_id, state):
    page.evaluate(
        f"""([sid, jid, st]) => {{
            const view = Alpine.$data(document.querySelector({VIEW!r}));
            view._handleJobUpdate({{ parent_session_id: sid, job_id: jid, state: st, prompt: 'x' }});
        }}""",
        [session_id, job_id, state],
    )


def test_sidebar_marks_session_with_awaiting_input_job_and_clears_on_resume(
    authenticated_page, e2e_session_store, base_url
):
    """A job paused on a question (awaiting_input) must flag its OWNING session
    in the sidebar - the toast fades, the durable marker must not."""
    page = authenticated_page
    user_id = page.evaluate("Alpine.store('app').userId")
    session = e2e_session_store.get_or_create_interactive(user_id, "test-agent")
    _load_session_into_sidebar(page, base_url, session.id)

    _job_update(page, session.id, "job-ni01", "awaiting_input")

    label = _view_call(page, f"view.sessionProgressLabel(view.allSessions.find(s => s.id === {session.id!r}))")
    dot = _view_call(page, f"view.dotClassNames(view.allSessions.find(s => s.id === {session.id!r}))")
    assert "input" in label.lower(), f"sidebar label should flag the blocked job, got {label!r}"
    assert "awaiting" in dot, f"status dot should carry the 'awaiting' marker, got {dot!r}"

    # Answered via respond_to_job -> the job resumes -> the marker clears.
    _job_update(page, session.id, "job-ni01", "running")
    label = _view_call(page, f"view.sessionProgressLabel(view.allSessions.find(s => s.id === {session.id!r}))")
    assert "input" not in label.lower(), f"marker should clear once the job resumes, got {label!r}"


def _dispatch_event(page, event_type, data):
    page.evaluate(
        f"""(payload) => {{
            const view = Alpine.$data(document.querySelector({VIEW!r}));
            view._handleAttentionEvent(payload.type, payload.data);
        }}""",
        {"type": event_type, "data": data},
    )


def test_sidebar_marks_session_on_cc_permission_prompt_until_cleared(authenticated_page, e2e_session_store, base_url):
    """A cc permission prompt (needs_attention) must set the persistent marker on
    the owning session, and attention_cleared (the next Stop) must clear it."""
    page = authenticated_page
    user_id = page.evaluate("Alpine.store('app').userId")
    session = e2e_session_store.get_or_create_interactive(user_id, "test-agent")
    _load_session_into_sidebar(page, base_url, session.id)

    _dispatch_event(
        page, "needs_attention", {"job_id": "job-pp01", "parent_session_id": session.id, "message": "approve?"}
    )
    label = _view_call(page, f"view.sessionProgressLabel(view.allSessions.find(s => s.id === {session.id!r}))")
    assert "input" in label.lower(), f"a permission prompt must set the durable marker, got {label!r}"

    _dispatch_event(page, "attention_cleared", {"job_id": "job-pp01", "parent_session_id": session.id})
    label = _view_call(page, f"view.sessionProgressLabel(view.allSessions.find(s => s.id === {session.id!r}))")
    assert "input" not in label.lower(), f"attention_cleared must clear the marker, got {label!r}"


def test_sidebar_aggregate_counts_sessions_needing_input(authenticated_page, e2e_session_store, base_url):
    """The sidebar surfaces an at-a-glance count of sessions awaiting input."""
    page = authenticated_page
    user_id = page.evaluate("Alpine.store('app').userId")
    session = e2e_session_store.get_or_create_interactive(user_id, "test-agent")
    _load_session_into_sidebar(page, base_url, session.id)

    assert _view_call(page, "view.awaitingCount()") == 0
    _emit_session_event(page, session.id, "ask_user")
    assert _view_call(page, "view.awaitingCount()") == 1, "an ask_user-blocked session must count"

    chip = page.locator("[data-testid='awaiting-count']")
    assert chip.count() == 1 and chip.is_visible(), "the aggregate chip must render when sessions need input"
    assert "1" in (chip.text_content() or "")

    _emit_session_event(page, session.id, "turn_start", extra="turn: 2")
    assert _view_call(page, "view.awaitingCount()") == 0
    assert not chip.is_visible(), "the chip must hide when nothing needs input"
