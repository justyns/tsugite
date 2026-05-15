"""E2E tests for the redesigned session sidebar."""

from tsugite.daemon.session_store import Session, SessionSource, SessionStatus


def test_sidebar_shows_active_recent_groups(authenticated_page, e2e_session_store):
    """Sessions are grouped into Active and Recent sections."""
    page = authenticated_page
    user_id = page.evaluate("Alpine.store('app').userId")

    # Create an active session
    e2e_session_store.get_or_create_interactive(user_id, "test-agent")

    # Create a completed session
    completed = Session(
        id="completed-1",
        agent="test-agent",
        source=SessionSource.BACKGROUND.value,
        status=SessionStatus.COMPLETED.value,
        prompt="finished task",
        result="done",
    )
    e2e_session_store.create_session(completed)

    page.locator(".console-tabs button.console-tab", has_text="Conversations").click()
    page.wait_for_function("Alpine.store('app').view === 'conversations'", timeout=3000)
    page.evaluate("Alpine.$data(document.querySelector('[x-data*=conversationsView]')).reload()")
    page.wait_for_function(
        "(() => { const v = Alpine.$data(document.querySelector('[x-data*=conversationsView]')); "
        "return v && v.allSessions && v.allSessions.some(s => s.id === 'completed-1'); })()",
        timeout=5000,
    )
    page.wait_for_selector(".console-section-head", timeout=5000)

    headers = [h.lower() for h in page.locator(".console-section-head").all_text_contents()]
    assert any("active" in h for h in headers)
    assert any("recent" in h for h in headers)


def test_sidebar_source_icons_render(authenticated_page, e2e_session_store):
    """Source icons appear next to session entries."""
    page = authenticated_page
    user_id = page.evaluate("Alpine.store('app').userId")

    e2e_session_store.get_or_create_interactive(user_id, "test-agent")

    page.locator(".console-tabs button.console-tab", has_text="Conversations").click()
    page.wait_for_function("Alpine.store('app').view === 'conversations'", timeout=3000)
    page.evaluate("Alpine.$data(document.querySelector('[x-data*=conversationsView]')).reload()")
    page.wait_for_selector(".source-icon", timeout=5000)

    icons = page.locator(".source-icon")
    assert icons.count() >= 1


def test_sidebar_metadata_chips_render(authenticated_page, e2e_session_store):
    """Metadata chips appear on sessions that have metadata."""
    page = authenticated_page

    session = Session(
        id="meta-session",
        agent="test-agent",
        source=SessionSource.BACKGROUND.value,
        status=SessionStatus.COMPLETED.value,
        prompt="task with metadata",
        metadata={"type": "code", "task": "https://example.com/task/1"},
    )
    e2e_session_store.create_session(session)

    page.locator(".console-tabs button.console-tab", has_text="Conversations").click()
    page.wait_for_function("Alpine.store('app').view === 'conversations'", timeout=3000)
    page.evaluate("Alpine.$data(document.querySelector('[x-data*=conversationsView]')).reload()")
    page.wait_for_function(
        "(() => { const v = Alpine.$data(document.querySelector('[x-data*=conversationsView]')); "
        "return v && v.allSessions && v.allSessions.some(s => s.id === 'meta-session'); })()",
        timeout=5000,
    )

    page.wait_for_selector("summary.console-section-head", timeout=5000)
    recent_toggle = page.locator("summary.console-section-head", has_text="recent")
    if recent_toggle.count() > 0:
        # The <details> is `open` by default; click only collapses it. Skip clicking.
        pass

    page.wait_for_selector(".console-session .chip", timeout=3000)
    chips = page.locator(".console-session .chip")
    assert chips.count() >= 1


def test_sidebar_progress_label_clears_after_turn_ends(authenticated_page, e2e_session_store, base_url):
    """A stale live-progress entry on a session's state must yield to the server's cleared progress.

    The bug: after a turn ends, the SSE turn-end event may be missed (reconnect, race
    with loadSessions). sessionsState[sid].progress keeps showing 'Turn N · Thinking...'
    indefinitely even though the server's session_progress_summary correctly reports
    status_text=''. Refreshing the page worked around it. loadSessions must reconcile.
    """
    page = authenticated_page
    user_id = page.evaluate("Alpine.store('app').userId")
    session = e2e_session_store.get_or_create_interactive(user_id, "test-agent")
    e2e_session_store.append_event(session.id, {"type": "turn_start", "turn": 3})
    e2e_session_store.append_event(session.id, {"type": "thought", "text": "thinking"})
    e2e_session_store.append_event(session.id, {"type": "final_result", "result": "done"})

    view_selector = "[x-data*=conversationsView]"
    page.goto(base_url + "#conversations")
    page.wait_for_function("Alpine.store('app').view === 'conversations'", timeout=3000)
    page.evaluate(f"Alpine.$data(document.querySelector({view_selector!r})).reload()")
    page.wait_for_function(
        f"(() => {{ const view = Alpine.$data(document.querySelector({view_selector!r})); "
        f"return view && view.allSessions && view.allSessions.some(s => s.id === {session.id!r}); }})()",
        timeout=5000,
    )

    page.evaluate(
        f"""
        const view = Alpine.$data(document.querySelector({view_selector!r}));
        view._sessionState({session.id!r}).progress = {{
            turnCount: 3, toolCount: 0, statusText: 'Thinking...', lastEventTime: new Date().toISOString()
        }};
        """
    )
    label_before = page.evaluate(
        f"(() => {{ const view = Alpine.$data(document.querySelector({view_selector!r})); "
        f"const s = view.allSessions.find(x => x.id === {session.id!r}); "
        f"return view.sessionProgressLabel(s); }})()"
    )
    assert "Turn 3" in label_before, f"expected stale label, got {label_before!r}"

    page.evaluate(f"Alpine.$data(document.querySelector({view_selector!r})).loadSessions()")
    page.wait_for_function(
        f"(() => {{ const view = Alpine.$data(document.querySelector({view_selector!r})); "
        f"const s = view.allSessions.find(x => x.id === {session.id!r}); "
        f"return view.sessionProgressLabel(s) === ''; }})()",
        timeout=3000,
    )


def test_ui_follows_session_after_compaction(authenticated_page, e2e_session_store, base_url):
    """When the active session compacts, an SSE 'compacted' update auto-follows the UI to the successor.

    Without this, the user keeps interacting with a now-completed predecessor and
    the next /chat POST gets routed to the successor by the server but the UI URL
    and selected-meta state stay on the old session - confusing.
    """
    from tsugite.daemon.session_store import Session, SessionSource, SessionStatus

    page = authenticated_page
    user_id = page.evaluate("Alpine.store('app').userId")
    predecessor = e2e_session_store.get_or_create_interactive(user_id, "test-agent")
    successor = e2e_session_store.create_session(
        Session(
            id="ui-follow-successor",
            agent="test-agent",
            source=SessionSource.INTERACTIVE.value,
            status=SessionStatus.ACTIVE.value,
            user_id=user_id,
        )
    )
    e2e_session_store.update_session(predecessor.id, status=SessionStatus.COMPLETED.value, superseded_by=successor.id)

    view_selector = "[x-data*=conversationsView]"
    page.goto(base_url + f"#conversations?session={predecessor.id}")
    page.wait_for_function("Alpine.store('app').view === 'conversations'", timeout=3000)
    page.evaluate(f"Alpine.$data(document.querySelector({view_selector!r})).reload()")
    # selectSession chases superseded_by forward; explicitly switch to predecessor
    # by URL to simulate a stale tab that loaded before compaction was visible.
    page.evaluate(
        f"const view = Alpine.$data(document.querySelector({view_selector!r})); "
        f"view.selectedSessionId = {predecessor.id!r};"
    )

    page.evaluate(
        f"""
        Alpine.store('app').lastEvent = {{
            type: 'session_update',
            data: {{ action: 'compacted', id: {predecessor.id!r}, successor_id: {successor.id!r} }},
            _ts: Date.now()
        }};
        """
    )
    page.wait_for_function(
        f"Alpine.$data(document.querySelector({view_selector!r})).selectedSessionId === {successor.id!r}",
        timeout=3000,
    )
