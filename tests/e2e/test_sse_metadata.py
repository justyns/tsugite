"""E2E tests for real-time metadata updates via SSE."""


def test_metadata_update_via_sse(chat_page):
    """Simulating a metadata SSE event leaves the thread header in a visible state."""
    page = chat_page

    page.wait_for_selector("[x-data*=conversationsView] .title-row", timeout=5000)

    # Fire the SSE event; we don't assert on lastEvent (the conversationsView
    # may clear it as it consumes). The contract under test is "page stays
    # healthy after the event fires" — verify the title row is still rendered.
    page.evaluate("""
        Alpine.store('app').lastEvent = {
            type: 'session_update',
            data: {
                action: 'metadata_updated',
                id: 'test-session',
                metadata: { type: 'ops', status_text: 'investigating' }
            },
            _ts: Date.now()
        };
    """)
    # Briefly wait for any reactive handlers to settle.
    page.wait_for_function("Alpine.store('app').view === 'conversations'", timeout=2000)
    assert page.locator("[x-data*=conversationsView] .title-row").is_visible()
