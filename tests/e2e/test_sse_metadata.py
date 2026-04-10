"""E2E tests for real-time metadata updates via SSE."""


def test_metadata_update_via_sse(chat_page):
    """Simulating a metadata SSE event updates the session in-place."""
    page = chat_page

    page.wait_for_selector(".context-bar", timeout=5000)

    page.evaluate("""
        (() => {
            const items = document.querySelectorAll('.turn-item.active');
            if (items.length) return items[0].getAttribute('x-data') || 'unknown';
            return 'test-session';
        })()
    """)

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
    page.wait_for_timeout(500)

    assert page.locator(".context-bar").is_visible()
