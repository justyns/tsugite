"""A driven cc-driver job that gets blocked (e.g. on a permission prompt) emits a
needs_attention SSE event; the web UI surfaces it as a toast. The driven terminal
is also embedded in the job tile, but the toast alerts you when you aren't looking
at it. Regression guard: the event used to be emitted with no consumer."""

from .helpers import open_conversations


def test_needs_attention_event_shows_toast(authenticated_page, e2e_server):
    page = authenticated_page
    open_conversations(page)
    _url, server = e2e_server
    page.wait_for_function("!!window.__tsugiteEventStream", timeout=5000)

    server.event_bus.emit("needs_attention", {"job_id": "job-1", "message": "cc job blocked on a permission prompt"})

    toast = page.wait_for_selector(".console-toast", timeout=8000)
    assert "permission prompt" in toast.inner_text()
