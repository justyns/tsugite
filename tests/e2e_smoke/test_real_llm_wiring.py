"""Real-LLM smoke tests.

Each test costs real provider tokens. Keep the prompts tiny and the
assertions loose — we're testing wiring, not model behaviour.

Gated by TSUGITE_E2E_REAL_LLM + provider key (see conftest).
"""


def _select_session(page, session_id):
    page.locator(".console-tabs button.console-tab", has_text="Conversations").click()
    page.wait_for_function("Alpine.store('app').view === 'conversations'", timeout=3000)
    page.evaluate("Alpine.$data(document.querySelector('[x-data*=conversationsView]')).reload()")
    page.wait_for_function(
        f"(() => {{ const v = Alpine.$data(document.querySelector('[x-data*=conversationsView]')); "
        f"return v && v.allSessions && v.allSessions.some(s => s.id === {session_id!r}); }})()",
        timeout=5000,
    )
    page.evaluate(
        f"Alpine.$data(document.querySelector('[x-data*=conversationsView]'))"
        f".selectSessionById({session_id!r}, {{follow: false}})"
    )
    page.wait_for_selector("textarea#message-input", timeout=5000)


def test_browser_send_message_gets_real_response(smoke_authenticated_page, smoke_session_store):
    """Through-the-browser smoke: send a message, see the real model respond in the bubble.

    The `.console-turn.agent` selector matches both progress bubbles
    (msg.type='progress', role 'tool') and final agent text (msg.type='agent',
    role 'agent'). Wait on the Alpine state for an actual agent message with
    the expected token instead.
    """
    page = smoke_authenticated_page
    user_id = page.evaluate("Alpine.store('app').userId")
    session = smoke_session_store.get_or_create_interactive(user_id, "smoke-agent")
    _select_session(page, session.id)

    page.locator("textarea#message-input").fill("Reply with one short word.")
    page.locator("textarea#message-input").press("Enter")

    # Wiring smoke: a real agent text bubble arrives. We don't pin the content
    # because the point of this tier is wiring, not model behaviour.
    page.wait_for_function(
        "Alpine.$data(document.getElementById('messages')).messages"
        ".some(m => m.type === 'agent' && (m.text || '').trim().length > 0)",
        timeout=60000,
    )


def test_direct_chat_endpoint_returns_real_response(smoke_base_url, smoke_auth_token):
    """Bypass the browser: POST /chat and read the real model's reply from the SSE stream."""
    import json

    import requests

    headers = {"Authorization": f"Bearer {smoke_auth_token}"}
    with requests.post(
        f"{smoke_base_url}/api/agents/smoke-agent/chat",
        json={"message": "Reply with one short word.", "user_id": "smoke-direct"},
        headers=headers,
        timeout=120,
        stream=True,
    ) as resp:
        assert resp.status_code == 200, f"unexpected status {resp.status_code}: {resp.text[:300]}"
        final = None
        for raw_line in resp.iter_lines(decode_unicode=True):
            if not raw_line or not raw_line.startswith("data: "):
                continue
            try:
                event = json.loads(raw_line[len("data: ") :])
            except json.JSONDecodeError:
                continue
            if event.get("type") == "final_result":
                final = event
                break

    # Wiring smoke: the endpoint completes the SSE stream and produces a
    # final_result with a non-empty result. Content is the model's call.
    assert final is not None, "no final_result event in SSE stream"
    result_text = final.get("data", {}).get("result") or final.get("result") or ""
    assert result_text.strip(), f"final_result has empty result: {final!r}"


def test_streaming_progress_events_reach_browser(smoke_authenticated_page, smoke_session_store):
    """Real streaming: a progress bubble appears in the DOM during a real turn."""
    page = smoke_authenticated_page
    user_id = page.evaluate("Alpine.store('app').userId")
    session = smoke_session_store.get_or_create_interactive(user_id, "smoke-agent")
    _select_session(page, session.id)

    page.locator("textarea#message-input").fill("Reply with exactly: ok.")
    page.locator("textarea#message-input").press("Enter")

    # `.console-codeblock` only renders when there's a progress message in the
    # bubble stream — proves SSE progress events landed and were rendered.
    page.wait_for_function(
        "Alpine.$data(document.getElementById('messages')).messages"
        ".some(m => m.type === 'progress' || m.type === 'progress-done')",
        timeout=60000,
    )
    # And the run must complete with an agent text bubble.
    page.wait_for_function(
        "Alpine.$data(document.getElementById('messages')).messages"
        ".some(m => m.type === 'agent' && (m.text || '').trim().length > 0)",
        timeout=60000,
    )
