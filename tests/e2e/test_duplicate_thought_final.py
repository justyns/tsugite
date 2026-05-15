"""Regression: agent's final-reply bubble must not render twice when the per-chat
streaming `thought` event handler and the `final_result` event handler both push
agent bubbles for the same prose (issue #290 — separate from #289)."""

from .helpers import CONV_VIEW


def _wait_idle(page):
    page.wait_for_function(
        f"(() => {{ const v = Alpine.$data(document.querySelector({CONV_VIEW!r})); "
        f"return v && !v.sessionsState[v.selectedSessionId]?.sending; }})()",
        timeout=15000,
    )


def _snapshot(page):
    return page.evaluate(
        f"""(() => {{
            const v = Alpine.$data(document.querySelector({CONV_VIEW!r}));
            const sid = v.selectedSessionId;
            return (v.sessionsState[sid]?.messages || []).map(m => ({{
                type: m.type,
                text: (m.text || '').slice(0, 120),
            }}));
        }})()"""
    )


def test_thought_and_final_result_do_not_duplicate_agent_bubble(chat_page, mock_chat):
    """Issue #290: when a turn's final prose is both emitted as a `thought` event
    (from tsugite/core/agent.py:813-820) and returned as the `final_result`, the
    streaming reader pushes two agent bubbles with identical text.

    Expected: exactly one agent bubble carrying the response text.
    """
    response_text = "Acknowledged. Watching for the dupe."
    mock_chat(response_text, events=[("thought", {"content": response_text})])

    page = chat_page
    textarea = page.locator("textarea#message-input")
    textarea.fill("Trigger thought + final_result")
    textarea.press("Enter")

    page.wait_for_selector(".console-turn.agent", timeout=15000)
    _wait_idle(page)

    page.screenshot(path="/tmp/tsugite-issue-state.png", full_page=True)

    snapshot = _snapshot(page)
    dup_count = sum(1 for m in snapshot if m["type"] == "agent" and response_text in (m["text"] or ""))
    assert dup_count == 1, f"Expected 1 agent bubble with response, got {dup_count}. snapshot={snapshot}"


def test_final_result_alone_produces_single_bubble(chat_page, mock_chat):
    """Adjacent: no `thought` event at all -> exactly one agent bubble from final_result."""
    response_text = "Plain reply, no thought."
    mock_chat(response_text)

    page = chat_page
    page.locator("textarea#message-input").fill("hi")
    page.locator("textarea#message-input").press("Enter")

    page.wait_for_selector(".console-turn.agent", timeout=15000)
    _wait_idle(page)

    snapshot = _snapshot(page)
    count = sum(1 for m in snapshot if m["type"] == "agent" and response_text in (m["text"] or ""))
    assert count == 1, f"Expected exactly 1 agent bubble, got {count}. snapshot={snapshot}"


def test_thought_and_final_with_different_text_keep_both_bubbles(chat_page, mock_chat):
    """Adjacent: when `thought` content differs from `final_result`, both bubbles must remain."""
    reasoning = "Private reasoning step A."
    final_text = "Public answer B."
    mock_chat(final_text, events=[("thought", {"content": reasoning})])

    page = chat_page
    page.locator("textarea#message-input").fill("differing-content")
    page.locator("textarea#message-input").press("Enter")

    page.wait_for_selector(".console-turn.agent", timeout=15000)
    _wait_idle(page)

    snapshot = _snapshot(page)
    reasoning_bubbles = [m for m in snapshot if m["type"] == "agent" and reasoning in (m["text"] or "")]
    final_bubbles = [m for m in snapshot if m["type"] == "agent" and final_text in (m["text"] or "")]
    assert len(reasoning_bubbles) == 1, f"Expected 1 reasoning bubble, snapshot={snapshot}"
    assert len(final_bubbles) == 1, f"Expected 1 final bubble, snapshot={snapshot}"


def test_two_turns_each_dedupe_independently(chat_page, mock_chat):
    """Adjacent: two consecutive turns each with thought+final yield 2 agent bubbles (one per turn)."""
    r1 = "First turn answer."
    r2 = "Second turn answer."

    page = chat_page
    textarea = page.locator("textarea#message-input")

    mock_chat(r1, events=[("thought", {"content": r1})])
    textarea.fill("first")
    textarea.press("Enter")
    page.wait_for_selector(".console-turn.agent", timeout=15000)
    _wait_idle(page)

    mock_chat(r2, events=[("thought", {"content": r2})])
    textarea.fill("second")
    textarea.press("Enter")
    page.wait_for_function(
        f"(() => {{ const v = Alpine.$data(document.querySelector({CONV_VIEW!r})); "
        f"const sid = v.selectedSessionId; const s = v.sessionsState[sid]; "
        f"return s && !s.sending && s.messages.some(m => m.type==='agent' && (m.text||'').includes({r2!r})); }})()",
        timeout=15000,
    )

    snapshot = _snapshot(page)
    c1 = sum(1 for m in snapshot if m["type"] == "agent" and r1 in (m["text"] or ""))
    c2 = sum(1 for m in snapshot if m["type"] == "agent" and r2 in (m["text"] or ""))
    assert c1 == 1, f"Turn 1: expected 1 bubble, got {c1}. snapshot={snapshot}"
    assert c2 == 1, f"Turn 2: expected 1 bubble, got {c2}. snapshot={snapshot}"


def test_thought_and_final_whitespace_only_diff_still_dedupes(chat_page, mock_chat):
    """Adjacent: thought vs final differing only in leading/trailing whitespace should still dedupe."""
    final_text = "Trimmed answer."
    thought_text = f"  {final_text}\n"
    mock_chat(final_text, events=[("thought", {"content": thought_text})])

    page = chat_page
    page.locator("textarea#message-input").fill("ws-diff")
    page.locator("textarea#message-input").press("Enter")

    page.wait_for_selector(".console-turn.agent", timeout=15000)
    _wait_idle(page)

    snapshot = _snapshot(page)
    count = sum(1 for m in snapshot if m["type"] == "agent" and final_text in (m["text"] or ""))
    assert count == 1, f"Whitespace-only diff should dedupe to 1 bubble, got {count}. snapshot={snapshot}"
