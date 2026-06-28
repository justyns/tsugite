"""E2E tests for step-trace rendering: code truncation, content blocks, and tool results.

These tests seed code_execution events directly. The pre-redesign architecture
wrapped code/output inside `<tsugite_execution_result>` XML envelopes in
assistant/user messages; the current event model stores them as fields on a
`code_execution` event. The frontend reconstructs `.tool-step` items from these
events.
"""

from unittest.mock import patch

import pytest

from tsugite.history.storage import SessionStorage

from .helpers import open_session_by_url


def _seed_session(e2e_adapter, e2e_tmp, label):
    """Open an empty SessionStorage for label; caller records the events."""
    unique_user = f"web-user-{label}"
    session = e2e_adapter.session_store.get_or_create_interactive(unique_user, "test-agent")
    history_dir = e2e_tmp / f"history-{label}"
    history_dir.mkdir(exist_ok=True)
    session_path = history_dir / f"{session.id}.jsonl"
    if session_path.exists():
        session_path.unlink()
    storage = SessionStorage.create("test-agent", model="test", session_path=session_path)
    return storage, history_dir, unique_user, session.id


def _open_progress_trace(page, user_id, session_id):
    """Reload as user_id, navigate to the session, and wait for the progress block."""
    open_session_by_url(page, page.url.split("#")[0], user_id, session_id)
    page.wait_for_selector(".console-turn.user", timeout=5000)
    page.wait_for_selector(".console-codeblock", timeout=5000)
    # progress block is open by default (msg._codeOpen ?? true)


def test_history_code_block_not_truncated(authenticated_page, e2e_adapter, e2e_tmp):
    """Long code in a code_execution event renders in full on history reload."""
    page = authenticated_page
    long_code = "\n".join([f"line_{i} = 'x' * 80" for i in range(40)])

    storage, history_dir, user_id, session_id = _seed_session(e2e_adapter, e2e_tmp, "truncate")
    storage.record("user_input", text="go")
    storage.record("code_execution", code=long_code, output="ran 40 lines", duration_ms=12)
    storage.record("session_end", status="success")

    with patch("tsugite.history.storage.get_history_dir", return_value=history_dir):
        _open_progress_trace(page, user_id, session_id)
        code_details = page.locator(".console-codeblock .tool-step details").first
        code_details.click()
        code_text = page.locator(".console-codeblock .tool-step details pre code").first.text_content() or ""

    assert "line_39 = 'x'" in code_text
    assert not code_text.endswith("...")
    assert len(code_text) >= len(long_code)


def test_history_tool_result_visible_between_code_steps(authenticated_page, e2e_adapter, e2e_tmp):
    """Two code_execution events produce a code+result step pair each."""
    page = authenticated_page

    storage, history_dir, user_id, session_id = _seed_session(e2e_adapter, e2e_tmp, "toolresult")
    storage.record("user_input", text="go")
    storage.record("code_execution", code="print('hi')", output="hello world", duration_ms=7)
    storage.record("code_execution", code="return_value('done')", output="", duration_ms=1)
    storage.record("session_end", status="success")

    with patch("tsugite.history.storage.get_history_dir", return_value=history_dir):
        _open_progress_trace(page, user_id, session_id)
        summaries = page.locator(".console-codeblock .tool-step details > summary")
        labels = [(summaries.nth(i).text_content() or "").lower() for i in range(summaries.count())]

    code_hits = [i for i, s in enumerate(labels) if "code" in s]
    result_hits = [i for i, s in enumerate(labels) if "result" in s]
    assert len(code_hits) >= 2, f"expected 2 code summaries, got {labels}"
    assert len(result_hits) >= 1, f"expected at least one result summary, got {labels}"
    assert code_hits[0] < result_hits[0] < code_hits[1], f"order off: {labels}"


def test_history_tool_result_not_truncated(authenticated_page, e2e_adapter, e2e_tmp):
    """Large code_execution outputs render in full — no arbitrary 500-char cap."""
    page = authenticated_page

    lines = [f"commit-{i:04d}|2026-04-{(i % 28) + 1:02d}|some commit message here" for i in range(40)]
    full_output = "\n".join(lines)
    assert len(full_output) > 1500, "fixture must exceed the old 500-char cap"

    storage, history_dir, user_id, session_id = _seed_session(e2e_adapter, e2e_tmp, "notrunc")
    storage.record("user_input", text="go")
    storage.record("code_execution", code="print('hi')", output=full_output, duration_ms=3)
    storage.record("session_end", status="success")

    with patch("tsugite.history.storage.get_history_dir", return_value=history_dir):
        _open_progress_trace(page, user_id, session_id)
        result_details = (
            page.locator(".console-codeblock .tool-step details")
            .filter(has=page.locator("summary", has_text="result"))
            .first
        )
        result_details.click()
        text = result_details.locator("pre code").first.text_content() or ""

    assert lines[0] in text
    assert lines[-1] in text
    assert "..." not in text[-10:]


def test_history_content_block_renders_with_model_response(authenticated_page, e2e_adapter, e2e_tmp):
    """`<content name="reply">...</content>` in raw_content surfaces as a content-block step."""
    page = authenticated_page

    storage, history_dir, user_id, session_id = _seed_session(e2e_adapter, e2e_tmp, "content")
    storage.record("user_input", text="go")
    storage.record(
        "model_response",
        provider="test",
        model="test",
        raw_content='Some prose. <content name="reply">A reply content block.</content>',
    )
    storage.record("session_end", status="success")

    with patch("tsugite.history.storage.get_history_dir", return_value=history_dir):
        _open_progress_trace(page, user_id, session_id)
        content_blocks = page.locator(".console-codeblock .tool-step details.content-block")
        assert content_blocks.count() >= 1
        content_blocks.first.click()
        pre_text = content_blocks.first.locator("pre code").first.text_content() or ""
        assert "A reply content block." in pre_text


@pytest.mark.skip(
    reason="Tests pre-redesign <tsugite_execution_result> XML envelope, which the event model replaced; rewrite needs current format-error event semantics"
)
def test_history_tool_result_unescapes_xml_entities(authenticated_page, e2e_adapter, e2e_tmp):
    pass


@pytest.mark.skip(
    reason="Tests pre-redesign prose-only-as-thought sub-turn semantics; needs new format_error/model_response event-flow contract"
)
def test_history_prose_only_assistant_message_renders(authenticated_page, e2e_adapter, e2e_tmp):
    pass


@pytest.mark.skip(
    reason="Original test asserted content-block ordering against pre-redesign step interleaving; needs rewrite once code_execution+model_response interleave behavior is pinned"
)
def test_history_content_block_survives_no_code_turn(authenticated_page, e2e_adapter, e2e_tmp):
    pass
