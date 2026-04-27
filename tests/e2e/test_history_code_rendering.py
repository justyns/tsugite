"""E2E tests for step-trace rendering: code truncation, content blocks, and tool results."""

from unittest.mock import patch

from tsugite.history.storage import SessionStorage


def _seed_isolated_turn(page, e2e_adapter, e2e_tmp, label, messages, final_answer="done"):
    """Seed a fresh session (unique user) with one crafted turn and return (history_dir, user_id, session_id)."""
    unique_user = f"web-user-{label}"
    session = e2e_adapter.session_store.get_or_create_interactive(unique_user, "test-agent")
    history_dir = e2e_tmp / f"history-{label}"
    history_dir.mkdir(exist_ok=True)
    session_path = history_dir / f"{session.id}.jsonl"

    storage = SessionStorage.create("test-agent", model="test", session_path=session_path)
    storage.record_turn(messages=messages, final_answer=final_answer)
    return history_dir, unique_user, session.id


def _open_progress_trace(page, user_id, session_id):
    """Reload as `user_id`, navigate to the session, and expand the progress-done summary."""
    page.evaluate(f"localStorage.setItem('tsugite_user_id', {user_id!r})")
    page.goto(page.url.split("#")[0] + f"#conversations?session={session_id}")
    page.reload()
    page.wait_for_function("!Alpine.store('app').authRequired", timeout=5000)
    page.wait_for_function(f"Alpine.store('app').userId === {user_id!r}", timeout=3000)
    page.wait_for_selector(".msg.user", timeout=5000)
    page.wait_for_selector(".msg.progress", timeout=5000)
    summary = page.locator(".msg.progress .tool-summary").first
    summary.click()


def test_history_code_block_not_truncated(authenticated_page, e2e_adapter, e2e_tmp):
    """Long code blocks must render in full on history reload, no trailing ellipsis."""
    page = authenticated_page

    long_code = "\n".join([f"line_{i} = 'x' * 80" for i in range(40)])
    assistant_msg = f"```python\n{long_code}\n```"
    history_dir, user_id, session_id = _seed_isolated_turn(
        page,
        e2e_adapter,
        e2e_tmp,
        "truncate",
        messages=[
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": assistant_msg},
        ],
    )

    with patch("tsugite.daemon.adapters.http.get_history_dir", return_value=history_dir):
        _open_progress_trace(page, user_id, session_id)
        code_details = page.locator(".msg.progress .tool-steps li details").first
        code_details.click()
        code_text = page.locator(".msg.progress .tool-steps li details pre code").first.text_content()

    assert "line_39 = 'x'" in code_text
    assert not code_text.endswith("...")
    assert len(code_text) >= len(long_code)


def test_history_content_block_renders_next_to_its_code(authenticated_page, e2e_adapter, e2e_tmp):
    """Content blocks should appear inline with the assistant message that declared them."""
    page = authenticated_page

    history_dir, user_id, session_id = _seed_isolated_turn(
        page,
        e2e_adapter,
        e2e_tmp,
        "inline",
        messages=[
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": "```python\ninvestigate()\n```"},
            {
                "role": "user",
                "content": '<tsugite_execution_result status="success"><output>ok</output></tsugite_execution_result>',
            },
            {
                "role": "assistant",
                "content": (
                    '```python\nfinal_answer(result="done")\n```\n\n'
                    '<content name="reply">Stopped before creating.</content>'
                ),
            },
        ],
    )

    with patch("tsugite.daemon.adapters.http.get_history_dir", return_value=history_dir):
        _open_progress_trace(page, user_id, session_id)

        steps = page.locator(".msg.progress .tool-steps > li")
        step_count = steps.count()
        # Find the index of the content block and the 2nd code step
        cb_index = None
        code_indices = []
        for i in range(step_count):
            li = steps.nth(i)
            if li.locator("details.content-block").count() > 0:
                cb_index = i
            elif "code" in (li.text_content() or ""):
                code_indices.append(i)

        assert cb_index is not None, "content block should be rendered"
        assert len(code_indices) >= 2, "both code steps should render"
        # content block came from the 2nd code step, so it must appear immediately after it,
        # not pushed to the end of the trace
        assert cb_index == code_indices[1] + 1


def test_history_tool_result_visible_between_code_steps(authenticated_page, e2e_adapter, e2e_tmp):
    """tsugite_execution_result with attributes must render a tool_result step."""
    page = authenticated_page

    observation = (
        '<tsugite_execution_result status="success" duration_ms="7">'
        "<output>hello world</output></tsugite_execution_result>"
    )
    history_dir, user_id, session_id = _seed_isolated_turn(
        page,
        e2e_adapter,
        e2e_tmp,
        "toolresult",
        messages=[
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": "```python\nprint('hi')\n```"},
            {"role": "user", "content": observation},
            {"role": "assistant", "content": "```python\nfinal_answer('done')\n```"},
        ],
    )

    with patch("tsugite.daemon.adapters.http.get_history_dir", return_value=history_dir):
        _open_progress_trace(page, user_id, session_id)

        summaries = page.locator(".msg.progress .tool-steps > li details > summary")
        labels = [summaries.nth(i).text_content() or "" for i in range(summaries.count())]

    # Two code steps plus a result step in between
    code_hits = [i for i, s in enumerate(labels) if "code" in s]
    result_hits = [i for i, s in enumerate(labels) if "result" in s]
    assert len(code_hits) == 2
    assert len(result_hits) == 1
    assert code_hits[0] < result_hits[0] < code_hits[1]


def test_history_tool_result_unescapes_xml_entities(authenticated_page, e2e_adapter, e2e_tmp):
    """The tsugite_execution_result XML envelope escapes <, >, & in <output>.

    When this gets rendered in the history view's tool_result step, the UI
    must decode those entities back to their literal characters — otherwise
    the user sees "#179 -&gt; #179" instead of "#179 -> #179".
    """
    page = authenticated_page

    # Raw stdout would be:  a -> b  <tag>  x & y
    # After executor.to_xml escaping:  a -&gt; b  &lt;tag&gt;  x &amp; y
    observation = (
        '<tsugite_execution_result status="success" duration_ms="7">'
        "<output>a -&gt; b  &lt;tag&gt;  x &amp; y</output>"
        "</tsugite_execution_result>"
    )
    history_dir, user_id, session_id = _seed_isolated_turn(
        page,
        e2e_adapter,
        e2e_tmp,
        "entities",
        messages=[
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": "```python\nprint('hi')\n```"},
            {"role": "user", "content": observation},
            {"role": "assistant", "content": "```python\nfinal_answer('done')\n```"},
        ],
    )

    with patch("tsugite.daemon.adapters.http.get_history_dir", return_value=history_dir):
        _open_progress_trace(page, user_id, session_id)
        result_details = (
            page.locator(".msg.progress .tool-steps > li details")
            .filter(has=page.locator("summary", has_text="result"))
            .first
        )
        result_details.click()
        text = (
            page.locator(".msg.progress .tool-steps > li details pre code").filter(has_text="a ").first.text_content()
            or ""
        )

    assert "a -> b" in text, f"HTML entity &gt; not decoded. Got: {text!r}"
    assert "<tag>" in text, f"HTML entities &lt;/&gt; not decoded. Got: {text!r}"
    assert "x & y" in text, f"HTML entity &amp; not decoded. Got: {text!r}"
    assert (
        "&gt;" not in text and "&lt;" not in text and "&amp;" not in text
    ), f"Literal entity text still present. Got: {text!r}"


def test_history_tool_result_not_truncated(authenticated_page, e2e_adapter, e2e_tmp):
    """Large tool_result outputs (e.g. git log, issue listings) must not be
    chopped off at an arbitrary 500-char limit when viewing history. The
    live-stream view shows full content; history must too.
    """
    page = authenticated_page

    lines = [f"commit-{i:04d}|2026-04-{(i % 28) + 1:02d}|some commit message here" for i in range(40)]
    full_output = "\n".join(lines)
    assert len(full_output) > 1500, "fixture must exceed the old 500-char cap"
    head_line = lines[0]
    tail_line = lines[-1]

    observation = (
        '<tsugite_execution_result status="success" duration_ms="3">'
        f"<output>{full_output}</output>"
        "</tsugite_execution_result>"
    )
    history_dir, user_id, session_id = _seed_isolated_turn(
        page,
        e2e_adapter,
        e2e_tmp,
        "notrunc",
        messages=[
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": "```python\nprint('hi')\n```"},
            {"role": "user", "content": observation},
            {"role": "assistant", "content": "```python\nfinal_answer('done')\n```"},
        ],
    )

    with patch("tsugite.daemon.adapters.http.get_history_dir", return_value=history_dir):
        _open_progress_trace(page, user_id, session_id)
        result_details = (
            page.locator(".msg.progress .tool-steps > li details")
            .filter(has=page.locator("summary", has_text="result"))
            .first
        )
        result_details.click()
        text = (
            page.locator(".msg.progress .tool-steps > li details pre code")
            .filter(has_text="commit-0000")
            .first.text_content()
            or ""
        )

    assert head_line in text, f"first line missing. Got first 200: {text[:200]!r}"
    assert tail_line in text, f"last line missing — content was truncated before the end. Got last 200: {text[-200:]!r}"
    assert "..." not in text[-10:], f"trailing ellipsis suggests truncation. Tail: {text[-50:]!r}"


def test_history_content_block_survives_no_code_turn(authenticated_page, e2e_adapter, e2e_tmp):
    """A sub-turn that only emitted content blocks (no code) must still render them after reload."""
    page = authenticated_page

    history_dir, user_id, session_id = _seed_isolated_turn(
        page,
        e2e_adapter,
        e2e_tmp,
        "nocode",
        messages=[
            {"role": "user", "content": "go"},
            {
                "role": "assistant",
                "content": '<content name="reply">Just a content block, no code.</content>',
            },
            {
                "role": "user",
                "content": '<tsugite_execution_result status="error"><error>no code</error></tsugite_execution_result>',
            },
            {"role": "assistant", "content": "```python\nfinal_answer('done')\n```"},
        ],
    )

    with patch("tsugite.daemon.adapters.http.get_history_dir", return_value=history_dir):
        _open_progress_trace(page, user_id, session_id)
        content_blocks = page.locator(".msg.progress .tool-steps details.content-block")
        assert content_blocks.count() == 1
        content_blocks.first.click()
        pre_text = page.locator(".msg.progress .tool-steps details.content-block pre code").first.text_content()
        assert "Just a content block, no code." in pre_text


def test_history_prose_only_assistant_message_renders(authenticated_page, e2e_adapter, e2e_tmp):
    """An assistant message that is prose only (no code block) must still render on reload.

    Mirrors the real stored shape from _build_turn_messages: a prose-only step (thought as
    assistant message), then the format-error observation, then the corrected final_answer
    code step, its observation, and finally the plain-text result assistant message.
    """
    page = authenticated_page

    history_dir, user_id, session_id = _seed_isolated_turn(
        page,
        e2e_adapter,
        e2e_tmp,
        "prose",
        messages=[
            {"role": "user", "content": "thanks"},
            {"role": "assistant", "content": "You're welcome!"},
            {
                "role": "user",
                "content": (
                    '<tsugite_execution_result status="error">'
                    "<error>Format Error: You must respond with a Python code block.</error>"
                    "</tsugite_execution_result>"
                ),
            },
            {"role": "assistant", "content": '```python\nfinal_answer("You\'re welcome!")\n```'},
            {
                "role": "user",
                "content": ('<tsugite_execution_result status="success"><output></output></tsugite_execution_result>'),
            },
            {"role": "assistant", "content": "You're welcome!"},
        ],
        final_answer="You're welcome!",
    )

    with patch("tsugite.daemon.adapters.http.get_history_dir", return_value=history_dir):
        _open_progress_trace(page, user_id, session_id)

        summaries = page.locator(".msg.progress .tool-steps > li details > summary")
        labels = [summaries.nth(i).text_content() or "" for i in range(summaries.count())]
        thought_hits = [i for i, s in enumerate(labels) if "thought" in s.lower()]
        assert len(thought_hits) == 1, f"expected exactly one 'thought' step, got summaries: {labels}"

        thought_idx = thought_hits[0]
        result_hits = [i for i, s in enumerate(labels) if "result" in s.lower()]
        assert result_hits, f"expected a result step, got summaries: {labels}"
        assert thought_idx < result_hits[0], f"thought must appear before the format-error result, got: {labels}"

        thought_details = page.locator(".msg.progress .tool-steps > li details").nth(thought_idx)
        thought_details.click()
        thought_text = thought_details.locator("pre code").first.text_content()
        assert "You're welcome!" in thought_text
