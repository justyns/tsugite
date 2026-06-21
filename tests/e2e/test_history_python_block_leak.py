"""A model_response whose executable ```python block calls
return_value(value=\"\"\"...\"\"\") with a NESTED markdown fence inside the string
must NOT leak the block's tail as a phantom agent bubble on history replay (#389).

The frontend's old _stripCodeFences regex was non-syntax-aware: it stopped at the
first nested ``` (the inner ```bash opener), stripped only the head of the python
block, and left the rest (`bash`, the commands, the trailing `\"\"\")`) as visible
prose. Worse, that phantom prose set sawInlineAgent, which suppressed the real
answer carried by the following final_result. The fix suppresses prose entirely
for executable turns (those containing a ```python fence); plain prose responses
still render.
"""

from unittest.mock import patch

from tsugite.history.storage import SessionStorage

from .helpers import CONV_VIEW, open_session_by_url

# Executable turn: a ```python block calling return_value with a triple-quoted
# string that itself contains a ```bash fence. `'''` outer so the inner `\"\"\"`
# and backticks sit verbatim. The old stripper left LEAKED_yay_command and the
# python-string terminator `\"\"\")` behind as prose.
_RAW_WITH_NESTED_FENCE = '''```python
return_value(value="""Done. Recommended cleanup commands:

```bash
LEAKED_yay_command
paccache -ruk0
```

Schedule this weekly.""")
```'''

# Event-199 shape: two ```python blocks (runtime executes only the first, drops
# the second) plus trailing prose that was never live-visible.
_RAW_TWO_BLOCKS_TRAILING_PROSE = """Let me check for a duplicate first.

```python
search_issues(query="x")
```

No duplicate. Filing it.

```python
create_issue(title="x")
```

PHANTOM_filed_issue and captured the syntax."""

_REAL_ANSWER = "All done: packages upgraded and the cache was trimmed."


def _new_session(e2e_adapter, e2e_tmp, label):
    unique_user = f"web-user-{label}"
    session = e2e_adapter.session_store.get_or_create_interactive(unique_user, "test-agent")
    history_dir = e2e_tmp / f"history-{label}"
    history_dir.mkdir(exist_ok=True)
    session_path = history_dir / f"{session.id}.jsonl"
    if session_path.exists():
        session_path.unlink()
    storage = SessionStorage.create("test-agent", model="test", session_path=session_path)
    return storage, history_dir, unique_user, session.id


def _render_body(page, history_dir, user_id, session_id, *, screenshot=None):
    with patch("tsugite_daemon.adapters.http.get_history_dir", return_value=history_dir):
        open_session_by_url(page, page.url.split("#")[0], user_id, session_id)
        page.wait_for_selector(".console-turn.user", timeout=5000)
        page.wait_for_selector(".console-turn.agent", timeout=5000)
        if screenshot:
            page.screenshot(path=screenshot, full_page=True)
        return page.evaluate(f"document.querySelector({CONV_VIEW!r}).innerText")


def test_nested_fence_python_block_does_not_leak_as_prose(authenticated_page, e2e_adapter, e2e_tmp):
    page = authenticated_page
    storage, history_dir, user_id, session_id = _new_session(e2e_adapter, e2e_tmp, "pyleak")
    storage.record("user_input", text="clean up the system")
    storage.record("model_response", provider="test", model="test", raw_content=_RAW_WITH_NESTED_FENCE)
    storage.record("code_execution", code='return_value(value="""...""")', output="", duration_ms=2)
    storage.record("final_result", result=_REAL_ANSWER, turns=1)

    body = _render_body(page, history_dir, user_id, session_id, screenshot="/tmp/tsugite-issue-state.png")

    # The contents of the executable python block must not surface as prose.
    assert "LEAKED_yay_command" not in body, "python-block content leaked into the rendered conversation"
    assert '""")' not in body, "return_value string terminator leaked as prose"
    # No agent bubble should carry the leaked fragment.
    phantom = page.evaluate(
        "[...document.querySelectorAll('.console-turn.agent')]"
        ".filter(t => t.innerText.includes('LEAKED_yay_command') || t.innerText.includes('\"\"\")')).length"
    )
    assert phantom == 0, f"{phantom} phantom agent bubble(s) rendered the python-block tail"
    # The real answer (from final_result) must still render - it was being
    # suppressed by the phantom inline-agent bubble.
    assert _REAL_ANSWER in body, "the real final answer must still render"


def test_two_python_blocks_with_trailing_prose_no_phantom(authenticated_page, e2e_adapter, e2e_tmp):
    """Event-199 shape: a response with two ```python fences plus trailing prose
    (only the first block ran) must not render that trailing prose as a phantom
    answer before the code execution / final result.
    """
    page = authenticated_page
    storage, history_dir, user_id, session_id = _new_session(e2e_adapter, e2e_tmp, "twoblocks")
    storage.record("user_input", text="file an issue")
    storage.record("model_response", provider="test", model="test", raw_content=_RAW_TWO_BLOCKS_TRAILING_PROSE)
    storage.record("code_execution", code='search_issues(query="x")', output="[]", duration_ms=2)
    storage.record("final_result", result=_REAL_ANSWER, turns=1)

    body = _render_body(page, history_dir, user_id, session_id)

    assert "PHANTOM_filed_issue" not in body, "trailing prose from an executable turn rendered as a phantom bubble"
    assert _REAL_ANSWER in body, "the real final answer must still render"


def test_plain_prose_response_still_renders(authenticated_page, e2e_adapter, e2e_tmp):
    """A model_response with no ```python fence is a normal prose answer and must
    render as an agent bubble (the suppression rule is scoped to executable turns).
    """
    page = authenticated_page
    storage, history_dir, user_id, session_id = _new_session(e2e_adapter, e2e_tmp, "plainprose")
    storage.record("user_input", text="status?")
    storage.record(
        "model_response",
        provider="test",
        model="test",
        raw_content="Here is the summary: everything PLAIN_PROSE_OK looks healthy.",
    )

    body = _render_body(page, history_dir, user_id, session_id)

    assert "PLAIN_PROSE_OK" in body, "plain prose response was wrongly suppressed"
    rendered = page.evaluate(
        "[...document.querySelectorAll('.console-turn.agent')].filter(t => t.innerText.includes('PLAIN_PROSE_OK')).length"
    )
    assert rendered == 1, f"expected exactly one agent bubble with the prose, got {rendered}"
