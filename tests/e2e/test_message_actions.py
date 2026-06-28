"""E2E tests for per-message copy actions, code-block copy, and the toast stack.

Covers UI affordances on top of message rendering:
- `.copy-btn` per-message button that writes the raw markdown to the clipboard
- `.copy-code` button injected into every `<pre>` by the markdown renderer
- The general-purpose toast stack driven by the `tsugite:toast` window event
"""

from unittest.mock import patch

from tsugite.history.storage import SessionStorage

from .helpers import open_session_by_url


def _seed_turn(e2e_adapter, e2e_tmp, label, user_text, agent_text):
    """Seed a session JSONL with one user_input + model_response pair."""
    unique_user = f"actions-user-{label}"
    session = e2e_adapter.session_store.get_or_create_interactive(unique_user, "test-agent")
    history_dir = e2e_tmp / f"history-actions-{label}"
    history_dir.mkdir(exist_ok=True)
    session_path = history_dir / f"{session.id}.jsonl"
    if session_path.exists():
        session_path.unlink()
    storage = SessionStorage.create("test-agent", model="test", session_path=session_path)
    storage.record("user_input", text=user_text)
    storage.record("model_response", provider="test", model="test", raw_content=agent_text)
    return history_dir, unique_user, session.id


def _open_session(page, user_id, session_id):
    open_session_by_url(page, page.url.split("#")[0], user_id, session_id)
    page.wait_for_selector(".console-turn.agent", timeout=5000)


def test_copy_markdown_button_present_on_every_rendered_message(authenticated_page, e2e_adapter, e2e_tmp):
    page = authenticated_page
    history_dir, user_id, session_id = _seed_turn(
        e2e_adapter,
        e2e_tmp,
        "present",
        user_text="hi",
        agent_text="# Answer\n\nSome **bold** text.",
    )

    with patch("tsugite.history.storage.get_history_dir", return_value=history_dir):
        _open_session(page, user_id, session_id)

        agent = page.locator(".console-turn.agent").last
        assert agent.locator(".copy-btn").count() == 1

        user = page.locator(".console-turn.user").last
        assert user.locator(".copy-btn").count() == 1

        # Hidden until hover
        opacity = agent.locator(".copy-btn").evaluate("el => getComputedStyle(el).opacity")
        assert opacity == "0"

        agent.hover()
        page.wait_for_function(
            "(() => { const btns = document.querySelectorAll('.console-turn.agent .copy-btn'); "
            "const btn = btns[btns.length - 1]; "
            "return btn && getComputedStyle(btn).opacity === '1'; })()",
            timeout=2000,
        )


def test_copy_markdown_button_writes_raw_markdown_to_clipboard(authenticated_page, e2e_adapter, e2e_tmp):
    page = authenticated_page
    page.context.grant_permissions(["clipboard-read", "clipboard-write"])

    # ```python is reserved for code-execution and stripped from prose; use ```text
    md = "# Heading\n\nA table:\n\n| col |\n| --- |\n| x |\n\n```text\nprint('hi')\n```\n"
    history_dir, user_id, session_id = _seed_turn(
        e2e_adapter,
        e2e_tmp,
        "copy-md",
        user_text="render this",
        agent_text=md,
    )

    with patch("tsugite.history.storage.get_history_dir", return_value=history_dir):
        _open_session(page, user_id, session_id)

        agent = page.locator(".console-turn.agent").last
        agent.hover()
        agent.locator(".copy-btn").click()

        page.wait_for_selector(".toast-stack .console-toast.toast-success", timeout=2000)
        toast_text = (page.locator(".toast-stack .console-toast.toast-success").first.text_content() or "").strip()
        assert toast_text == "Copied"

        clipboard = page.evaluate("navigator.clipboard.readText()")
        # The text in Alpine's msg.text must land in the clipboard byte-for-byte
        # — no HTML escaping, no rendering. Storage may normalize trailing
        # whitespace, so compare against the in-memory value.
        expected = page.evaluate("""
            () => {
              const view = Alpine.$data(document.getElementById('messages'));
              const agents = view.messages.filter(m => m.type === 'agent');
              return agents[agents.length - 1].text;
            }
            """)
        assert clipboard == expected
        assert "# Heading" in clipboard
        assert "| col |" in clipboard
        assert "```text" in clipboard
        assert "<h1" not in clipboard
        assert "<table" not in clipboard


def test_copy_user_message_writes_raw_user_text(authenticated_page, e2e_adapter, e2e_tmp):
    page = authenticated_page
    page.context.grant_permissions(["clipboard-read", "clipboard-write"])

    user_text = "please **render** this and keep _formatting_"
    history_dir, user_id, session_id = _seed_turn(
        e2e_adapter,
        e2e_tmp,
        "copy-user",
        user_text=user_text,
        agent_text="ok",
    )

    with patch("tsugite.history.storage.get_history_dir", return_value=history_dir):
        _open_session(page, user_id, session_id)

        user = page.locator(".console-turn.user").last
        user.hover()
        user.locator(".copy-btn").click()

        page.wait_for_selector(".toast-stack .console-toast.toast-success", timeout=2000)
        clipboard = page.evaluate("navigator.clipboard.readText()")
        assert clipboard == user_text


def test_copy_code_block_button_injected_per_pre(authenticated_page, e2e_adapter, e2e_tmp):
    page = authenticated_page

    md = "```text\nprint('one')\n```\n\nbetween\n\n```js\nconsole.log('two');\n```\n"
    history_dir, user_id, session_id = _seed_turn(
        e2e_adapter,
        e2e_tmp,
        "pre-count",
        user_text="two blocks",
        agent_text=md,
    )

    with patch("tsugite.history.storage.get_history_dir", return_value=history_dir):
        _open_session(page, user_id, session_id)
        agent = page.locator(".console-turn.agent").last

        pre_count = agent.locator("pre").count()
        copy_count = agent.locator("pre .copy-code").count()
        assert pre_count == 2
        assert copy_count == pre_count


def test_copy_code_block_copies_full_code_preserving_whitespace(authenticated_page, e2e_adapter, e2e_tmp):
    """Regression: `textContent` (not `innerText`) must be used so whitespace is preserved verbatim."""
    page = authenticated_page
    page.context.grant_permissions(["clipboard-read", "clipboard-write"])

    long_code = "\n".join(f"    line_{i} = {i}" for i in range(40))
    # ```python is reserved for code-execution and stripped from prose.
    md = f"Here you go:\n\n```text\n{long_code}\n```\n"
    history_dir, user_id, session_id = _seed_turn(
        e2e_adapter,
        e2e_tmp,
        "copy-code",
        user_text="show code",
        agent_text=md,
    )

    with patch("tsugite.history.storage.get_history_dir", return_value=history_dir):
        _open_session(page, user_id, session_id)
        agent = page.locator(".console-turn.agent").last
        pre = agent.locator("pre").first
        pre.hover()
        pre.locator(".copy-code").click()

        page.wait_for_selector(".toast-stack .console-toast.toast-success", timeout=2000)
        clipboard = page.evaluate("navigator.clipboard.readText()")

    assert "    line_0 = 0" in clipboard
    assert "    line_39 = 39" in clipboard
    # The button's glyph must never leak into the copy.
    assert "⧉" not in clipboard
    for i in range(40):
        assert f"    line_{i} = {i}" in clipboard


def test_toast_stack_handles_rapid_fire(authenticated_page):
    page = authenticated_page

    page.evaluate("""
        window.dispatchEvent(new CustomEvent('tsugite:toast', { detail: { text: 'a', duration: 5000 } }));
        window.dispatchEvent(new CustomEvent('tsugite:toast', { detail: { text: 'b', duration: 5000 } }));
        window.dispatchEvent(new CustomEvent('tsugite:toast', { detail: { text: 'c', duration: 5000 } }));
        """)

    page.wait_for_function(
        "document.querySelectorAll('.toast-stack .console-toast').length === 3",
        timeout=2000,
    )
    texts = page.locator(".toast-stack .console-toast").evaluate_all("els => els.map(e => e.textContent)")
    assert texts == ["a", "b", "c"]


def test_toast_success_and_error_variants_have_correct_border(authenticated_page):
    page = authenticated_page

    page.evaluate("""
        window.dispatchEvent(new CustomEvent('tsugite:toast', { detail: { text: 'ok', kind: 'success', duration: 5000 } }));
        window.dispatchEvent(new CustomEvent('tsugite:toast', { detail: { text: 'bad', kind: 'error', duration: 5000 } }));
        """)
    page.wait_for_function(
        "document.querySelectorAll('.toast-stack .console-toast').length === 2",
        timeout=2000,
    )

    assert page.locator(".toast-stack .toast-success").count() == 1
    assert page.locator(".toast-stack .toast-error").count() == 1

    ok_color = page.locator(".toast-success").evaluate("el => getComputedStyle(el).borderLeftColor")
    err_color = page.locator(".toast-error").evaluate("el => getComputedStyle(el).borderLeftColor")

    # --ok / --error are defined on [data-theme], i.e. <body>.
    expected_ok = page.evaluate("getComputedStyle(document.body).getPropertyValue('--ok').trim()")
    expected_err = page.evaluate("getComputedStyle(document.body).getPropertyValue('--error').trim()")

    def _normalize(css_color):
        return page.evaluate(
            "c => { const d = document.createElement('div'); d.style.color = c; "
            "document.body.appendChild(d); const v = getComputedStyle(d).color; "
            "document.body.removeChild(d); return v; }",
            css_color,
        )

    assert _normalize(ok_color) == _normalize(expected_ok)
    assert _normalize(err_color) == _normalize(expected_err)


def test_toast_auto_dismisses_by_id_not_shift(authenticated_page):
    """If two toasts fire in the same millisecond, both must dismiss cleanly.

    Regression against a `toasts.shift()`-based implementation where the
    wrong-id shift would remove the newer toast first and strand the older one.
    """
    page = authenticated_page

    page.evaluate("""
        const ev = () => new CustomEvent('tsugite:toast', { detail: { text: 't', duration: 250 } });
        window.dispatchEvent(ev());
        window.dispatchEvent(ev());
        """)
    page.wait_for_function(
        "document.querySelectorAll('.toast-stack .console-toast').length === 2",
        timeout=2000,
    )
    page.wait_for_function(
        "document.querySelectorAll('.toast-stack .console-toast').length === 0",
        timeout=2000,
    )
