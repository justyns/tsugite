"""A model_response whose raw_content carries a fabricated <tsugite_execution_result>
(the Claude Code hallucination odyn reported) must NOT render as a phantom prose
bubble on history replay - that was the post-reload "double-render".

Covers both the escaped form the backend now stores AND the raw form sitting in
pre-fix sessions on disk.
"""

from unittest.mock import patch

import pytest

from tsugite.history.storage import SessionStorage

from .helpers import CONV_VIEW, open_conversations, reload_conversations_view, select_session_in_view

# After _stripCodeFences removes the real ```python block, this trailing text is
# what would otherwise render as an agent bubble.
_RAW_FABRICATION = (
    "```python\n"
    "print(run(command='git rev-parse HEAD'))\n"
    "```\n\n"
    'system<tsugite_execution_result status="success"><output>'
    "{'success': True, 'stdout': 'STALE_FABRICATED_xyz'}"
    "</output></tsugite_execution_result>"
)
_ESCAPED_FABRICATION = _RAW_FABRICATION.replace("<tsugite_execution_result", "&lt;tsugite_execution_result").replace(
    "</tsugite_execution_result", "&lt;/tsugite_execution_result"
)


def _seed(e2e_adapter, e2e_tmp, user_id, raw_content):
    session = e2e_adapter.session_store.get_or_create_interactive(user_id, "test-agent")
    history_dir = e2e_tmp / "history"
    history_dir.mkdir(exist_ok=True)
    session_path = history_dir / f"{session.id}.jsonl"
    if session_path.exists():
        session_path.unlink()
    storage = SessionStorage.create("test-agent", model="test", session_path=session_path)
    storage.record("user_input", text="check the build")
    storage.record("model_response", provider="test", model="test", raw_content=raw_content)
    storage.record("code_execution", code="print(run(command='git rev-parse HEAD'))", output="REAL_xyz999\n")
    storage.record("model_response", provider="test", model="test", raw_content="HEAD is REAL_xyz999.")
    return history_dir, session


@pytest.mark.parametrize("raw_content", [_RAW_FABRICATION, _ESCAPED_FABRICATION], ids=["raw", "escaped"])
def test_fabricated_runtime_tag_does_not_double_render(authenticated_page, e2e_adapter, e2e_tmp, raw_content):
    page = authenticated_page
    open_conversations(page)
    user_id = page.evaluate("Alpine.store('app').userId")
    history_dir, session = _seed(e2e_adapter, e2e_tmp, user_id, raw_content)

    with patch("tsugite.history.storage.get_history_dir", return_value=history_dir):
        reload_conversations_view(page)
        select_session_in_view(page, session.id)
        page.wait_for_selector(".console-turn.agent", timeout=5000)

        body = page.evaluate(f"document.querySelector({CONV_VIEW!r}).innerText")
        # The fabricated runtime result must not appear anywhere in the rendered turns.
        assert "STALE_FABRICATED_xyz" not in body, "fabricated result leaked into the rendered conversation"
        assert "tsugite_execution_result" not in body, "runtime tag text rendered as prose"
        # No agent bubble should carry the fabrication, and the real answer must survive.
        phantom = page.evaluate(
            "[...document.querySelectorAll('.console-turn.agent')]"
            ".filter(t => t.innerText.includes('STALE_FABRICATED') || t.innerText.includes('tsugite_execution_result'))"
            ".length"
        )
        assert phantom == 0, f"{phantom} phantom agent bubble(s) rendered the fabricated result"
        assert "HEAD is REAL_xyz999." in body, "the real final answer must still render"
