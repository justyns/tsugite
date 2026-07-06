"""Worker-session replay: opening a job worker session must show the actual
code, tool arguments, tool output, and inter-turn thought text - not just
markers that they ran.

Worker sessions record BOTH the UI-handler event family (code / tool_call /
tool_result / thought, from the daemon's LoggingProgressHandler) and the agent
recording family (code_execution / model_response). The replay builder only
understood the latter, so:
- per-tool arguments (tool_call) never rendered,
- thought text (claude_code turns deliver prose via `thought`, with empty
  model_response raw_content) never rendered,
- interrupted turns - which record `code`/`tool_result` but never reach the
  post-execution `code_execution` recording - lost code and output entirely.
Event shapes/order verified against a live instance's history.db."""

from unittest.mock import patch

from tsugite_daemon.session_store import Session, SessionSource

from tsugite.history.storage import SessionStorage

from .helpers import open_session_by_url

WORKER_CODE = 'readme = read_file(path="kb/README.md")\nprint(readme[:100])'
TOOL_ARGS = {"path": "kb/README.md", "with_meta": True}
TOOL_OUTPUT = "# kb - typed reference knowledge base"


def _storage_for(e2e_session_store, e2e_tmp, user_id, sid):
    e2e_session_store.create_session(
        Session(
            id=sid,
            agent="test-agent",
            source=SessionSource.SPAWNED.value,
            user_id=user_id,
            title=f"{sid} · worker",
        )
    )
    history_dir = e2e_tmp / "history"
    history_dir.mkdir(exist_ok=True)
    path = history_dir / f"{sid}.jsonl"
    if path.exists():
        path.unlink()
    storage = SessionStorage.create("test-agent", model="claude_code:sonnet", session_path=path)
    return history_dir, storage


def _seed_complete_worker(e2e_session_store, e2e_tmp, user_id):
    """Full turn: code -> tool_call -> code_execution -> tool_result (dup output),
    then a thought-only turn, then final_result."""
    history_dir, storage = _storage_for(e2e_session_store, e2e_tmp, user_id, "session-e2eworkerfull")
    storage.record("user_input", text="Create the KB page for threads")
    storage.record("turn_start", turn=1)
    storage.record(
        "model_response",
        provider="claude_code",
        model="claude-sonnet-5",
        turn=0,
        raw_content="Reading the KB layout first.\n\n```python\n" + WORKER_CODE + "\n```",
    )
    storage.record("code", content=WORKER_CODE)
    storage.record("tool_call", tool="read_file", arguments=TOOL_ARGS)
    storage.record("tool_result_audit", tool="read_file", success=True, duration_ms=1, summary="kb/README.md")
    storage.record("code_execution", code=WORKER_CODE, output=TOOL_OUTPUT, duration_ms=812, tools_called=["read_file"])
    storage.record("tool_result", tool="unknown", success=True, output=TOOL_OUTPUT)
    storage.record("turn_start", turn=2)
    storage.record("model_response", provider="claude_code", model="claude-sonnet-5", turn=1, raw_content="")
    storage.record("thought", content="The layout is clear; writing the page now.")
    storage.record("final_result", result="## Summary\n\nCreated the page.", turns=2)
    storage.record("session_end", status="success")
    return history_dir, "session-e2eworkerfull"


def _seed_interrupted_worker(e2e_session_store, e2e_tmp, user_id):
    """Interrupted turn: the UI-handler events landed but the turn never reached
    the post-execution code_execution recording (cancelled/failed mid-execution).
    The terminal session_end event matches what the runner's cancel/fail paths
    append; without ANY terminal event the replay treats the session as mid-turn
    and defers the trailing bubble to live rehydration (separate issue)."""
    history_dir, storage = _storage_for(e2e_session_store, e2e_tmp, user_id, "session-e2eworkerint")
    storage.record("user_input", text="Create the KB page for threads")
    storage.record("turn_start", turn=1)
    storage.record("code", content=WORKER_CODE)
    storage.record("tool_call", tool="read_file", arguments=TOOL_ARGS)
    storage.record("tool_result", tool="unknown", success=True, output=TOOL_OUTPUT)
    storage.record("session_end", status="error")
    return history_dir, "session-e2eworkerint"


def _bubbles(page):
    return page.evaluate(
        """() => {
            const v = Alpine.$data(document.querySelector('[x-data*=conversationsView]'));
            return (v.messages || []).map(m => ({
                type: m.type,
                text: m.text || '',
                steps: (m.steps || []).map(s => ({
                    summary: s.summary || s.html || '',
                    content: s.content || '',
                })),
            }));
        }"""
    )


def _all_steps(bubbles):
    return [s for b in bubbles for s in b["steps"]]


def test_complete_worker_turn_renders_tools_and_thought(authenticated_page, e2e_session_store, e2e_tmp):
    page = authenticated_page
    user_id = page.evaluate("Alpine.store('app').userId")
    history_dir, sid = _seed_complete_worker(e2e_session_store, e2e_tmp, user_id)

    with patch("tsugite.history.storage.get_history_dir", return_value=history_dir):
        open_session_by_url(page, page.url.split("#")[0], user_id, sid)
        page.wait_for_selector(".console-turn", timeout=5000)
        page.screenshot(path="/tmp/tsugite-issue-426-state.png", full_page=True)

        bubbles = _bubbles(page)
        steps = _all_steps(bubbles)

        code_steps = [s for s in steps if WORKER_CODE in s["content"]]
        assert len(code_steps) == 1, f"expected exactly one code step (no dup from `code` event); got {len(code_steps)}"

        tool_steps = [s for s in steps if "read_file" in s["summary"]]
        assert tool_steps, f"tool_call must render a read_file step with arguments; steps: {[s['summary'] for s in steps]}"
        assert "kb/README.md" in tool_steps[0]["content"], "tool arguments must be shown"

        output_steps = [s for s in steps if TOOL_OUTPUT in s["content"]]
        assert len(output_steps) == 1, f"output must render exactly once (tool_result dup suppressed); got {len(output_steps)}"

        agent_texts = " | ".join(b["text"] for b in bubbles if b["type"] == "agent")
        assert "writing the page now" in agent_texts, f"thought text must render as agent prose; got: {agent_texts!r}"


def test_interrupted_worker_turn_still_shows_code_and_output(authenticated_page, e2e_session_store, e2e_tmp):
    page = authenticated_page
    user_id = page.evaluate("Alpine.store('app').userId")
    history_dir, sid = _seed_interrupted_worker(e2e_session_store, e2e_tmp, user_id)

    with patch("tsugite.history.storage.get_history_dir", return_value=history_dir):
        open_session_by_url(page, page.url.split("#")[0], user_id, sid)
        page.wait_for_selector(".console-turn", timeout=5000)

        steps = _all_steps(_bubbles(page))
        assert any(WORKER_CODE in s["content"] for s in steps), (
            "an interrupted turn (code event, no code_execution) must still show the code that ran"
        )
        assert any(TOOL_OUTPUT in s["content"] for s in steps), (
            "an interrupted turn's tool_result output must still render"
        )
