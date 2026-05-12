# Tsugite testing guide

A pragmatic map of what's covered, what isn't, and the patterns used.

## Four layers

| Layer | Where | What it runs | Speed | LLM? |
|---|---|---|---|---|
| Unit | `tests/test_*.py` | Pure Python, in-process | <1s each | No |
| Integration (mocked) | `tests/daemon/`, `tests/integration/test_concurrent_*`, `tests/integration/test_nested_*` | Real subsystems, mocked transport / monkeypatched `run_agent` | 1–5s | No |
| Integration (real LLM) | `tests/integration/test_cli_e2e.py`, `test_history_e2e.py`, `test_multistep_e2e.py`, `test_tool_execution.py`, `test_exec_directive_e2e.py` | Subprocess `tsu run` + agent runtime + real provider | 1–10s each | Yes (auto-skipped without API key) |
| E2E browser (mocked) | `tests/e2e/` | Real daemon + real Chromium (Playwright) | 5–30s each | No (mock-chat tripwire enforced) |
| E2E smoke (real LLM) | `tests/e2e_smoke/` | Real daemon + real Chromium + real provider | 10–60s each | Yes (gated by `TSUGITE_E2E_REAL_LLM=1`) |

Run targets:

```bash
uv run pytest                              # unit + mocked integration + mocked e2e
uv run pytest tests/integration/           # real-LLM integration (auto-skips without API key)
uv run pytest tests/e2e/                   # browser, mocked LLM (needs chromium installed)
TSUGITE_E2E_REAL_LLM=1 OPENAI_API_KEY=$OPENAI_API_KEY \
  uv run pytest tests/e2e_smoke/           # browser, real provider (costs tokens)
uv run pytest --cov=tsugite --cov-report=html
```

E2E setup: `uv run playwright install chromium`. Pulls Alpine.js + axe-core from a CDN, so the browser tier needs internet.

## E2E architecture

`tests/e2e/conftest.py` spins up a real daemon for the test session:

- Free port on `127.0.0.1`
- Per-session pytest tmpdir for session store, tokens, webhooks
- Workspace mocked to "not found" (no real workspace files)
- `handle_message` defaults to a **tripwire raiser** that fails any test trying to POST `/chat` without first calling `mock_chat(...)`. Real LLM calls from `tests/e2e/` are structurally impossible.
- An autouse `_reset_daemon_state` fixture clears `session_store._sessions`, `server._active_backends`, `server._active_progress`, and `server._active_chat_tasks` between tests to eliminate cross-test pollution
- uvicorn runs in a daemon thread, dies with the process

Fixtures most tests use:
- `authenticated_page` — Playwright page with auth token pre-injected
- `chat_page` — adds: conversations view open, session created and selected
- `mock_chat` — factory to configure the fake agent response + emitted events
- `e2e_session_store` — direct access to the SessionStore for seeding state
- `mobile_page` (in `test_mobile.py`) — `authenticated_page` resized to 375×800

Shared helpers (`tests/e2e/helpers.py`):
- `wait_for_alpine_ready(page)`, `open_conversations(page)`, `reload_conversations_view(page)`, `wait_for_session_in_list(page, sid)`, `select_session_in_view(page, sid)`, `open_session_by_url(page, base_url, user_id, sid)`

## Integration test isolation

`tests/integration/conftest.py` autouse fixtures:
- `_isolate_data_dirs` — points `XDG_DATA_HOME` / `XDG_CONFIG_HOME` / `XDG_CACHE_HOME` at `tmp_path` so subprocess `tsu run` can't read or write the user's real workspace, history, config, or secrets
- `_register_all_tools` — manually re-registers tools after `reset_tool_registry` (the parent conftest clears the registry between tests)

`return_value` is the canonical "end run" tool name; `final_answer` is an executor-namespace alias that still works in agent Python code but is **not** a registered tool. Don't put `"final_answer"` in `tools:` frontmatter — registry validation will reject it.

## Testable: examples

### 1. UI behavior on user input (mocked LLM)

```python
def test_send_message_shows_response(chat_page, mock_chat):
    mock_chat("I can help with that!")
    page = chat_page
    page.locator("textarea#message-input").fill("Hello agent")
    page.locator("textarea#message-input").press("Enter")
    page.wait_for_selector(".console-turn.agent", timeout=15000)
    assert "I can help with that!" in page.locator(".console-turn.agent").last.text_content()
```

### 2. Tool calls in a response

You hand-craft the events. The fake agent emits them via the same SSE path the real one uses, so the frontend can't tell the difference.

```python
def test_tool_call_progress_display(chat_page, mock_chat):
    mock_chat(
        "Found 3 files.",
        events=[
            ("tool_result", {"tool": "list_files", "output": "a\nb\nc", "success": True}),
        ],
    )
    page = chat_page
    page.locator("textarea#message-input").fill("List the files")
    page.locator("textarea#message-input").press("Enter")
    page.wait_for_selector(".console-turn.agent", timeout=15000)
    assert "list_files" in page.locator(".console-codeblock .tool-step").first.text_content()
```

Event types accepted: `tool_call`, `tool_result`, `thought`, `reasoning_content`, `code_execution`, `hook_status`, `reaction`, `final_result`, `error`, `cancelled`, etc.

### 3. Seeding history via the event model

The new session JSONL is a flat event log. Replace old `record_turn(messages=..., final_answer=...)` patterns with explicit events:

```python
storage = SessionStorage.create("test-agent", model="test", session_path=path)
storage.record("user_input", text="hi")
storage.record("model_response", provider="test", model="test", raw_content="hello back")
storage.record("code_execution", code="print('x')", output="x", duration_ms=12)
```

The frontend reconstructs bubbles from these events; ```python code fences inside `raw_content` are stripped from prose (they're tool-execution territory). Use other language tags (` ```text `, ` ```js `) when you want markdown code blocks to render.

### 4. Multi-session and multi-tab

```python
def test_two_pages_see_same_session_state(authenticated_page, e2e_session_store, base_url, e2e_auth_token):
    # First page in the existing context, second in a fresh context
    context = page1.context.browser.new_context()
    page2 = context.new_page()
    ...
```

See `test_multi_session.py` for the full pattern.

### 5. Mobile viewport behaviour

```python
@pytest.fixture
def mobile_page(authenticated_page):
    authenticated_page.set_viewport_size({"width": 375, "height": 800})
    return authenticated_page
```

At ≤640px the conversations view collapses to one pane: sidebar OR thread, controlled by `mobile-hidden` on `.console-sidebar` / `.console-main`. The `.console-thread-header .mobile-back` button clears `selectedSessionId` and re-shows the sidebar.

### 6. Accessibility ratchet

`test_a11y.py` injects axe-core (via CDN) and asserts no NEW serious/critical violations beyond a per-view baseline. Add a rule id to `BASELINE[view]` when intentionally introducing a known issue; remove from `BASELINE` when fixing.

### 7. Backend behavior without a browser

Starlette `TestClient` over the real router — see `tests/daemon/`.

### 8. Real-LLM smoke (gated)

`tests/e2e_smoke/` exercises the full provider-real path. Auto-skipped without `TSUGITE_E2E_REAL_LLM=1`. Three tests:
- Browser send-and-receive
- Direct `POST /chat` SSE stream
- Progress events reach the browser during a real turn

Assertions are deliberately loose — this tier tests *wiring*, not model behaviour.

## Not testable in the current setup

### LLM-dependent (covered by smoke tier only)

- **Provider-specific quirks** — Anthropic streaming vs OpenAI vs Claude Code vs Ollama, beyond what the smoke tier exercises against one provider
- **Token counting** against real provider counters
- **Multi-turn tool reasoning** — model reacting to tool output and deciding next step

### External services

- Real Discord bot gateway connections
- Real webhook delivery from Forgejo/GitHub
- Real ACP protocol against an ACP server
- Gmail / Calendar / Drive MCP integrations

Workaround: adapter code is unit-tested with mocked transports.

### OS / environment

- Signal handling, daemon-reload, log rotation
- File-permission edge cases beyond `tmp_path`
- Real workspace discovery on arbitrary directory structures (the XDG-isolation fixture covers the subprocess case)

### Performance / scale

- Session stores with thousands of sessions
- SSE bus under many concurrent subscribers
- Compaction on hour-long histories

### Visual regressions

- No screenshot diffing (Python sync Playwright lacks built-in `toHaveScreenshot`; revisit with a snapshot library when the UI stabilizes)

## Known issues

- **One intermittent failure under heavy xdist parallel load.** `test_chat::test_send_message_shows_response` and (less often) `test_multi_session::test_switching_sessions_preserves_per_session_messages` can flake when 16 workers compete for CPU/IO. Both pass in isolation and serial runs are 100% clean (71/0/6). The state-reset fixture clears server state between tests, but real-resource contention still affects SSE timing.
- **6 skipped tests with `@pytest.mark.skip`** — feature-removed assertions in `test_history_code_rendering.py` (XML-envelope decoding from the pre-redesign event format), and design-dependent CSS assertions in `test_markdown_rendering.py` (table alignment, table styling, wide-table horizontal scroll). Reason strings explain each.

## Gotchas

- **E2E fixtures are session-scoped.** `e2e_session_store`, `e2e_adapter`, and `e2e_server` persist across all tests in one pytest worker. The autouse `_reset_daemon_state` fixture clears the most-relevant in-memory state at the start of each test, but a still-running coroutine from a hung prior test could in principle interfere — investigate if a test starts failing only when ordered after a slow one.
- **Alpine loads from a CDN.** Auth-ready predicates must tolerate the brief window before `window.Alpine` is defined. Use `wait_for_alpine_ready(page)` from `helpers.py` rather than rolling your own.
- **`mock_chat` tripwire enforces "no real LLM in `tests/e2e/`".** The adapter's default `handle_message` raises with a clear message until a test calls `mock_chat(...)`. Don't bypass it — if you need a real provider call, write the test in `tests/e2e_smoke/`.
- **`.console-tab` matches both the IDE tab bar and the mobile menu drawer.** Always scope to `.console-tabs button.console-tab` to avoid strict-mode violations.

## Recommended additions if you want more coverage

- **Screenshot diffing** with `pytest-playwright-snapshot` or manual PIL diff against golden images — would catch CSS regressions silently passing today. Needs UI to stabilize first.
- **More smoke tests in `tests/e2e_smoke/`** — provider-comparison runs (each provider key gates its own test parametrize), tool-call execution path, multi-turn flow.
- **Per-test daemon for the chat tests** — would eliminate the 1 intermittent flake at the cost of slower runs. Track only if the flake becomes a CI signal problem.

## Test pyramid (current state)

```
            ╱─────────────────────────╲
           ╱   E2E smoke (3, gated)    ╲     ← Real provider, real browser
          ╱      paid, opt-in           ╲       Wiring smoke only
         ╱─────────────────────────────╲
        ╱   E2E (~70, mocked LLM,       ╲    ← Playwright, real daemon, mocked LLM
       ╱     all green serial)            ╲       UI behaviour + multi-session + a11y + mobile
      ╱───────────────────────────────────╲
     ╱  Integration (~200+, mocked         ╲   ← Real subsystems + 17 real-LLM CLI tests
    ╱     and 17 real-LLM, all green)       ╲     Cross-component behaviour
   ╱──────────────────────────────────────────╲
  ╱  Unit (~2300, fast, all green)             ╲   ← Pure Python, no I/O or minimal
 ╱──────────────────────────────────────────────╲    Logic correctness
```
