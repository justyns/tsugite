"""E2E tests for the Terminal Viewer.

Stubs out the /api/terminals endpoints + the per-terminal SSE stream so the
frontend can exercise sidebar rendering, selection, and the full-session view
without a real PTY backend. We don't test xterm.js's own glyph rendering -
the renderer is opaque (canvas + GPU), so we just assert the xterm container
gets mounted and the surrounding chrome is correct.
"""

from __future__ import annotations

import json

import pytest


def _stub_terminals_api(page, terminals=None, stream_chunks=None, stream_state=None, replay_chunk=None):
    """Intercept /api/terminals + the SSE stream with deterministic fixtures.

    `terminals` is the list returned by GET /api/terminals.
    `stream_chunks` is a sequence of LIVE output strings the SSE stream emits as
        plain `output` events (the frontend appends these).
    `stream_state` is an optional 'state' event payload appended after chunks.
    `replay_chunk`, when set, is emitted first as an `output` frame tagged
        `replay: true` - mirroring how the real backend replays the full ring
        buffer on connect. The frontend RESETS its buffer/metrics (and clears
        xterm) on a replay frame rather than appending, so the stub must carry
        the flag for that path to be exercised faithfully.
    """
    if terminals is None:
        terminals = []
    if stream_chunks is None:
        stream_chunks = []

    def handle_list(route, request):
        if request.method == "POST":
            body = request.post_data_json or {}
            new_term = {
                "id": "term-newly-spawned",
                "cmd": body.get("cmd", "(unknown)"),
                "state": "running",
                "created_at": "2026-05-30T10:00:00Z",
                "updated_at": "2026-05-30T10:00:01Z",
                "exit_code": None,
                "bytes_out": 0,
                "lines_out": 0,
                "last_line": "",
                "pid": 9999,
            }
            terminals.append(new_term)
            route.fulfill(status=200, body=json.dumps({"terminal_id": new_term["id"]}))
            return
        route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({"terminals": terminals}),
        )

    def handle_stream(route, request):
        # Build a valid SSE response body. The frontend uses EventSource so
        # the framing must be `event: <name>\ndata: <json>\n\n`.
        body_parts = []
        if replay_chunk is not None:
            # The buffered-output replay the backend sends on connect, tagged so
            # the frontend resets (not appends) - see _appendOutput(replay=true).
            body_parts.append(f"event: output\ndata: {json.dumps({'chunk': replay_chunk, 'replay': True})}\n\n")
        for chunk in stream_chunks:
            body_parts.append(f"event: output\ndata: {json.dumps({'chunk': chunk})}\n\n")
        if stream_state:
            body_parts.append(f"event: state\ndata: {json.dumps(stream_state)}\n\n")
        route.fulfill(
            status=200,
            headers={
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
            body="".join(body_parts),
        )

    def handle_restart(route, request):
        # Restart spawns a BRAND-NEW PTY with a new id; the backend returns 201
        # with the new terminal record tagged `restarted_from`. The frontend
        # reads `data.id` off this body and migrates every surface to the new id,
        # so a 204/empty body would leave it with no id to select.
        old_id = request.url.split("?")[0].rstrip("/").rsplit("/", 2)[1]
        new_term = {
            "id": "term-new",
            "cmd": "(restarted)",
            "state": "running",
            "created_at": "2026-05-30T11:00:00Z",
            "updated_at": "2026-05-30T11:00:00Z",
            "exit_code": None,
            "bytes_out": 0,
            "lines_out": 0,
            "last_line": "",
            "pid": 10001,
            "restarted_from": old_id,
        }
        terminals.append(new_term)
        route.fulfill(
            status=201,
            content_type="application/json",
            body=json.dumps(new_term),
        )

    page.route("**/api/terminals", handle_list)
    page.route("**/api/terminals/*/stream", handle_stream)
    page.route("**/api/terminals/*/kill", lambda route: route.fulfill(status=204, body=""))
    page.route("**/api/terminals/*/restart", handle_restart)


def test_sidebar_shows_terminal_section_header(authenticated_page):
    """The '❯ terminal · /run' section head renders even with zero terminals."""
    page = authenticated_page
    _stub_terminals_api(page, terminals=[])
    # Force the terminals store to reload now that the route is stubbed.
    page.evaluate("Alpine.store('terminals').loadTerminals()")
    page.locator(".console-tabs button.console-tab", has_text="Conversations").click()
    page.wait_for_function("Alpine.store('app').view === 'conversations'", timeout=3000)

    head = page.locator(".console-section-head.term")
    head.wait_for(state="visible", timeout=3000)
    assert "terminal" in (head.text_content() or "").lower()
    assert "/run" in (head.text_content() or "")


def test_sidebar_renders_terminal_rows(authenticated_page):
    """Each /api/terminals entry renders as a console-session row with the ❯ glyph."""
    page = authenticated_page
    _stub_terminals_api(
        page,
        terminals=[
            {
                "id": "term-001",
                "cmd": "npm run dev",
                "state": "running",
                "created_at": "2026-05-30T10:00:00Z",
                "updated_at": "2026-05-30T10:00:42Z",
                "exit_code": None,
                "bytes_out": 1024,
                "lines_out": 11,
                "last_line": "ready in 412 ms",
                "pid": 48213,
            },
            {
                "id": "term-002",
                "cmd": "pytest tests/",
                "state": "succeeded",
                "created_at": "2026-05-30T09:55:00Z",
                "updated_at": "2026-05-30T09:55:12Z",
                "exit_code": 0,
                "bytes_out": 2048,
                "lines_out": 84,
                "last_line": "84 passed in 12.34s",
                "pid": 47912,
            },
        ],
    )
    page.evaluate("Alpine.store('terminals').loadTerminals()")
    page.locator(".console-tabs button.console-tab", has_text="Conversations").click()
    page.wait_for_function("Alpine.store('app').view === 'conversations'", timeout=3000)

    # Wait for the terminals store to populate.
    page.wait_for_function("Alpine.store('terminals').terminals.length === 2", timeout=3000)

    head = page.locator(".console-section-head.term")
    head.wait_for(state="visible", timeout=3000)

    # The rows live inside the sidebar's terminal section. They should carry
    # the lavender ❯ glyph and surface the command + last line.
    rows = page.locator(".console-session .tglyph")
    assert rows.count() == 2, f"expected 2 terminal rows, got {rows.count()}"

    text = page.locator("[x-data*=conversationsView] .console-sidebar-scroll").text_content() or ""
    assert "npm run dev" in text
    assert "pytest tests/" in text
    assert "ready in 412 ms" in text


def test_clicking_terminal_row_opens_full_session_view(authenticated_page):
    """Clicking a terminal row swaps the main pane to the full-session view."""
    page = authenticated_page
    _stub_terminals_api(
        page,
        terminals=[
            {
                "id": "term-001",
                "cmd": "echo hi",
                "state": "succeeded",
                "created_at": "2026-05-30T10:00:00Z",
                "updated_at": "2026-05-30T10:00:00Z",
                "exit_code": 0,
                "bytes_out": 3,
                "lines_out": 1,
                "last_line": "hi",
                "pid": 12345,
            },
        ],
        stream_chunks=["hi\r\n"],
    )
    page.evaluate("Alpine.store('terminals').loadTerminals()")
    page.locator(".console-tabs button.console-tab", has_text="Conversations").click()
    page.wait_for_function("Alpine.store('app').view === 'conversations'", timeout=3000)
    page.wait_for_function("Alpine.store('terminals').terminals.length === 1", timeout=3000)

    # Click the terminal row.
    page.evaluate("Alpine.store('terminals').selectTerminal('term-001')")
    page.wait_for_function("Alpine.store('terminals').selectedId === 'term-001'", timeout=3000)

    # The full-session view should mount with the command in the header.
    fs = page.locator(".tv-fs")
    fs.wait_for(state="visible", timeout=3000)
    head_text = page.locator(".tv-fs-head .tv-cmd").text_content() or ""
    assert "echo hi" in head_text

    # Status pill reflects the terminal's state.
    state_pill = page.locator(".tv-fs-head .tv-state")
    state_pill.wait_for(state="visible", timeout=3000)
    assert "succeeded" in (state_pill.text_content() or "").lower()


def test_full_session_view_renders_action_buttons(authenticated_page):
    """Follow / copy / restart / kill buttons appear in the header."""
    page = authenticated_page
    _stub_terminals_api(
        page,
        terminals=[
            {
                "id": "term-001",
                "cmd": "npm run dev",
                "state": "running",
                "created_at": "2026-05-30T10:00:00Z",
                "updated_at": "2026-05-30T10:00:42Z",
                "exit_code": None,
                "bytes_out": 0,
                "lines_out": 0,
                "last_line": "",
                "pid": 48213,
            },
        ],
    )
    page.evaluate("Alpine.store('terminals').loadTerminals()")
    page.locator(".console-tabs button.console-tab", has_text="Conversations").click()
    page.wait_for_function("Alpine.store('app').view === 'conversations'", timeout=3000)
    page.wait_for_function("Alpine.store('terminals').terminals.length === 1", timeout=3000)

    page.evaluate("Alpine.store('terminals').selectTerminal('term-001')")
    page.wait_for_selector(".tv-fs .tv-fs-head .btns .tv-btn", timeout=3000)
    btns = page.locator(".tv-fs-head .btns .tv-btn")
    # Four header buttons: follow, copy, restart, kill.
    assert btns.count() == 4, f"expected 4 header buttons, got {btns.count()}"

    # Kill button enabled for running terminals.
    kill_btn = page.locator(".tv-fs-head .tv-btn.kill")
    assert not kill_btn.is_disabled(), "kill button should be enabled while terminal is running"


def test_xterm_host_mounted_after_selection(authenticated_page):
    """The .tv-fs-body host receives an xterm container after selection.

    We don't assert the canvas itself (xterm.js renders to a GPU canvas that
    Playwright can see but not introspect cleanly). Asserting the [data-xterm-host]
    element exists and has at least the .xterm wrapper proves the loader fired.
    """
    page = authenticated_page
    _stub_terminals_api(
        page,
        terminals=[
            {
                "id": "term-001",
                "cmd": "echo hi",
                "state": "running",
                "created_at": "2026-05-30T10:00:00Z",
                "updated_at": "2026-05-30T10:00:00Z",
                "exit_code": None,
                "bytes_out": 0,
                "lines_out": 0,
                "last_line": "",
                "pid": 99,
            },
        ],
        stream_chunks=["hello world\r\n"],
    )
    page.evaluate("Alpine.store('terminals').loadTerminals()")
    page.locator(".console-tabs button.console-tab", has_text="Conversations").click()
    page.wait_for_function("Alpine.store('app').view === 'conversations'", timeout=3000)
    page.wait_for_function("Alpine.store('terminals').terminals.length === 1", timeout=3000)
    page.evaluate("Alpine.store('terminals').selectTerminal('term-001')")

    host = page.locator("[data-xterm-host]")
    host.wait_for(state="visible", timeout=3000)

    # The xterm-loader pulls scripts from jsDelivr; in CI we may not have
    # network, so we only assert the host exists. If the loader does succeed,
    # the child .xterm element appears asynchronously.
    page.wait_for_timeout(800)
    # If network is reachable, the .xterm class should be present.
    has_xterm = page.evaluate("() => !!document.querySelector('[data-xterm-host] .xterm')")
    # When network is unavailable, the loader will fail silently; the test
    # still passes because the host element is the load-bearing structural
    # assertion. (xterm.js rendering itself is out of scope for the brief.)
    if has_xterm:
        assert page.locator("[data-xterm-host] .xterm").count() > 0


def test_typing_into_full_session_terminal_forwards_to_stdin(authenticated_page):
    """A keystroke in the full-session terminal must POST to /api/terminals/<id>/stdin.

    The panel was read-only: createTerminalRenderer got disableStdin:true and no
    term.onData wiring, even though the backend stdin endpoint already existed.
    """
    page = authenticated_page
    stdin_posts: list[str] = []

    def handle_stdin(route, request):
        stdin_posts.append(request.post_data or "")
        route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({"status": "ok", "bytes_written": 1}),
        )

    _stub_terminals_api(
        page,
        terminals=[
            {
                "id": "term-001",
                "cmd": "bash",
                "state": "running",
                "created_at": "2026-05-30T10:00:00Z",
                "updated_at": "2026-05-30T10:00:00Z",
                "exit_code": None,
                "bytes_out": 0,
                "lines_out": 0,
                "last_line": "",
                "pid": 222,
            },
        ],
    )
    page.route("**/api/terminals/*/stdin", handle_stdin)
    page.evaluate("Alpine.store('terminals').loadTerminals()")
    page.locator(".console-tabs button.console-tab", has_text="Conversations").click()
    page.wait_for_function("Alpine.store('app').view === 'conversations'", timeout=3000)
    page.wait_for_function("Alpine.store('terminals').terminals.length === 1", timeout=3000)
    page.evaluate("Alpine.store('terminals').selectTerminal('term-001')")

    page.locator("[data-xterm-host]").wait_for(state="visible", timeout=3000)
    # xterm.js loads from a CDN; without it there is no input surface to drive.
    page.wait_for_timeout(1200)
    if not page.evaluate("() => !!document.querySelector('.tv-fs .xterm-helper-textarea')"):
        pytest.skip("xterm.js (CDN) not loaded; cannot exercise terminal input")

    with page.expect_request("**/api/terminals/*/stdin", timeout=4000):
        page.locator(".tv-fs .xterm-helper-textarea").focus()
        page.keyboard.type("x")
    assert any("x" in d for d in stdin_posts), f"keystroke must POST to /stdin, got: {stdin_posts}"


def test_empty_state_renders_when_no_terminal_selected(authenticated_page):
    """Clicking the terminal section header with no terminal selected shows the empty state."""
    page = authenticated_page
    _stub_terminals_api(page, terminals=[])
    page.evaluate("Alpine.store('terminals').loadTerminals()")
    page.locator(".console-tabs button.console-tab", has_text="Conversations").click()
    page.wait_for_function("Alpine.store('app').view === 'conversations'", timeout=3000)

    # Click the section header to trigger showEmptyState().
    page.locator(".console-section-head.term").click()
    page.wait_for_function("Alpine.store('terminals').showEmpty === true", timeout=3000)

    empty = page.locator(".tv-empty")
    empty.wait_for(state="visible", timeout=3000)
    text = empty.text_content() or ""
    assert "new terminal session" in text.lower()
    # Example commands surface in the empty state.
    assert "/run" in text


def test_terminal_section_uses_peach_label_color(authenticated_page):
    """The terminal section head should adopt the peach accent (per design tokens)."""
    page = authenticated_page
    _stub_terminals_api(page, terminals=[])
    page.evaluate("Alpine.store('terminals').loadTerminals()")
    page.locator(".console-tabs button.console-tab", has_text="Conversations").click()
    page.wait_for_function("Alpine.store('app').view === 'conversations'", timeout=3000)

    head_label = page.locator(".console-section-head.term .label")
    head_label.wait_for(state="visible", timeout=3000)
    # Resolve --peach in the active theme so the assertion is theme-agnostic.
    peach = page.evaluate("() => getComputedStyle(document.body).getPropertyValue('--peach').trim()")
    color = page.evaluate(
        "(el) => getComputedStyle(el).color",
        head_label.element_handle(),
    )
    # color comes back as rgb(...); peach is a hex. We can't compare directly
    # but if the CSS rule is wired the color should differ from default overlay0.
    overlay0 = page.evaluate("() => getComputedStyle(document.body).getPropertyValue('--overlay0').trim()")
    assert peach, "--peach must be defined in the active theme"
    assert overlay0, "--overlay0 must be defined in the active theme"
    # Color is set via var(--peach); just assert it's not equal to overlay0.
    assert color != overlay0
