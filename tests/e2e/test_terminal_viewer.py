"""Terminals view: a terminal created via the API renders in the rail, its
canvas mounts, and the two-click kill (arm then confirm) flow works.

Terminal creation always spawns a real OS pty/subprocess (no fake backend
exists for this path); `sleep 30` keeps it alive for the duration of the test
without doing anything, and the fixture's PtyManager.shutdown() reaps it if
the kill flow doesn't.
"""

import re

from playwright.sync_api import expect

from .helpers import auth_headers, open_view, wait_for_authed


def test_terminal_create_canvas_and_kill(authenticated_page, base_url, e2e_auth_token, terminal_backend):
    page = authenticated_page

    resp = page.request.post(
        f"{base_url}/api/terminals",
        headers=auth_headers(e2e_auth_token),
        data={"cmd": "sleep 30"},
    )
    assert resp.ok, resp.text()
    terminal_id = resp.json()["id"]

    open_view(page, "terminals")

    row = page.get_by_role("option", name="sleep 30")
    expect(row).to_be_visible()
    row.click()

    canvas = page.get_by_role("log", name="Terminal canvas — keystrokes go to the pty")
    expect(canvas).to_be_attached(timeout=5000)

    kill_button = page.get_by_role("button", name="Kill", exact=True)
    expect(kill_button).to_be_visible()
    kill_button.click()

    confirm_button = page.get_by_role("button", name="confirm kill?")
    expect(confirm_button).to_be_visible(timeout=1000)
    confirm_button.click()

    # No longer live once killed - the header swaps Kill for Restart.
    expect(page.get_by_role("button", name="Restart")).to_be_visible(timeout=5000)

    get_resp = page.request.get(f"{base_url}/api/terminals/{terminal_id}", headers=auth_headers(e2e_auth_token))
    assert get_resp.ok
    assert get_resp.json()["state"] != "running"


SIZE_REPORTER = 'trap "stty size" WINCH; stty size; while :; do sleep 0.1; done'
RUNNING_REPORTER = re.compile(r"stty size.*running")
CANVAS_NAME = "Terminal canvas — keystrokes go to the pty"
SIZE_LINE = re.compile(r"^(\d+) (\d+)\s*$", re.M)


def _open_reporter(page):
    open_view(page, "terminals")
    page.get_by_role("option", name=RUNNING_REPORTER).click()
    return page.get_by_role("log", name=CANVAS_NAME)


def _settled_size(page, canvas, seen):
    """The newest (cols, rows) the child printed that is not in `seen`, once it
    matches the row count xterm is rendering."""
    for _ in range(50):
        pairs = [(int(c), int(r)) for r, c in SIZE_LINE.findall(canvas.locator(".xterm-rows").inner_text())]
        if pairs and pairs[-1] not in seen and canvas.locator(".xterm-rows > div").count() == pairs[-1][1]:
            return pairs[-1]
        page.wait_for_timeout(100)
    raise AssertionError(f"child never settled on a size beyond {seen}: {pairs}")


def test_the_pty_follows_the_canvas_size(authenticated_page, base_url, e2e_auth_token, terminal_backend):
    """The canvas sizes the pty to its fitted geometry on open and again on a
    window resize, and the child sees each size through SIGWINCH."""
    page = authenticated_page
    page.set_viewport_size({"width": 1400, "height": 900})
    resp = page.request.post(
        f"{base_url}/api/terminals", headers=auth_headers(e2e_auth_token), data={"cmd": SIZE_REPORTER}
    )
    assert resp.ok, resp.text()

    canvas = _open_reporter(page)
    expect(canvas.locator(".xterm-rows")).to_contain_text("24 80")

    cols, rows = _settled_size(page, canvas, {(80, 24)})
    assert cols > 80

    page.set_viewport_size({"width": 900, "height": 600})
    small_cols, small_rows = _settled_size(page, canvas, {(80, 24), (cols, rows)})
    assert small_cols < cols and small_rows < rows


def test_focus_picks_which_viewer_sizes_the_pty(authenticated_page, base_url, e2e_auth_token, terminal_backend):
    """With one terminal open in two windows of different sizes, the pty takes
    the size of the window whose canvas was focused last."""
    wide = authenticated_page
    wide.set_viewport_size({"width": 1400, "height": 900})
    resp = wide.request.post(
        f"{base_url}/api/terminals", headers=auth_headers(e2e_auth_token), data={"cmd": SIZE_REPORTER}
    )
    assert resp.ok, resp.text()
    wide_canvas = _open_reporter(wide)
    wide_size = _settled_size(wide, wide_canvas, {(80, 24)})

    narrow = wide.context.new_page()
    narrow.set_viewport_size({"width": 900, "height": 600})
    narrow.goto(base_url)
    wait_for_authed(narrow)
    narrow_canvas = _open_reporter(narrow)
    narrow_size = _settled_size(narrow, narrow_canvas, {(80, 24), wide_size})
    assert narrow_size < wide_size

    wide_canvas.click()
    assert _settled_size(wide, wide_canvas, {(80, 24), narrow_size}) == wide_size

    narrow_canvas.click()
    assert _settled_size(narrow, narrow_canvas, {(80, 24), wide_size}) == narrow_size
