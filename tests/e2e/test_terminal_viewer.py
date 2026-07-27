"""Terminals view: a terminal created via the API renders in the rail, its
canvas mounts, and the two-click kill (arm then confirm) flow works.

Terminal creation always spawns a real OS pty/subprocess (no fake backend
exists for this path); `sleep 30` keeps it alive for the duration of the test
without doing anything, and the fixture's PtyManager.shutdown() reaps it if
the kill flow doesn't.
"""

from playwright.sync_api import expect

from .helpers import auth_headers, open_view


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
