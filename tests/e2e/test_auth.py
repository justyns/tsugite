"""Auth gate and app-boot smoke tests.

Covers: the unauthenticated gate rendering, a valid token unlocking the app
shell, a bad token failing to grant lasting access, and every main nav view
mounting without a JS error (folds in what the old Alpine suite's
`test_page_load.py::test_tab_loads_without_errors` covered, over the new view
set).
"""

import pytest

from .helpers import auth_headers, wait_for_authed

VIEWS = [
    "chats",
    "terminals",
    "files",
    "jobs",
    "schedules",
    "usage",
    "agents",
    "skills",
    "tools",
    "webhooks",
    "secrets",
    "plugins",
]


def test_auth_gate_shown_without_token(page, base_url):
    page.goto(base_url)
    page.wait_for_selector('[data-testid="auth-gate"]', timeout=3000)
    assert page.locator('[data-testid="auth-gate"]').is_visible()


def test_valid_token_unlocks_app(page, base_url, e2e_auth_token):
    page.goto(base_url)
    page.wait_for_selector('[data-testid="auth-gate"]', timeout=3000)

    page.locator("#token-input").fill(e2e_auth_token)
    page.locator('[data-testid="token-connect"]').click()

    wait_for_authed(page)
    assert not page.locator('[data-testid="auth-gate"]').is_visible()


def test_invalid_token_does_not_grant_lasting_access(page, base_url):
    """The gate has no client-side validation - it dismisses optimistically and
    only re-gates once the first authenticated fetch 401s. There's no inline
    "invalid token" message in the rebuild (unlike the old Alpine `.auth-error`
    banner); the observable, security-relevant behavior is that a bogus token
    doesn't leave the user authenticated.
    """
    page.goto(base_url)
    page.wait_for_selector('[data-testid="auth-gate"]', timeout=3000)

    page.locator("#token-input").fill("tsu_totally_bogus_token")
    page.locator('[data-testid="token-connect"]').click()

    page.wait_for_selector('[data-testid="auth-gate"]', timeout=5000)
    assert page.locator('[data-testid="auth-gate"]').is_visible()


def test_runtime_reachable(authenticated_page, base_url, e2e_auth_token):
    """Basic connectivity smoke: a booted, authed app can reach the daemon's
    runtime info and sees the fixture's configured agent file."""
    resp = authenticated_page.request.get(f"{base_url}/api/runtime", headers=auth_headers(e2e_auth_token))
    assert resp.ok
    assert resp.json()["agent_file"]


@pytest.mark.parametrize("view", VIEWS)
def test_view_loads_without_js_errors(authenticated_page, view):
    page = authenticated_page
    errors = []
    page.on("pageerror", lambda exc: errors.append(str(exc)))

    page.locator(f'[data-testid="nav-{view}"]').click()
    page.wait_for_selector(f'[data-testid="view-{view}"]', timeout=5000)

    assert not errors, f"JS errors on {view!r} view: {errors}"
