"""Small helpers shared across Playwright e2e tests.

Built around the Svelte rebuild's frozen test contract (see
`frontend/src/lib/testids.ts`): app readiness is a single always-rendered
`[data-testid="app-ready"]` marker (present whether or not auth is gated),
the authenticated app shell is signalled by the nav rail mounting, and views
are switched through the real nav rail (which drives the hash router) rather
than any store/global reached into from the test. There are no Alpine-style
store hooks left to poke at - tests either drive real DOM interactions or
seed state through the daemon fixtures / HTTP API.
"""

from __future__ import annotations

# Fixed synthetic user id injected into localStorage by the `authenticated_page`
# fixture, so tests that seed sessions/data for "the current user" via
# e2e_session_store etc. have a stable id to key off instead of reading
# whatever default the frontend happens to fall back to.
E2E_USER_ID = "e2e-user"

APP_READY = '[data-testid="app-ready"]'
NAV_RAIL = '[data-testid="nav-rail"]'
AUTH_GATE = '[data-testid="auth-gate"]'


def wait_for_app_ready(page, timeout: int = 10000) -> None:
    """Wait for the Svelte app to mount.

    This marker renders unconditionally on first paint, regardless of auth
    state - it only means "the app booted", not "the user is signed in".
    """
    page.wait_for_selector(APP_READY, state="attached", timeout=timeout)


def wait_for_authed(page, timeout: int = 10000) -> None:
    """Wait for the app shell to render past the auth gate.

    The nav rail only exists in the DOM once `auth.gated` is false, so its
    presence is a direct signal the token was accepted and the shell mounted.
    """
    wait_for_app_ready(page, timeout=timeout)
    page.wait_for_selector(NAV_RAIL, timeout=timeout)


def open_view(page, view_id: str, timeout: int = 5000) -> None:
    """Click a nav rail entry and wait for that view's surface to mount."""
    page.locator(f'[data-testid="nav-{view_id}"]').click()
    page.wait_for_selector(f'[data-testid="view-{view_id}"]', timeout=timeout)


def auth_headers(token: str) -> dict:
    """Bearer-auth header dict for direct `page.request` API calls."""
    return {"Authorization": f"Bearer {token}"}
