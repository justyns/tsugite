"""Small helpers shared across Playwright e2e tests.

Patterns extracted from copy-pasted snippets across the suite so future tests
don't have to know the brittle predicates (e.g. tolerating the brief window
before `window.Alpine` is defined, scoping `.console-tab` to the IDE tab bar
to avoid the mobile-menu duplicate).
"""

from __future__ import annotations

from typing import Optional

ALPINE_READY = "typeof Alpine !== 'undefined' && Alpine.store('app') && !Alpine.store('app').authRequired"
CONV_VIEW = "[x-data*=conversationsView]"
# The view's TRUE reactive context (registered by the component's init). Writes
# through Alpine.$data(...) wrappers can miss Alpine's dependency graph for
# nested state, so helpers must drive the view through this instead.
CONV_REF = "window.__tsugiteConv"


def wait_for_alpine_ready(page, timeout: int = 10000) -> None:
    """Wait for Alpine to load and auth to no longer be required.

    Tolerates the brief window before window.Alpine is defined (Alpine loads
    as an ES module from a CDN).
    """
    page.wait_for_function(ALPINE_READY, timeout=timeout)


def open_conversations(page) -> None:
    """Click the Conversations tab and wait for the view to switch.

    Scoped to `.console-tabs button.console-tab` to avoid matching the mobile
    menu's duplicate button.
    """
    page.locator(".console-tabs button.console-tab", has_text="Conversations").click()
    page.wait_for_function("Alpine.store('app').view === 'conversations'", timeout=3000)


def reload_conversations_view(page) -> None:
    """Force the conversationsView to reload its session list.

    Needed when a test added sessions to the store after page mount; the
    initial /api/agents/{agent}/sessions fetch is stale.
    """
    page.evaluate(f"{CONV_REF}.reload()")


def wait_for_session_in_list(page, session_id: str, timeout: int = 5000) -> None:
    """Wait until the conversationsView's allSessions array includes session_id."""
    page.wait_for_function(
        f"(() => {{ const v = {CONV_REF}; "
        f"return v && v.allSessions && v.allSessions.some(s => s.id === {session_id!r}); }})()",
        timeout=timeout,
    )


def select_session_in_view(page, session_id: str, timeout: int = 3000) -> None:
    """Programmatically select a session and wait for it to take effect."""
    page.evaluate(f"{CONV_REF}.selectSessionById({session_id!r}, {{follow: false}})")
    page.wait_for_function(
        f"{CONV_REF}.selectedSessionId === {session_id!r}",
        timeout=timeout,
    )


def open_session_by_url(page, base_url: str, user_id: Optional[str], session_id: str) -> None:
    """Navigate to a session via the URL hash; reload first to ensure fresh state.

    If user_id is given, set it in localStorage before navigation so the page
    loads as that user. Used by history-seeding tests where the seeded session
    is owned by a synthetic user.
    """
    if user_id is not None:
        page.evaluate(f"localStorage.setItem('tsugite_user_id', {user_id!r})")
    page.goto(page.url.split("#")[0] + f"#conversations?session={session_id}")
    page.reload()
    wait_for_alpine_ready(page, timeout=5000)
    if user_id is not None:
        page.wait_for_function(f"Alpine.store('app').userId === {user_id!r}", timeout=3000)
