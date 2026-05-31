"""End-to-end tests for the shared tsu-modal shell.

These tests cover the *shell* itself, independent of any single migrated
modal: store-driven open/close, Escape + backdrop dismissal, non-dismissable
modals, stacking semantics, and that the shell paints opacity:1 across all
four Catppuccin themes (no x-transition got-stuck regressions).
"""

from __future__ import annotations

import pytest

from .helpers import wait_for_alpine_ready


THEMES = ["frappe", "latte", "macchiato", "mocha"]


def _open_settings(page) -> None:
    """Open settings — the canonical migrated modal that ships with the shell."""
    page.evaluate("Alpine.store('tsu').open('settings')")
    page.wait_for_selector(".tsu-modal-backdrop.settings-modal", state="visible", timeout=3000)


def _authenticated(page, base_url, token):
    page.goto(base_url + "/api/health")
    page.evaluate(f"localStorage.setItem('tsugite_token', '{token}')")
    page.goto(base_url)
    wait_for_alpine_ready(page)
    return page


def test_modal_opens_via_store_and_closes_via_store(page, base_url, e2e_auth_token):
    page = _authenticated(page, base_url, e2e_auth_token)
    _open_settings(page)

    # Store reflects the open modal.
    assert page.evaluate("Alpine.store('tsu').isOpen('settings')") is True
    assert page.evaluate("Alpine.store('tsu').top") == "settings"

    page.evaluate("Alpine.store('tsu').close('settings')")
    page.wait_for_function(
        "() => { const el = document.querySelector('.tsu-modal-backdrop.settings-modal');"
        " return !el || el.style.display === 'none'; }",
        timeout=2000,
    )
    assert page.evaluate("Alpine.store('tsu').isOpen('settings')") is False


def test_escape_closes_dismissable_modal(page, base_url, e2e_auth_token):
    page = _authenticated(page, base_url, e2e_auth_token)
    _open_settings(page)

    # Esc on window — the modal's @keydown.escape.window listener should fire.
    page.keyboard.press("Escape")
    page.wait_for_function(
        "() => !Alpine.store('tsu').isOpen('settings')",
        timeout=2000,
    )


def test_backdrop_click_closes_dismissable_modal(page, base_url, e2e_auth_token):
    page = _authenticated(page, base_url, e2e_auth_token)
    _open_settings(page)

    # Click the backdrop at a corner so we miss the centered panel.
    backdrop = page.locator(".tsu-modal-backdrop.settings-modal")
    box = backdrop.bounding_box()
    assert box is not None
    page.mouse.click(box["x"] + 10, box["y"] + 10)

    page.wait_for_function(
        "() => !Alpine.store('tsu').isOpen('settings')",
        timeout=2000,
    )


def test_non_dismissable_modal_ignores_escape_and_backdrop(page, base_url, e2e_auth_token):
    """A modal opened with { dismissable: false } stays open after Esc + backdrop click."""
    page = _authenticated(page, base_url, e2e_auth_token)

    # Inject a non-dismissable modal that uses the shell directly.
    page.evaluate(
        """
        () => {
          const tpl = `
            <div id="t-forced" x-data="tsuModal('forced', { dismissable: false })"
                 x-show="open" x-cloak class="tsu-modal-backdrop forced-modal"
                 @keydown.escape.window="onEscape()" @click.self="onBackdrop()">
              <div class="tsu-modal --sm" role="dialog" aria-modal="true"
                   x-trap.noscroll="open">
                <header class="tsu-modal-head"><h2 class="tsu-modal-title">forced</h2></header>
                <div class="tsu-modal-body"><p>cannot dismiss</p></div>
              </div>
            </div>`;
          document.body.insertAdjacentHTML('beforeend', tpl);
          Alpine.initTree(document.getElementById('t-forced'));
          Alpine.store('tsu').open('forced');
        }
        """
    )
    page.wait_for_selector(".tsu-modal-backdrop.forced-modal", state="visible", timeout=3000)

    page.keyboard.press("Escape")
    page.wait_for_timeout(300)
    assert page.evaluate("Alpine.store('tsu').isOpen('forced')") is True

    box = page.locator(".tsu-modal-backdrop.forced-modal").bounding_box()
    assert box is not None
    page.mouse.click(box["x"] + 10, box["y"] + 10)
    page.wait_for_timeout(300)
    assert page.evaluate("Alpine.store('tsu').isOpen('forced')") is True

    page.evaluate("Alpine.store('tsu').close('forced')")


def test_stacking_close_peels_only_top(page, base_url, e2e_auth_token):
    """Opening B atop A and pressing Esc closes only B."""
    page = _authenticated(page, base_url, e2e_auth_token)

    # Inject two ad-hoc modals so we don't depend on which view is mounted.
    page.evaluate(
        """
        () => {
          const tpl = `
            <div id="t-stack-a" x-data="tsuModal('stack-a')" x-show="open" x-cloak
                 class="tsu-modal-backdrop stack-a"
                 @keydown.escape.window="onEscape()" @click.self="onBackdrop()">
              <div class="tsu-modal --md" role="dialog" aria-modal="true"
                   x-trap.noscroll="open && isTop">
                <header class="tsu-modal-head"><h2 class="tsu-modal-title">A</h2></header>
                <div class="tsu-modal-body"><p>layer A</p></div>
              </div>
            </div>
            <div id="t-stack-b" x-data="tsuModal('stack-b')" x-show="open" x-cloak
                 class="tsu-modal-backdrop stack-b"
                 @keydown.escape.window="onEscape()" @click.self="onBackdrop()">
              <div class="tsu-modal --sm" role="dialog" aria-modal="true"
                   x-trap.noscroll="open && isTop">
                <header class="tsu-modal-head"><h2 class="tsu-modal-title">B</h2></header>
                <div class="tsu-modal-body"><p>layer B</p></div>
              </div>
            </div>`;
          document.body.insertAdjacentHTML('beforeend', tpl);
          Alpine.initTree(document.getElementById('t-stack-a'));
          Alpine.initTree(document.getElementById('t-stack-b'));
          Alpine.store('tsu').open('stack-a');
          Alpine.store('tsu').open('stack-b');
        }
        """
    )

    page.wait_for_function(
        "() => Alpine.store('tsu').stack.length === 2 && Alpine.store('tsu').top === 'stack-b'",
        timeout=2000,
    )

    page.keyboard.press("Escape")
    page.wait_for_function(
        "() => Alpine.store('tsu').stack.length === 1 && Alpine.store('tsu').top === 'stack-a'",
        timeout=2000,
    )

    # A is still open; B is closed.
    assert page.evaluate("Alpine.store('tsu').isOpen('stack-a')") is True
    assert page.evaluate("Alpine.store('tsu').isOpen('stack-b')") is False

    page.evaluate("Alpine.store('tsu').close('stack-a')")


def test_modal_pins_to_bottom_on_mobile_viewport(page, base_url, e2e_auth_token):
    """At narrow viewports the panel should anchor to the bottom of the screen
    (sheet-style) rather than float centered — see the @media (max-width:560px)
    branch in tsu-modal.css."""
    page = _authenticated(page, base_url, e2e_auth_token)
    page.set_viewport_size({"width": 390, "height": 820})
    _open_settings(page)
    page.wait_for_timeout(400)

    align = page.evaluate(
        "(() => { const el = document.querySelector('.tsu-modal-backdrop.settings-modal');"
        " return getComputedStyle(el).alignItems; })()"
    )
    assert align == "flex-end", f"expected sheet-style alignItems flex-end, got {align!r}"

    page.evaluate("Alpine.store('tsu').close('settings')")


@pytest.mark.parametrize("theme", THEMES)
def test_modal_paints_visibly_in_each_theme(page, base_url, e2e_auth_token, theme):
    """The pure-CSS entrance keyframes must settle at opacity:1 (not the stuck-0
    state from the original Alpine x-transition bug)."""
    page = _authenticated(page, base_url, e2e_auth_token)
    page.evaluate(f"Alpine.store('app').theme = {theme!r}")
    # `data-theme` is applied to <body> by the Alpine binding; wait for it.
    page.wait_for_function(
        f"() => document.body.getAttribute('data-theme') === {theme!r}",
        timeout=2000,
    )
    _open_settings(page)

    # Give the keyframe its 0.16s budget to complete.
    page.wait_for_timeout(400)

    backdrop_opacity = page.evaluate(
        "(() => { const el = document.querySelector('.tsu-modal-backdrop.settings-modal');"
        " return getComputedStyle(el).opacity; })()"
    )
    panel_opacity = page.evaluate(
        "(() => { const el = document.querySelector('.tsu-modal-backdrop.settings-modal .tsu-modal');"
        " return getComputedStyle(el).opacity; })()"
    )
    assert float(backdrop_opacity) > 0.95, f"backdrop stuck at opacity {backdrop_opacity} on {theme}"
    assert float(panel_opacity) > 0.95, f"panel stuck at opacity {panel_opacity} on {theme}"

    page.evaluate("Alpine.store('tsu').close('settings')")
