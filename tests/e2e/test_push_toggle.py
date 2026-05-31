"""Push notification toggle: regression for the settings 'enable notifications'
button getting stuck on 'working…' when the underlying subscribe promise hangs.

Repro shape: stub `window.tsugiteSubscribePush` so it returns a Promise that
never resolves. With the buggy inline @click handler, the button's
`pushLoading` flag stays true forever — no timeout, no error toast, no
recovery. The fix must give the user a way out (timeout + surfaced error).
"""


def _wait_for_settings(page):
    page.wait_for_selector(".tsu-modal-backdrop.settings-modal", state="visible", timeout=5000)


def test_push_toggle_recovers_when_subscribe_hangs(page, base_url, e2e_auth_token):
    # Inject a stub for `tsugiteSubscribePush` that returns a hanging Promise
    # *before* any script runs, so the stub survives the SW-driven reload
    # (controllerchange → location.reload) that fires on first page load.
    page.add_init_script(
        "Object.defineProperty(window, 'tsugiteSubscribePush', {"
        "  configurable: true, writable: true,"
        "  value: () => new Promise(() => {})"
        "});"
    )

    page.goto(base_url + "/api/health")
    page.evaluate(f"localStorage.setItem('tsugite_token', '{e2e_auth_token}')")
    page.goto(base_url)
    page.wait_for_function(
        "typeof Alpine !== 'undefined' && Alpine.store('app') && !Alpine.store('app').authRequired",
        timeout=10000,
    )

    # Wait for the SW-driven reload (if any) to finish, then re-stub since the
    # add_init_script value may be overwritten when app.js re-assigns to
    # window.tsugiteSubscribePush. Use defineProperty with a setter that
    # ignores re-assignments to keep the stub locked in.
    page.evaluate(
        "(() => {"
        "  let v = () => new Promise(() => {});"
        "  Object.defineProperty(window, 'tsugiteSubscribePush', {"
        "    configurable: true,"
        "    get: () => v,"
        "    set: (_) => {},"
        "  });"
        "})()"
    )

    # Open settings via the store (more reliable than clicking the tab action
    # since the action is in two places — header and mobile menu).
    page.evaluate("Alpine.store('tsu').open('settings')")
    _wait_for_settings(page)

    # Find the push toggle button. The settings modal only renders it when
    # the browser exposes PushManager — Playwright Chromium does.
    btn = page.locator(".tsu-modal-backdrop.settings-modal button", has_text="enable notifications")
    btn.wait_for(state="visible", timeout=5000)
    btn.click()

    # Immediately we should see "working…" — but with the buggy code this
    # state persists forever. Give the fix up to 8s to time out and surface
    # an error / reset the button.
    page.wait_for_function(
        "() => {"
        " const b = [...document.querySelectorAll('.tsu-modal-backdrop.settings-modal button')]"
        "  .find(b => /working/i.test(b.textContent));"
        " return b !== undefined;"
        "}",
        timeout=2000,
    )

    # Capture the buggy in-progress state — on unfixed code the button stays
    # here forever; on fixed code this is the brief window before the 10s
    # timeout fires.
    page.screenshot(path="/tmp/tsugite-issue-state.png", full_page=True)

    # The bug: this never happens. After the fix: a timeout aborts the
    # subscribe and pushLoading is reset within ~10s. We give it 12s headroom.
    page.wait_for_function(
        "() => {"
        " const b = [...document.querySelectorAll('.tsu-modal-backdrop.settings-modal button')]"
        "  .find(b => /enable notifications|disable notifications/i.test(b.textContent));"
        " return b !== undefined && !b.disabled;"
        "}",
        timeout=12000,
    )

    # And the fix should surface the failure via a toast (not just console).
    page.wait_for_selector(".console-toast", timeout=2000)

    # Capture the recovered state for the fix screenshot — button back to
    # "enable notifications", toast visible. Only reachable when fixed.
    page.screenshot(path="/tmp/tsugite-issue-state-fixed.png", full_page=True)


def test_push_toggle_recovers_when_unsubscribe_hangs(page, base_url, e2e_auth_token):
    """Same recovery contract for the unsubscribe branch.

    Symmetric to the subscribe case — `tsugiteTogglePush` races both
    branches against the timeout, so if `tsugiteUnsubscribePush` hangs the
    button must still recover and surface an error.
    """
    page.add_init_script(
        "Object.defineProperty(window, 'tsugiteUnsubscribePush', {"
        "  configurable: true, writable: true,"
        "  value: () => new Promise(() => {})"
        "});"
    )

    page.goto(base_url + "/api/health")
    page.evaluate(f"localStorage.setItem('tsugite_token', '{e2e_auth_token}')")
    page.goto(base_url)
    page.wait_for_function(
        "typeof Alpine !== 'undefined' && Alpine.store('app') && !Alpine.store('app').authRequired",
        timeout=10000,
    )

    page.evaluate(
        "(() => {"
        "  let v = () => new Promise(() => {});"
        "  Object.defineProperty(window, 'tsugiteUnsubscribePush', {"
        "    configurable: true,"
        "    get: () => v,"
        "    set: (_) => {},"
        "  });"
        "})()"
    )

    # Open settings and force pushSubscribed=true so the button reads as
    # "disable notifications" and the click hits the unsubscribe branch.
    page.evaluate("Alpine.store('tsu').open('settings')")
    _wait_for_settings(page)
    page.evaluate(
        "(() => {"
        "  const panel = document.querySelector('.tsu-modal-backdrop.settings-modal .tsu-modal');"
        "  const data = Alpine.$data(panel);"
        "  data.pushSubscribed = true;"
        "})()"
    )

    btn = page.locator(".tsu-modal-backdrop.settings-modal button", has_text="disable notifications")
    btn.wait_for(state="visible", timeout=5000)
    btn.click()

    page.wait_for_function(
        "() => {"
        " const b = [...document.querySelectorAll('.tsu-modal-backdrop.settings-modal button')]"
        "  .find(b => /enable notifications|disable notifications/i.test(b.textContent));"
        " return b !== undefined && !b.disabled;"
        "}",
        timeout=12000,
    )
    page.wait_for_selector(".console-toast", timeout=2000)
