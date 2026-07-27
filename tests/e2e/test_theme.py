"""Theme switch: picking a theme actually re-skins the app - the design token
values change (checked via getComputedStyle), not just a data-theme attribute
flip with no visual effect."""


def _bg0(page) -> str:
    return page.evaluate("getComputedStyle(document.documentElement).getPropertyValue('--bg0').trim()")


def test_theme_switch_changes_token_values(authenticated_page):
    page = authenticated_page

    assert page.evaluate("document.documentElement.dataset.theme") == "mocha"
    before = _bg0(page)
    assert before

    # Theme selection lives in the settings drawer (not the top bar).
    page.locator('[data-testid="settings-trigger"]').click()
    page.locator('[data-testid="theme-switch"]').get_by_role("button", name="latte", exact=True).click()
    page.keyboard.press("Escape")

    assert page.evaluate("document.documentElement.dataset.theme") == "latte"
    after = _bg0(page)
    assert after and after != before
