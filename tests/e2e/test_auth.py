"""Auth flow tests."""


def test_valid_token_unlocks_app(page, base_url, e2e_auth_token):
    """Entering a valid token in the auth gate dismisses it and loads agents."""
    page.goto(base_url)
    page.wait_for_selector(".auth-gate.open", timeout=3000)

    page.locator(".auth-gate input[type='password']").fill(e2e_auth_token)
    page.locator(".auth-btn").click()

    # After successful auth, page reloads and auth gate should be gone
    page.wait_for_function("!Alpine.store('app').authRequired", timeout=10000)
    assert not page.locator(".auth-gate.open").is_visible()


def test_invalid_token_shows_error(page, base_url):
    """Entering a bad token shows an error message and keeps the gate open."""
    page.goto(base_url)
    page.wait_for_selector(".auth-gate.open", timeout=3000)

    page.locator(".auth-gate input[type='password']").fill("tsu_totally_bogus_token")
    page.locator(".auth-btn").click()

    page.wait_for_selector(".auth-error", state="visible", timeout=5000)
    assert "Invalid" in page.locator(".auth-error").text_content()
    assert page.locator(".auth-gate.open").is_visible()
