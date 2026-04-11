"""Basic page load and navigation tests."""

import pytest


def test_app_loads(authenticated_page):
    page = authenticated_page
    assert "Tsugite" in page.title() or page.locator("nav h1").text_content() == "Tsugite"


def test_auth_gate_shown_without_token(page, base_url):
    page.goto(base_url)
    page.wait_for_selector(".auth-gate.open", timeout=3000)
    assert page.locator(".auth-gate.open").is_visible()


def test_agent_selector_populated(authenticated_page):
    page = authenticated_page
    page.wait_for_function("Alpine.store('app').agents.length > 0", timeout=5000)
    agents = page.evaluate("Alpine.store('app').agents.map(a => a.name)")
    assert "test-agent" in agents


@pytest.mark.parametrize(
    "tab", ["conversations", "workspace", "schedules", "webhooks", "usage"]
)
def test_tab_loads_without_errors(authenticated_page, tab):
    """Each main tab should load without JS errors."""
    page = authenticated_page
    errors = []
    page.on("pageerror", lambda exc: errors.append(str(exc)))

    page.locator("nav button", has_text=tab.capitalize()).click()
    page.wait_for_function(f"Alpine.store('app').view === '{tab}'", timeout=3000)
    page.wait_for_timeout(500)  # let any async init settle

    assert not errors, f"JS errors on {tab} tab: {errors}"
