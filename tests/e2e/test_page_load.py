"""Basic page load and navigation tests."""


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
