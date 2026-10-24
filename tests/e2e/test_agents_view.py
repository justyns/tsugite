"""Agents view: creating an agent from the form seeds its workspace and lists it, and an edit persists."""

from playwright.sync_api import expect

from .helpers import open_view


def test_create_then_edit_an_agent(authenticated_page, e2e_session_store, e2e_workspace):
    agents = e2e_session_store.agents
    if agents.find_default() is None:
        agents.ensure_default(e2e_workspace, "default")
    page = authenticated_page
    open_view(page, "agents")

    page.locator('[data-testid="agent-create"]').click()
    page.get_by_label("id", exact=True).fill("Bad Id")
    expect(page.locator('[data-testid="agent-record-save"]')).to_be_disabled()
    page.get_by_label("id", exact=True).fill("e2e-atlas")
    page.get_by_label("name").fill("Atlas")
    page.get_by_label("template").select_option("default")
    page.get_by_label("color").select_option("teal")
    page.locator('[data-testid="agent-record-save"]').click()

    row = page.locator('[data-testid="agent-record-e2e-atlas"]')
    expect(row).to_contain_text("Atlas")
    assert agents.get("e2e-atlas").color == "teal"

    page.get_by_label("name").fill("Atlas Prime")
    page.locator('[data-testid="agent-record-save"]').click()
    expect(row).to_contain_text("Atlas Prime")
    assert agents.get("e2e-atlas").name == "Atlas Prime"
