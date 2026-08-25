"""Workspace wiki view: seeded markdown notes render, a [[wikilink]] navigates
between them, and the backlinks panel lists the referring note.

The wiki has no server-side index (see files/wiki.ts) - it's computed
client-side over `GET /api/workspace`, so seeding is just
writing real files into the fixture workspace directory. Wikilink resolution
works from paths alone; backlinks need the on-demand content scan (the meta
pane's "Scan workspace" affordance) - loading the view must never bulk-read
file contents by itself.
"""

from playwright.sync_api import expect

from .helpers import open_view


def test_wikilink_navigation_and_backlinks(authenticated_page, e2e_workspace):
    (e2e_workspace / "e2e_wiki_target.md").write_text("# Target\n\nThe target note.\n")
    (e2e_workspace / "e2e_wiki_source.md").write_text("# Source\n\nLinks to [[e2e_wiki_target]].\n")

    page = authenticated_page
    open_view(page, "files")

    # The flat, testid-bearing file list only renders while a search query is
    # active (the default tree view has no per-node testid) - search by the
    # seeded basename to get a stable, unambiguous handle on it.
    page.get_by_label("Search workspace").fill("e2e_wiki_source")
    page.locator('[data-testid="file-node-e2e_wiki_source.md"]').click()

    doc = page.locator('[data-testid="files-doc"]')
    expect(doc.locator("h1")).to_have_text("Source")

    wikilink = doc.locator("a.wikilink")
    expect(wikilink).to_be_visible()
    wikilink.click()

    expect(doc.locator("h1")).to_have_text("Target")

    # Backlinks are behind the explicit whole-workspace scan, never eager.
    page.get_by_role("button", name="Scan workspace").click()
    expect(page.locator('[data-testid="files-backlinks"]')).to_contain_text("e2e_wiki_source.md")
