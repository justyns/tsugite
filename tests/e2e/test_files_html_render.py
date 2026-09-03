"""Workspace file browser renders .html files, with source as the fallback.

Drives the real daemon + a real chromium: an HTML file seeded into the fixture
workspace is opened through the file rail, rendered inside the sandboxed frame
(asserted by reading the frame's own DOM, not just the markup we handed it), then
toggled back to source.

The isolation contract is asserted here too: the frame carries `sandbox=""`, its
CSP admits no network source, and the script in the seeded file must not have
run.
"""

from playwright.sync_api import expect

from .helpers import open_view

HTML_FRAME = '[data-testid="files-html-frame"]'

REPORT = """<html>
<head><title>Report</title><link rel="stylesheet" href="e2e_report.css"></head>
<body>
  <h1>E2E coverage report</h1>
  <p id="verdict">not tampered</p>
  <script>document.getElementById('verdict').textContent = 'SCRIPT RAN';</script>
</body>
</html>
"""


def _open_html(page, e2e_workspace):
    (e2e_workspace / "e2e_report.css").write_text("#verdict { font-weight: 700 }\n")
    (e2e_workspace / "e2e_report.html").write_text(REPORT)

    open_view(page, "files")
    # The testid-bearing flat list only renders while a search is active.
    page.get_by_label("Search workspace").fill("e2e_report.html")
    page.locator('[data-testid="file-node-e2e_report.html"]').click()


def test_html_opens_rendered_and_toggles_back_to_source(authenticated_page, e2e_workspace):
    page = authenticated_page
    _open_html(page, e2e_workspace)

    # Rendered by default: the frame's own document shows the heading.
    frame = page.locator(HTML_FRAME)
    expect(frame).to_be_visible()
    expect(page.frame_locator(HTML_FRAME).locator("h1")).to_have_text("E2E coverage report")

    # Scripts never run - the sandbox denies them and the CSP denies them again.
    expect(page.frame_locator(HTML_FRAME).locator("#verdict")).to_have_text("not tampered")

    # Source view is the fallback, and shows the file as it is on disk.
    page.get_by_role("button", name="raw", exact=True).click()
    expect(page.locator(HTML_FRAME)).to_have_count(0)
    expect(page.locator('[data-testid="files-doc"]')).to_contain_text("SCRIPT RAN")

    # ...and back.
    page.get_by_role("button", name="rendered", exact=True).click()
    expect(page.frame_locator(HTML_FRAME).locator("h1")).to_have_text("E2E coverage report")


def test_the_preview_frame_grants_nothing_and_blocks_the_network(authenticated_page, e2e_workspace):
    page = authenticated_page
    _open_html(page, e2e_workspace)

    frame = page.locator(HTML_FRAME)
    expect(frame).to_be_visible()
    # Empty sandbox: without allow-same-origin the frame cannot read the app's
    # localStorage bearer token.
    assert frame.get_attribute("sandbox") == ""
    srcdoc = frame.get_attribute("srcdoc")
    assert "default-src 'none'" in srcdoc
    assert "http" not in srcdoc.split("Content-Security-Policy")[1].split(">")[0]


def test_a_same_workspace_stylesheet_is_inlined_so_the_report_keeps_its_styling(authenticated_page, e2e_workspace):
    page = authenticated_page
    _open_html(page, e2e_workspace)

    verdict = page.frame_locator(HTML_FRAME).locator("#verdict")
    expect(verdict).to_have_text("not tampered")
    # The relative stylesheet was read back through the authenticated workspace
    # API and inlined; the frame itself made no request.
    expect(verdict).to_have_css("font-weight", "700")


def test_a_markdown_file_still_renders_as_markdown(authenticated_page, e2e_workspace):
    (e2e_workspace / "e2e_not_html.md").write_text("# Still markdown\n")

    page = authenticated_page
    open_view(page, "files")
    page.get_by_label("Search workspace").fill("e2e_not_html")
    page.locator('[data-testid="file-node-e2e_not_html.md"]').click()

    expect(page.locator('[data-testid="files-doc"]').locator("h1")).to_have_text("Still markdown")
    expect(page.locator(HTML_FRAME)).to_have_count(0)
