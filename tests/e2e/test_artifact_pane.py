"""An agent opens a document in a pane beside the chat.

These drive the whole path in a real browser: the `open_artifact` tool is wired
to the running daemon exactly as the gateway wires it, the tool is called, and the
frame it broadcasts travels the live SSE stream into the app, which records it,
splits the focused pane, and mounts the artifact surface.
"""

import pytest
from playwright.sync_api import expect

import tsugite.tools.artifacts as artifacts_tool
from tsugite.tools.artifacts import open_artifact

from .helpers import E2E_USER_ID, wait_for_authed

PANE = '[data-testid="mux-pane"]'
ARTIFACT = '[data-testid="artifact-pane"]'
ARTIFACT_FRAME = '[data-testid="artifact-html-frame"]'
CONVERSATION = '[data-testid="chat-conversation"]'


@pytest.fixture
def agent_tool(e2e_server, e2e_adapter, e2e_workspace):
    """Wire open_artifact onto the live daemon, the way Gateway._start does."""
    _url, server = e2e_server
    artifacts_tool.set_artifact_bridge(e2e_adapter, server.event_bus)
    yield open_artifact
    artifacts_tool.set_artifact_bridge(None, None)


@pytest.fixture
def chat_open(authenticated_page, e2e_session_store):
    """A single-pane space showing the chat, ready for the agent to split."""
    page = authenticated_page
    e2e_session_store.get_or_create_interactive(E2E_USER_ID)
    page.reload()
    wait_for_authed(page)
    page.wait_for_selector(CONVERSATION)
    expect(page.locator(PANE)).to_have_count(1)
    return page


def test_the_agent_opens_a_markdown_file_beside_the_chat(chat_open, agent_tool, e2e_workspace):
    (e2e_workspace / "e2e_artifact_note.md").write_text("# Agent note\n\nRead me while we talk.\n")
    page = chat_open

    agent_tool(path="e2e_artifact_note.md", title="Agent note")

    # A second pane appears beside the chat; the chat is still there and usable.
    expect(page.locator(PANE)).to_have_count(2)
    expect(page.locator(ARTIFACT)).to_be_visible()
    expect(page.frame_locator(ARTIFACT_FRAME).locator("h1")).to_have_text("Agent note")
    expect(page.locator(CONVERSATION)).to_be_visible()
    expect(page.locator('[data-testid="chat-composer"]')).to_be_visible()

    # The user can see the agent did this.
    expect(page.locator('[data-testid="artifact-agent-badge"]')).to_contain_text("agent")


def test_the_agent_opens_an_html_artifact_in_the_sandboxed_frame(chat_open, agent_tool, e2e_workspace):
    (e2e_workspace / "e2e_artifact.html").write_text(
        "<html><body><h1>Agent report</h1>"
        "<p id='v'>clean</p>"
        "<script>document.getElementById('v').textContent='SCRIPT RAN'</script>"
        "</body></html>"
    )
    page = chat_open

    agent_tool(path="e2e_artifact.html", title="Agent report")

    expect(page.locator(ARTIFACT_FRAME)).to_be_visible()
    assert page.locator(ARTIFACT_FRAME).get_attribute("sandbox") == ""
    expect(page.frame_locator(ARTIFACT_FRAME).locator("h1")).to_have_text("Agent report")
    # Same isolation as the file browser: the script did not run.
    expect(page.frame_locator(ARTIFACT_FRAME).locator("#v")).to_have_text("clean")


def test_a_second_open_replaces_the_first_instead_of_stacking_panes(chat_open, agent_tool, e2e_workspace):
    (e2e_workspace / "e2e_artifact_first.md").write_text("# First artifact\n")
    (e2e_workspace / "e2e_artifact_second.md").write_text("# Second artifact\n")
    page = chat_open

    agent_tool(path="e2e_artifact_first.md", title="First")
    expect(page.frame_locator(ARTIFACT_FRAME).locator("h1")).to_have_text("First artifact")
    expect(page.locator(PANE)).to_have_count(2)

    agent_tool(path="e2e_artifact_second.md", title="Second")

    expect(page.frame_locator(ARTIFACT_FRAME).locator("h1")).to_have_text("Second artifact")
    expect(page.locator(ARTIFACT)).to_have_count(1)
    expect(page.locator(PANE)).to_have_count(2)


def test_the_pane_is_dismissible_and_leaves_the_chat_alone(chat_open, agent_tool, e2e_workspace):
    (e2e_workspace / "e2e_artifact_close.md").write_text("# Closeable\n")
    page = chat_open

    agent_tool(path="e2e_artifact_close.md", title="Closeable")
    expect(page.locator(ARTIFACT)).to_be_visible()

    page.locator('[data-testid="artifact-close"]').click()

    expect(page.locator(ARTIFACT)).to_have_count(0)
    expect(page.locator(PANE)).to_have_count(1)
    expect(page.locator(CONVERSATION)).to_be_visible()


def test_generated_content_with_no_file_on_disk_still_opens(chat_open, agent_tool):
    page = chat_open

    agent_tool(content="# Ephemeral\n\nnot on disk\n", content_type="markdown", title="Ephemeral")

    expect(page.frame_locator(ARTIFACT_FRAME).locator("h1")).to_have_text("Ephemeral")
    expect(page.frame_locator(ARTIFACT_FRAME).locator("body")).to_contain_text("not on disk")


def test_a_path_outside_the_workspace_never_opens_a_pane(chat_open, agent_tool, e2e_tmp):
    (e2e_tmp / "e2e_secret.md").write_text("# Secret\n")
    page = chat_open

    with pytest.raises(ValueError, match="outside the workspace"):
        agent_tool(path="../e2e_secret.md")

    expect(page.locator(ARTIFACT)).to_have_count(0)
    expect(page.locator(PANE)).to_have_count(1)
