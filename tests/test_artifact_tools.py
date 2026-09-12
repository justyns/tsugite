"""Tests for `open_artifact`: path validation and event payload shape.

The tool is the daemon's one sanctioned way for an agent to put a document beside
the chat, so the interesting surface is what it refuses (anything outside the
workspace) and exactly what it puts on the wire (the frontend reducer is pinned
to that payload).
"""

import pytest

from tsugite.cli.helpers import set_workspace_dir
from tsugite.exceptions import ToolUnavailableError
from tsugite.tools import artifacts as artifacts_tool
from tsugite.tools.artifacts import AGENT_ARTIFACT_ID, ARTIFACT_EVENT, open_artifact


class FakeBus:
    def __init__(self):
        self.emitted = []

    def emit(self, event_type, data=None):
        self.emitted.append((event_type, data or {}))

    @property
    def last(self):
        return self.emitted[-1][1]


class FakeRuntime:
    def __init__(self, workspace_dir):
        self.workspace_dir = workspace_dir


class FakeAdapter:
    def __init__(self, workspace_dir):
        self.runtime = FakeRuntime(workspace_dir)


@pytest.fixture
def workspace(tmp_path):
    ws = tmp_path / "ws"
    (ws / "reports").mkdir(parents=True)
    (ws / "notes.md").write_text("# Notes\n\nbody\n")
    (ws / "reports" / "cov.html").write_text("<html><body><h1>Coverage</h1></body></html>")
    (tmp_path / "outside.md").write_text("secret\n")
    return ws


@pytest.fixture
def bus(workspace, monkeypatch):
    bus = FakeBus()
    artifacts_tool.set_artifact_bridge(FakeAdapter(workspace), bus)
    monkeypatch.setattr(artifacts_tool, "get_current_session_id", lambda: "sess-1")
    yield bus
    artifacts_tool.set_artifact_bridge(None, None)


class TestPathValidation:
    def test_opens_a_workspace_file(self, bus):
        out = open_artifact(path="notes.md")

        assert "notes.md" in out
        event_type, data = bus.emitted[-1]
        assert event_type == "session_event"
        assert data["event_type"] == ARTIFACT_EVENT
        assert data["session_id"] == "sess-1"
        assert data["path"] == "notes.md"

    def test_normalizes_a_nested_path(self, bus):
        open_artifact(path="./reports/../reports/cov.html")
        assert bus.last["path"] == "reports/cov.html"

    def test_rejects_traversal_out_of_the_workspace(self, bus):
        with pytest.raises(PermissionError, match="outside the workspace"):
            open_artifact(path="../outside.md")
        assert bus.emitted == []

    def test_rejects_an_absolute_path(self, bus, tmp_path):
        with pytest.raises(PermissionError, match="outside the workspace"):
            open_artifact(path=str(tmp_path / "outside.md"))
        assert bus.emitted == []

    def test_rejects_a_symlink_escaping_the_workspace(self, bus, workspace, tmp_path):
        link = workspace / "escape.md"
        link.symlink_to(tmp_path / "outside.md")
        with pytest.raises(PermissionError, match="outside the workspace"):
            open_artifact(path="escape.md")
        assert bus.emitted == []

    def test_rejects_a_missing_file(self, bus):
        with pytest.raises(ValueError, match="does not exist"):
            open_artifact(path="nope.md")

    def test_rejects_a_directory(self, bus):
        with pytest.raises(ValueError, match="not a file"):
            open_artifact(path="reports")

    def test_rejects_an_http_url(self, bus):
        with pytest.raises(PermissionError, match="external URL"):
            open_artifact(path="https://example.com/report.html")
        assert bus.emitted == []

    @pytest.mark.parametrize(
        "path",
        [
            "//example.com/report.html",
            "javascript:alert(1)",
            "data:text/html,<script>alert(1)</script>",
            "mailto:someone@example.com",
        ],
    )
    def test_rejects_a_protocol_relative_or_non_http_scheme(self, bus, path):
        with pytest.raises(PermissionError, match="external URL"):
            open_artifact(path=path)
        assert bus.emitted == []


class TestArguments:
    def test_requires_path_or_content(self, bus):
        with pytest.raises(ValueError, match="path.*or.*content"):
            open_artifact()

    def test_refuses_both_path_and_content(self, bus):
        with pytest.raises(ValueError, match="not both"):
            open_artifact(path="notes.md", content="hi")

    def test_rejects_an_unknown_content_type(self, bus):
        with pytest.raises(ValueError, match="content_type"):
            open_artifact(content="hi", content_type="pdf")

    def test_rejects_an_unknown_mode(self, bus):
        with pytest.raises(ValueError, match="mode"):
            open_artifact(path="notes.md", mode="edit")

    def test_rejects_an_unknown_placement(self, bus):
        with pytest.raises(ValueError, match="placement"):
            open_artifact(path="notes.md", placement="floating")

    def test_rejects_oversized_inline_content(self, bus):
        with pytest.raises(ValueError, match="too large"):
            open_artifact(content="x" * (artifacts_tool.MAX_INLINE_CONTENT + 1), content_type="text")

    def test_refuses_without_a_daemon_bridge(self):
        artifacts_tool.set_artifact_bridge(None, None)
        with pytest.raises(ToolUnavailableError, match="no daemon"):
            open_artifact(content="hi", content_type="text")


class TestEventPayload:
    def test_infers_content_type_and_mode_from_the_extension(self, bus):
        open_artifact(path="reports/cov.html")
        assert bus.last["content_type"] == "html"
        assert bus.last["mode"] == "rendered"

        open_artifact(path="notes.md")
        assert bus.last["content_type"] == "markdown"
        assert bus.last["mode"] == "rendered"

    def test_a_file_with_no_rendered_form_opens_as_source(self, bus, workspace):
        (workspace / "run.log").write_text("line\n")
        open_artifact(path="run.log")
        assert bus.last["content_type"] == "text"
        assert bus.last["mode"] == "source"

    def test_an_explicit_source_mode_wins(self, bus):
        open_artifact(path="reports/cov.html", mode="source")
        assert bus.last["mode"] == "source"

    def test_a_rendered_request_for_plain_text_falls_back_to_source(self, bus, workspace):
        (workspace / "run.log").write_text("line\n")
        open_artifact(path="run.log", mode="rendered")
        assert bus.last["mode"] == "source"

    def test_titles_default_to_the_file_name(self, bus):
        open_artifact(path="reports/cov.html")
        assert bus.last["title"] == "cov.html"

        open_artifact(path="notes.md", title="Release notes")
        assert bus.last["title"] == "Release notes"

    def test_a_path_artifact_carries_no_inline_content(self, bus):
        open_artifact(path="notes.md")
        assert bus.last["content"] is None

    def test_ephemeral_content_rides_in_the_payload(self, bus):
        open_artifact(content="# Report\n", content_type="markdown", title="Report")
        assert bus.last["path"] is None
        assert bus.last["content"] == "# Report\n"
        assert bus.last["content_type"] == "markdown"
        assert bus.last["title"] == "Report"

    def test_ephemeral_content_defaults_to_text(self, bus):
        open_artifact(content="plain")
        assert bus.last["content_type"] == "text"
        assert bus.last["mode"] == "source"

    def test_placement_rides_through(self, bus):
        open_artifact(path="notes.md", placement="below")
        assert bus.last["placement"] == "below"

    def test_repeated_opens_reuse_one_artifact_slot_by_default(self, bus):
        open_artifact(path="notes.md")
        open_artifact(path="reports/cov.html")
        ids = [data["artifact_id"] for _t, data in bus.emitted]
        assert ids == [AGENT_ARTIFACT_ID, AGENT_ARTIFACT_ID]

    def test_replace_existing_false_opens_its_own_slot(self, bus):
        open_artifact(path="notes.md")
        open_artifact(path="reports/cov.html", replace_existing=False)
        first, second = (data["artifact_id"] for _t, data in bus.emitted)
        assert first == AGENT_ARTIFACT_ID
        assert second != AGENT_ARTIFACT_ID

    def test_marks_the_pane_as_agent_opened(self, bus):
        open_artifact(path="notes.md")
        assert bus.last["opened_by"] == "agent"


class TestRegistration:
    def test_is_registered_as_a_daemon_only_tool(self):
        from tsugite.tools import _daemon_tools, get_tool, set_daemon_mode

        assert "open_artifact" in _daemon_tools
        set_daemon_mode(True)
        try:
            info = get_tool("open_artifact")
            assert info.require_daemon is True
            assert info.category == "artifacts"
        finally:
            set_daemon_mode(False)


class TestSessionWorkspace:
    """A job worker runs inside a provisioned worktree bound to the workspace
    ContextVar, so that tree is what a path resolves against - not the daemon's
    default workspace, which every adapter shares."""

    @pytest.fixture
    def worktree(self, tmp_path):
        wt = tmp_path / "worktree"
        wt.mkdir()
        (wt / "only-here.md").write_text("# worktree\n")
        return wt

    def test_resolves_against_the_bound_workspace(self, bus, worktree):
        set_workspace_dir(worktree)

        open_artifact(path="only-here.md")

        assert bus.last["path"] == "only-here.md"

    def test_refuses_a_file_that_exists_only_in_the_daemon_workspace(self, bus, worktree):
        set_workspace_dir(worktree)

        with pytest.raises(ValueError, match="does not exist in the workspace"):
            open_artifact(path="notes.md")
        assert bus.emitted == []

    def test_traversal_out_of_the_bound_workspace_is_still_refused(self, bus, worktree, workspace):
        set_workspace_dir(worktree)

        with pytest.raises(PermissionError, match="outside the workspace"):
            open_artifact(path="../ws/notes.md")
        assert bus.emitted == []
