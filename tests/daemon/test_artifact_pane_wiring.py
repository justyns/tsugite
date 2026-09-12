"""`open_artifact` end-to-end through the daemon's real SSE broadcaster.

The tool tests (tests/test_artifact_tools.py) pin argument handling against a fake
bus. This pins the wiring the gateway actually builds: a real HTTPAgentAdapter
supplying the workspace root, a real SSEBroadcaster, and a real subscriber queue,
so the frame is asserted as it comes off the wire.
"""

import json

import pytest
from tsugite_daemon.adapters.http import HTTPAgentAdapter
from tsugite_daemon.adapters.http.sse import SSEBroadcaster
from tsugite_daemon.config import RuntimeDefaults
from tsugite_daemon.session_store import SessionStore

import tsugite.tools.artifacts as artifacts_tool
from tsugite.tools.artifacts import ARTIFACT_EVENT, open_artifact


@pytest.fixture
def workspace(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "report.html").write_text("<html><body><h1>Report</h1></body></html>")
    (tmp_path / "outside.txt").write_text("secret")
    return ws


@pytest.fixture
def adapter(workspace, tmp_path):
    runtime = RuntimeDefaults(workspace_dir=workspace, agent_file="default")
    return HTTPAgentAdapter(runtime=runtime, session_store=SessionStore(tmp_path / "sessions.json"))


@pytest.fixture
def wired(adapter, monkeypatch):
    """Wire the bridge the way Gateway._start does, on a real broadcaster."""
    broadcaster = SSEBroadcaster()
    artifacts_tool.set_artifact_bridge(adapter, broadcaster)
    monkeypatch.setattr(artifacts_tool, "get_current_session_id", lambda: "sess-42")
    yield broadcaster
    artifacts_tool.set_artifact_bridge(None, None)


@pytest.mark.asyncio
async def test_the_frame_a_browser_receives_carries_the_artifact(wired):
    queue = wired.subscribe()

    open_artifact(path="report.html", title="Coverage")

    frame = queue.get_nowait()
    assert frame["type"] == "session_event"
    data = frame["data"]
    assert data["event_type"] == ARTIFACT_EVENT
    assert data["session_id"] == "sess-42"
    assert data["path"] == "report.html"
    assert data["content_type"] == "html"
    assert data["mode"] == "rendered"
    assert data["title"] == "Coverage"
    assert data["placement"] == "right"
    assert data["opened_by"] == "agent"
    # The frame is JSON-serialized onto the SSE stream verbatim.
    json.dumps(frame)


@pytest.mark.asyncio
async def test_a_path_outside_the_adapter_workspace_never_reaches_the_wire(wired):
    queue = wired.subscribe()

    with pytest.raises(PermissionError, match="outside the workspace"):
        open_artifact(path="../outside.txt")

    assert queue.empty()
    assert wired.seq == 0


@pytest.mark.asyncio
async def test_a_second_open_replaces_the_same_pane(wired):
    queue = wired.subscribe()

    open_artifact(path="report.html")
    open_artifact(content="# later", content_type="markdown", title="Later")

    first, second = queue.get_nowait()["data"], queue.get_nowait()["data"]
    assert first["artifact_id"] == second["artifact_id"]
    assert second["content"] == "# later"
