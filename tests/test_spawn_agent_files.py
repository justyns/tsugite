"""spawn_agent(files=...) hands workspace files to the subagent subprocess.

The parent validates paths against its workspace, gates them for the child's
model, and puts inlinable ones on the stdin-JSON `files` key (the child
materializes them); non-inlinable ones degrade to a prompt path hint.
"""

import json
from unittest.mock import MagicMock

import pytest

from tsugite.tools.agents import spawn_agent

JPEG = b"\xff\xd8\xff\xe0" + b"0" * 100


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """A workspace dir bound as the current agent's cwd, with a spawnable agent."""
    (tmp_path / "looker.md").write_text("---\nname: looker\n---\nLook at things.\n")
    monkeypatch.setattr("tsugite.tools.agents.get_workspace_dir", lambda: tmp_path)
    monkeypatch.setattr("tsugite.models.model_supports_vision", lambda m: True)
    return tmp_path


def _mock_popen(monkeypatch):
    """Patch subprocess.Popen with a proc that reports success; capture stdin."""
    capture = {}
    proc = MagicMock()
    proc.stdin.write.side_effect = lambda data: capture.__setitem__("stdin", data)
    proc.stdout = iter(['{"type": "final_result", "result": "OK"}\n'])
    proc.wait.return_value = 0
    proc.stderr.read.return_value = ""
    monkeypatch.setattr("subprocess.Popen", lambda *a, **k: proc)
    return capture


def test_inline_files_ride_to_child_as_paths(workspace, monkeypatch):
    (workspace / "photo.jpg").write_bytes(JPEG)
    (workspace / "notes.txt").write_text("hello")
    capture = _mock_popen(monkeypatch)

    result = spawn_agent(
        str(workspace / "looker.md"),
        "What is in this image?",
        files=["photo.jpg", "notes.txt"],
        model_override="claude_code:haiku",
    )

    assert result == "OK"
    payload = json.loads(capture["stdin"])
    assert str((workspace / "photo.jpg").resolve()) in payload["files"]
    assert str((workspace / "notes.txt").resolve()) in payload["files"]


def test_non_inlinable_file_degrades_to_prompt_hint(workspace, monkeypatch):
    (workspace / "diagram.svg").write_bytes(b"<svg/>")
    capture = _mock_popen(monkeypatch)

    spawn_agent(
        str(workspace / "looker.md"),
        "Describe this",
        files=["diagram.svg"],
        model_override="claude_code:haiku",
    )

    payload = json.loads(capture["stdin"])
    assert payload.get("files") in (None, [])  # svg never inlines
    assert "diagram.svg" in payload["prompt"]  # but it is not lost


def test_no_files_leaves_payload_unchanged(workspace, monkeypatch):
    capture = _mock_popen(monkeypatch)

    spawn_agent(str(workspace / "looker.md"), "Just do it")

    payload = json.loads(capture["stdin"])
    assert "files" not in payload
    assert payload["prompt"] == "Just do it"


def test_traversal_rejected(workspace):
    (workspace.parent / "secret.txt").write_text("x")
    with pytest.raises(ValueError, match="escapes"):
        spawn_agent(str(workspace / "looker.md"), "read", files=["../secret.txt"])


def test_missing_file_rejected(workspace):
    with pytest.raises(ValueError, match="not found"):
        spawn_agent(str(workspace / "looker.md"), "read", files=["ghost.jpg"])


def test_child_materializes_delegated_files(tmp_path, monkeypatch):
    """The subagent-mode child reads `files` from stdin and attaches the pixels."""
    import io

    from tsugite.agent_runner import runner
    from tsugite.attachments.base import AttachmentContentType

    img = tmp_path / "p.jpg"
    img.write_bytes(JPEG)
    payload = json.dumps({"prompt": "hi", "context": {}, "files": [str(img)]})
    monkeypatch.setenv("TSUGITE_SUBAGENT_MODE", "1")
    monkeypatch.setattr("sys.stdin", io.StringIO(payload))

    captured = {}

    async def fake_async(**kwargs):
        captured.update(kwargs)
        return "done"

    monkeypatch.setattr(runner, "run_agent_async", fake_async)

    result = runner.run_agent(agent_path=tmp_path / "x.md", prompt="")

    assert result == "done"
    attachments = captured["attachments"]
    assert any(a.content_type == AttachmentContentType.IMAGE for a in attachments)
