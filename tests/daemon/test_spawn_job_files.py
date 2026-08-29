"""spawn_job(files=...) delivers workspace files to the Job's worker session.

The tool validates paths against the spawner's workspace and forwards them (as
path strings) to the orchestrator, which stashes them on the Job and threads them
onto the worker session's metadata; the runner materializes them into first-turn
attachments once the target model is known, so a delegated image reaches a vision
model as pixels. Non-inlinable files degrade to a path hint on the message.
"""

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from tsugite_daemon.session_store import Session, SessionSource, SessionStore

from tsugite.attachments.base import AttachmentContentType

JPEG = b"\xff\xd8\xff\xe0" + b"0" * 100


# -- tool side: validate + forward paths to the orchestrator --


def _wire_tool(monkeypatch, tmp_path):
    """Point the jobs tool at a stub orchestrator + capture the forwarded kwargs."""
    import tsugite.tools.jobs as jobs_tool

    monkeypatch.setattr("tsugite.cli.helpers.get_workspace_dir", lambda: tmp_path)
    monkeypatch.setattr(jobs_tool, "_jobs_orchestrator", SimpleNamespace(create_and_start_job=lambda **k: None))
    monkeypatch.setattr("tsugite_daemon.session_runner.get_current_session_id", lambda: "parent-1")

    seen: dict = {}

    def fake_call(fn, *args, timeout=30, **kwargs):
        seen.update(kwargs)
        return SimpleNamespace(id="job-1", notify_when="never"), SimpleNamespace(id="session-1")

    monkeypatch.setattr(jobs_tool, "_call", fake_call)
    return jobs_tool, seen


def test_spawn_job_forwards_resolved_files(tmp_path, monkeypatch):
    jobs_tool, seen = _wire_tool(monkeypatch, tmp_path)
    img = tmp_path / "photo.jpg"
    img.write_bytes(JPEG)

    jobs_tool.spawn_job(prompt="look at this", files=["photo.jpg"])

    assert seen["delegation_files"] == [str(img.resolve())]


def test_spawn_job_without_files_forwards_none(tmp_path, monkeypatch):
    jobs_tool, seen = _wire_tool(monkeypatch, tmp_path)

    jobs_tool.spawn_job(prompt="plain task")

    assert seen["delegation_files"] is None


def test_spawn_job_rejects_traversal(tmp_path, monkeypatch):
    jobs_tool, _ = _wire_tool(monkeypatch, tmp_path)
    (tmp_path.parent / "secret.txt").write_text("x")

    with pytest.raises(ValueError, match="escapes"):
        jobs_tool.spawn_job(prompt="read", files=["../secret.txt"])


def test_spawn_job_rejects_missing(tmp_path, monkeypatch):
    jobs_tool, _ = _wire_tool(monkeypatch, tmp_path)

    with pytest.raises(ValueError, match="not found"):
        jobs_tool.spawn_job(prompt="read", files=["ghost.jpg"])


# -- runner side: materialize into first-turn attachments --


@pytest.fixture
def mock_adapter():
    adapter = MagicMock()
    adapter.handle_message = AsyncMock(return_value="done")
    adapter.runtime = MagicMock()
    adapter.runtime.workspace_dir = Path("/tmp/test")
    adapter.session_store = MagicMock()
    return adapter


@pytest.mark.asyncio
async def test_run_session_delivers_image_as_first_turn_attachment(tmp_path, mock_adapter, monkeypatch):
    from tsugite_daemon.session_runner import SessionRunner

    monkeypatch.setattr("tsugite.models.model_supports_vision", lambda m: True)
    img = tmp_path / "photo.jpg"
    img.write_bytes(JPEG)

    store = SessionStore(tmp_path / "store.json")
    runner = SessionRunner(store, mock_adapter)
    session = Session(
        id="s1",
        source=SessionSource.SPAWNED.value,
        prompt="what is in this image?",
        model="claude_code:haiku",
        metadata={"delegation_files": [str(img)]},
    )
    runner.start_session(session)
    await asyncio.sleep(0.3)

    ctx = mock_adapter.handle_message.call_args[1]["channel_context"]
    attachments = ctx.metadata["uploaded_attachments"]
    assert any(a.content_type == AttachmentContentType.IMAGE for a in attachments)


@pytest.mark.asyncio
async def test_run_session_non_inlinable_file_degrades_to_hint(tmp_path, mock_adapter, monkeypatch):
    from tsugite_daemon.session_runner import SessionRunner

    monkeypatch.setattr("tsugite.models.model_supports_vision", lambda m: True)
    svg = tmp_path / "diagram.svg"
    svg.write_bytes(b"<svg/>")

    store = SessionStore(tmp_path / "store.json")
    runner = SessionRunner(store, mock_adapter)
    session = Session(
        id="s2",
        source=SessionSource.SPAWNED.value,
        prompt="describe this",
        model="claude_code:haiku",
        metadata={"delegation_files": [str(svg)]},
    )
    runner.start_session(session)
    await asyncio.sleep(0.3)

    call = mock_adapter.handle_message.call_args
    assert "diagram.svg" in call[1]["message"]  # not lost
    assert "uploaded_attachments" not in call[1]["channel_context"].metadata


@pytest.mark.asyncio
async def test_run_session_without_files_unchanged(tmp_path, mock_adapter):
    from tsugite_daemon.session_runner import SessionRunner

    store = SessionStore(tmp_path / "store.json")
    runner = SessionRunner(store, mock_adapter)
    session = Session(id="s3", source=SessionSource.SPAWNED.value, prompt="hi")
    runner.start_session(session)
    await asyncio.sleep(0.3)

    call = mock_adapter.handle_message.call_args
    assert call[1]["message"] == "hi"
    assert "uploaded_attachments" not in call[1]["channel_context"].metadata
