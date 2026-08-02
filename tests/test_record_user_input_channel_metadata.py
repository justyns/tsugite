"""Tests for `record_user_input` threading channel_metadata through to the event payload."""

from tsugite.agent_runner.history_integration import record_user_input
from tsugite.attachments.base import Attachment, AttachmentContentType
from tsugite.history import get_history_backend


def _att(name: str, user_upload: bool) -> Attachment:
    return Attachment(
        name=name,
        content="x",
        content_type=AttachmentContentType.TEXT,
        mime_type="text/plain",
        user_upload=user_upload,
    )


def _new_storage(tmp_path):
    return get_history_backend().create(
        agent_name="test-agent",
        model="anthropic:claude-3-5-sonnet-20241022",
        workspace=None,
    )


def test_reply_to_session_metadata_lands_on_user_input_event(tmp_path):
    """When the daemon's reply_to_session passes metadata (job_id, kind) through,
    the resulting user_input event must carry a `channel` field that includes
    those keys - so the frontend can switch from text-regex to
    event.data.channel.kind === 'job_notify'."""
    storage = _new_storage(tmp_path)
    record_user_input(
        storage,
        "Job job-xyz finished",
        channel_metadata={"job_id": "job-xyz", "kind": "job_notify", "source": "job_complete"},
    )
    events = list(storage.iter_events())
    user_events = [e for e in events if e.type == "user_input"]
    assert len(user_events) == 1
    channel = user_events[0].data.get("channel")
    assert channel is not None, "user_input event must include a 'channel' field when channel_metadata was passed"
    assert channel["job_id"] == "job-xyz"
    assert channel["kind"] == "job_notify"


def test_channel_metadata_optional_does_not_break_existing_callers(tmp_path):
    """Calling record_user_input without channel_metadata (the existing CLI path)
    must work exactly as before - no 'channel' key, no crash."""
    storage = _new_storage(tmp_path)
    record_user_input(storage, "hello world")
    events = list(storage.iter_events())
    user_events = [e for e in events if e.type == "user_input"]
    assert len(user_events) == 1
    assert "channel" not in user_events[0].data, (
        "channel must be absent when channel_metadata is not provided to preserve existing event shape"
    )
    assert user_events[0].data.get("text") == "hello world"


def test_only_user_uploads_are_recorded_as_attachments(tmp_path):
    """The recorded attachments field is display-only (the web UI renders it as
    clickable uploads/<name> chips), so it must list only files the user actually
    uploaded. The agent's auto-included context (workspace memory like USER.md)
    is not a user attachment and lives outside uploads/, so it must not appear -
    it rendered as a dead 'file not found' chip on every message otherwise."""
    storage = _new_storage(tmp_path)
    record_user_input(
        storage,
        "look at this",
        attachments=[_att("USER.md", False), _att("photo.jpg", True), _att("MEMORY.md", False)],
    )
    ev = next(e for e in storage.iter_events() if e.type == "user_input")
    assert [a["name"] for a in ev.data["attachments"]] == ["photo.jpg"]


def test_no_attachments_key_when_only_auto_context(tmp_path):
    """A turn carrying only auto-included context (no user upload) records no
    attachments key, preserving the plain-message event shape."""
    storage = _new_storage(tmp_path)
    record_user_input(storage, "hi", attachments=[_att("USER.md", False), _att("AGENTS.md", False)])
    ev = next(e for e in storage.iter_events() if e.type == "user_input")
    assert "attachments" not in ev.data
