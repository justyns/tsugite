"""Tests for `record_user_input` threading channel_metadata through to the event payload."""

from tsugite.agent_runner.history_integration import record_user_input
from tsugite.history import SessionStorage


def _new_storage(tmp_path):
    return SessionStorage.create(
        agent_name="test-agent",
        model="anthropic:claude-3-5-sonnet-20241022",
        workspace=None,
        session_path=tmp_path / "session.jsonl",
    )


def test_reply_to_session_metadata_lands_on_user_input_event(tmp_path):
    """When the daemon's reply_to_session passes metadata (job_id, kind) through,
    the resulting user_input event must carry a `channel` field that includes
    those keys — so the frontend can switch from text-regex to
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
    must work exactly as before — no 'channel' key, no crash."""
    storage = _new_storage(tmp_path)
    record_user_input(storage, "hello world")
    events = list(storage.iter_events())
    user_events = [e for e in events if e.type == "user_input"]
    assert len(user_events) == 1
    assert "channel" not in user_events[0].data, (
        "channel must be absent when channel_metadata is not provided to preserve existing event shape"
    )
    assert user_events[0].data.get("text") == "hello world"
