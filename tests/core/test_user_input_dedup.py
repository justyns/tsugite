"""A failed turn must not record the user's message twice.

The live runner records `user_input` at turn start (agent_runner.runner). On a
turn that fails before producing a `model_response`, the daemon's post-hoc
`save_run_to_history` re-opens the same session and, seeing no `model_response`,
used to re-record the whole turn -- including a second identical `user_input`,
which surfaced as a duplicate user bubble in the web UI. The two recorders share
no in-memory flag, so the guard has to be derived from the session's own events:
a turn runs from its `user_input` to its `session_end`, so an already-recorded
`user_input` with no `session_end` after it belongs to the current turn.
"""

from pathlib import Path

from tsugite.agent_runner.history_integration import record_user_input, save_run_to_history
from tsugite.history import SessionStorage, get_history_backend


def _storage(tmp_path: Path) -> SessionStorage:
    return SessionStorage.create(agent_name="t", model="openai:gpt-4o-mini", session_path=tmp_path / "s.jsonl")


def _user_inputs(storage):
    return [e for e in storage.iter_events() if e.type == "user_input"]


def test_record_user_input_skips_dup_in_incomplete_turn(tmp_path: Path):
    """The live runner already recorded this turn's user_input; the post-hoc
    error-path recorder must not add a second one. The intervening event is a
    model_request (the turn's failed model call), not the user_input itself, so
    a naive "last event is an identical user_input" guard would miss it."""
    storage = _storage(tmp_path)
    storage.record("user_input", text="hello")
    storage.record("model_request")

    record_user_input(storage, "hello")

    assert len(_user_inputs(storage)) == 1


def test_record_user_input_records_repeat_in_new_turn(tmp_path: Path):
    """The same message sent as a separate, completed turn (a session_end
    between) is a legit repeat, not a retry duplicate -- it must record."""
    storage = _storage(tmp_path)
    storage.record("user_input", text="hello")
    storage.record("model_response", raw_content="hi there")
    storage.record("final_result", result="hi there")
    storage.record("session_end", status="success")

    record_user_input(storage, "hello")

    assert [e.data["text"] for e in _user_inputs(storage)] == ["hello", "hello"]


def test_save_run_to_history_no_dup_user_input_on_failed_turn(tmp_path: Path):
    """End-to-end reproduction of the reported bug: the live runner recorded
    user_input, the turn failed before any model_response, and the daemon error
    path called save_run_to_history against the same session."""
    backend = get_history_backend()
    storage = backend.create(agent_name="t", model="openai:gpt-4o-mini", session_id="failsess")
    storage.record("user_input", text="hello")
    storage.record("model_request")

    save_run_to_history(
        agent_path=Path("nonexistent-agent.md"),
        agent_name="t",
        prompt="hello",
        result="[Error: bogus model]",
        model="openai:gpt-4o-mini",
        continue_conversation_id="failsess",
        status="error",
        error_message="bogus model",
    )

    reloaded = backend.load("failsess")
    assert len([e for e in reloaded.iter_events() if e.type == "user_input"]) == 1
