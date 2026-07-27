"""Auto-heal for an unresumable (poisoned) provider session.

A resumed session-owning transcript (e.g. a Claude Code sidecar) that picked up an
empty text block gets rejected with `400 ... text content blocks must be
non-empty` on every resume. The runner retries on a fresh session AND records a
`resume_reset` boundary so later messages stop re-resolving the dead session id
from history (which is why odyn's *next* message failed immediately too).
"""

from tsugite.agent_runner.history_integration import get_resumable_session_state, record_resume_reset
from tsugite.exceptions import AgentExecutionError, is_unresumable_history_error
from tsugite.history import get_history_backend


def test_classifier_matches_the_poison_400_only():
    poison = "API Error: 400 messages: text content blocks must be non-empty (subtype=success)"
    assert is_unresumable_history_error(poison)
    assert is_unresumable_history_error(AgentExecutionError(poison))
    assert not is_unresumable_history_error("Prompt is too long")
    assert not is_unresumable_history_error("API Error: 400 invalid model id")


def _conv_with_recorded_session(session_id: str) -> str:
    backend = get_history_backend()
    session = backend.create(agent_name="bot", model="claude_code:opus")
    session.record("user_input", text="hi")
    session.record("model_response", raw_content="ok", state_delta={"session_id": session_id})
    return session.session_id


def test_resume_reset_severs_the_recorded_session():
    cid = _conv_with_recorded_session("cc-poisoned")
    assert get_resumable_session_state(cid).session_id == "cc-poisoned"

    record_resume_reset(cid)
    assert get_resumable_session_state(cid) is None


def test_a_session_recorded_after_the_reset_is_resumable_again():
    # The fresh retry records a new clean session id after the reset boundary;
    # that one must resume normally on the next message (the self-heal path).
    cid = _conv_with_recorded_session("cc-poisoned")
    record_resume_reset(cid)
    get_history_backend().load(cid).record("model_response", raw_content="ok", state_delta={"session_id": "cc-fresh"})
    assert get_resumable_session_state(cid).session_id == "cc-fresh"


def test_record_resume_reset_is_best_effort_on_unknown_conversation():
    # Must never raise (it runs inside the failing turn's except block).
    assert record_resume_reset("does-not-exist") is None
    assert record_resume_reset("") is None


def test_record_resume_reset_returns_the_recorded_notice_payload():
    # The runner reuses the return value to surface the same notice live on the
    # healing turn, so a successful record must hand back the recorded data.
    cid = _conv_with_recorded_session("cc-poisoned")
    payload = record_resume_reset(cid)
    assert payload is not None
    assert payload["reason"] == "poisoned_transcript"
    assert "reset" in payload["message"]
