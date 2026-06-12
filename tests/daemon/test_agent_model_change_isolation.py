"""Changing an agent's model via the config endpoint must not retroactively
flip existing sessions to the new model.

Sessions can have their own `model_override`. Sessions without one resolve
their model via the agent default (`agent_config.model`). Mutating that
default after sessions are alive means those sessions' next turn quietly
switches model — a cross-session leak in the same shape as #315 (agent-wide
state that should be per-session).

Fix: when the agent default changes, "freeze" each active session that
doesn't have an explicit override by stamping the OLD agent model as the
session's own `model_override`. Existing sessions stay on the model they
were resolving to; only new sessions pick up the new default.
"""

import pytest

from tsugite.daemon.session_store import Session, SessionSource, SessionStatus, SessionStore


@pytest.fixture
def store(tmp_path):
    return SessionStore(tmp_path / "store.json")


def _make(store, sid, agent="agent-a", status=SessionStatus.ACTIVE.value, override=None):
    s = Session(id=sid, agent=agent, source=SessionSource.INTERACTIVE.value, user_id="u1", status=status)
    store.create_session(s)
    if override:
        store.set_model_override(sid, override)
    return s


def test_freeze_pins_active_sessions_to_current_model(store):
    _make(store, "active-no-override")
    _make(store, "active-with-override", override="anthropic:sonnet")
    _make(store, "finished-no-override", status=SessionStatus.COMPLETED.value)
    _make(store, "other-agent", agent="agent-b")

    store.freeze_session_models_to_current("agent-a", "anthropic:opus")

    assert store.get_model_override("active-no-override") == "anthropic:opus", (
        "an active session without an override must be pinned to the agent's old model so it doesn't quietly switch"
    )
    assert store.get_model_override("active-with-override") == "anthropic:sonnet", (
        "an explicit override must not be overwritten by the freeze"
    )
    assert store.get_model_override("finished-no-override") is None, (
        "finished sessions don't take more turns, so no need to pin them — leave alone"
    )
    assert store.get_model_override("other-agent") is None, (
        "other agents' sessions must not be touched by a model change on agent-a"
    )


def test_freeze_does_nothing_when_current_model_is_none(store):
    """If the agent had no model set, there's nothing to freeze to."""
    _make(store, "s1")
    store.freeze_session_models_to_current("agent-a", None)
    assert store.get_model_override("s1") is None


def test_freeze_skips_superseded_sessions(store):
    """Sessions that have been compacted-into a successor shouldn't be touched —
    the successor is the one that will actually run.
    """
    _make(store, "predecessor")
    _make(store, "successor")
    store._sessions["predecessor"].superseded_by = "successor"

    store.freeze_session_models_to_current("agent-a", "anthropic:opus")

    # Predecessor is effectively done — the live session is the successor.
    # The successor (still active, no override) gets pinned.
    assert store.get_model_override("predecessor") is None
    assert store.get_model_override("successor") == "anthropic:opus"
