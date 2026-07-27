"""List-style agent tools must project a bounded summary, never dump per-item bulk.

schedule_list once reused the daemon's full serializer and dragged every
schedule's run_history into the listing; at scale that blew past the exec-output
cap and got truncated mid-object into a malformed blob (see
test_schedule_list_trim). The other list tools already hand-build a lean
projection, but their backing records DO carry unbounded free-form fields
(Session.scratchpad/result/metadata, a conversation's full assistant response).
The cheapest way to reintroduce the bug is to "simplify" one of these to
asdict(record) / **summary. These guards lock the lean shape so that regression
fails loudly; full detail stays in session_status(id) / read_conversation(id).
"""

import json
from types import SimpleNamespace

from tsugite_daemon.session_store import Session, SessionSource, SessionStatus

import tsugite.tools.sessions as sessions_tool
from tsugite.history import get_history_backend
from tsugite.tools.history import list_conversations


def _wire_sessions(monkeypatch, sessions):
    store = SimpleNamespace(list_sessions=lambda **kw: list(sessions))
    monkeypatch.setattr(sessions_tool, "_session_runner", SimpleNamespace(store=store))
    monkeypatch.setattr(sessions_tool, "_call", lambda fn, *a, timeout=30, **k: fn(*a, **k))


def test_list_sessions_omits_heavy_fields_but_keeps_summary(monkeypatch):
    session = Session(
        id="sess-1",
        agent="bot",
        source=SessionSource.BACKGROUND.value,
        status=SessionStatus.COMPLETED.value,
        prompt="p" * 500,
        title="nightly run",
    )
    session.scratchpad = "s" * 10_000
    session.result = "r" * 10_000
    session.metadata = {"blob": "m" * 10_000}

    _wire_sessions(monkeypatch, [session])
    [row] = sessions_tool.list_sessions()

    assert set(row.keys()) == {
        "id",
        "agent",
        "source",
        "status",
        "title",
        "prompt",
        "created_at",
        "parent_id",
    }, "list view must expose only the lean projection, not the full Session dict"
    for heavy in ("scratchpad", "result", "metadata"):
        assert heavy not in row, f"list view must not dump per-session {heavy}"
    assert len(row["prompt"]) == 200, "prompt must be truncated to 200 chars"
    assert row["status"] == SessionStatus.COMPLETED.value
    assert row["title"] == "nightly run"


def test_list_conversations_omits_full_response_but_keeps_summary():
    backend = get_history_backend()
    session = backend.create(agent_name="bot", model="test:model")
    sid = session.session_id
    session.record("user_input", text="hello")
    session.record("model_response", raw_content="R" * 5_000, usage={"total_tokens": 10})
    session.record("session_end", status="success")

    [row] = list_conversations()

    assert set(row.keys()) == {
        "conversation_id",
        "agent",
        "model",
        "created_at",
        "turn_count",
        "total_tokens",
        "total_cost",
        "status",
        "duration_ms",
        "functions_used",
    }, "list view must expose only the summary, not per-turn transcript"
    assert "last_response" not in row, "list view must not dump the full assistant response"
    assert "R" * 5_000 not in json.dumps(row), "the full response body must not ride the list view"
    assert row["conversation_id"] == sid
    assert row["turn_count"] == 1
    assert row["status"] == "success"
