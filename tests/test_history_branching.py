"""#400 branching: fork a conversation at a point into an independent continuation.

The source session is untouched; the branch contains the source events up to the cut,
with provider session state scrubbed so it starts a fresh subprocess instead of
--resume-ing into the source's. Lineage lives in dedicated tree columns, never the
linear compaction chain (superseded_by).
"""

import pytest

from tsugite.history.sqlite_backend import SqliteHistoryBackend
from tsugite.history.sqlite_conn import close_all


@pytest.fixture
def backend(tmp_path):
    b = SqliteHistoryBackend(db_path=tmp_path / "history.db")
    yield b
    close_all()


def _seed(backend):
    s = backend.create("chat", "claude_code:opus")
    s.record("user_input", text="first question")
    s.record("model_response", raw_content="first answer", state_delta={"session_id": "PROVIDER-123"})
    s.record("user_input", text="second question")
    s.record("model_response", raw_content="second answer", state_delta={"session_id": "PROVIDER-456"})
    return s


def _event_at(session, type_, occurrence=0):
    matches = [e for e in session.load_events() if e.type == type_]
    return matches[occurrence]


def test_branch_copies_head_and_leaves_source_untouched(backend):
    src = _seed(backend)
    cut = _event_at(src, "model_response", 0)  # branch right after the first answer
    before = [(e.type, e.data) for e in src.load_events()]

    new_id = backend.create_branch(src.session_id, at_event_id=cut.id)
    branch = backend.load(new_id)

    btypes = [e.type for e in branch.load_events()]
    assert btypes == ["session_start", "user_input", "model_response"]  # head only, up to the cut
    # source is unchanged
    assert [(e.type, e.data) for e in src.load_events()] == before


def test_branch_scrubs_provider_state_delta(backend):
    src = _seed(backend)
    cut = _event_at(src, "model_response", 0)
    new_id = backend.create_branch(src.session_id, at_event_id=cut.id)

    branch_mr = _event_at(backend.load(new_id), "model_response", 0)
    assert "state_delta" not in branch_mr.data  # fresh provider session, no --resume
    # source still carries its provider state
    assert _event_at(src, "model_response", 0).data["state_delta"] == {"session_id": "PROVIDER-123"}


def test_branch_lineage_columns_and_fresh_ids(backend):
    src = _seed(backend)
    cut = _event_at(src, "user_input", 1)
    src_ids = {e.id for e in src.load_events()}

    new_id = backend.create_branch(src.session_id, at_event_id=cut.id)
    row = (
        backend._conn()
        .execute(
            "SELECT branched_from_session_id, branch_point_event_id FROM sessions WHERE session_id=?",
            (new_id,),
        )
        .fetchone()
    )
    assert row["branched_from_session_id"] == src.session_id
    assert row["branch_point_event_id"] == cut.id

    branch_ids = {e.id for e in backend.load(new_id).load_events()}
    assert src_ids.isdisjoint(branch_ids)  # copied events got fresh destination rowids


def test_branch_continues_independently(backend):
    src = _seed(backend)
    cut = _event_at(src, "model_response", 0)
    new_id = backend.create_branch(src.session_id, at_event_id=cut.id)

    backend.load(new_id).record("user_input", text="branch-only follow up")
    src.record("user_input", text="source-only follow up")

    branch_texts = [e.data.get("text") for e in backend.load(new_id).load_events() if e.type == "user_input"]
    assert "branch-only follow up" in branch_texts
    assert "source-only follow up" not in branch_texts


def test_branch_rejects_foreign_or_session_start_cut(backend):
    src = _seed(backend)
    other = backend.create("chat", "m")
    foreign_id = other.load_events()[0].id
    with pytest.raises(ValueError):
        backend.create_branch(src.session_id, at_event_id=foreign_id)  # belongs to another session

    start_id = _event_at(src, "session_start", 0).id
    with pytest.raises(ValueError):
        backend.create_branch(src.session_id, at_event_id=start_id)  # nothing to continue
