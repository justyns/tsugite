"""Tests for TerminalSessionStore: CRUD, state-machine guards, atomic save, concurrency."""

import threading

import pytest

from tsugite.daemon.terminal_store import (
    TerminalSession,
    TerminalSessionStore,
    TerminalState,
    TerminalStateTransitionError,
)


@pytest.fixture
def store_path(tmp_path):
    return tmp_path / "terminal_sessions.json"


@pytest.fixture
def store(store_path):
    return TerminalSessionStore(store_path)


def _make_terminal(cmd="echo hi", cwd="/tmp", parent_session_id="parent-1"):
    return TerminalSession(id="", cmd=cmd, cwd=cwd, parent_session_id=parent_session_id)


def test_add_assigns_id_and_persists(store, store_path):
    t = store.add(_make_terminal())
    assert t.id.startswith("term-")
    assert store.get(t.id) is t
    assert store_path.exists()


def test_round_trip_via_reload(store_path):
    s1 = TerminalSessionStore(store_path)
    t = s1.add(_make_terminal(cmd="ls -la"))
    s1.update(t.id, pid=1234, bytes_out=42)
    s1.update_state(t.id, TerminalState.RUNNING.value)

    s2 = TerminalSessionStore(store_path)
    loaded = s2.get(t.id)
    assert loaded.cmd == "ls -la"
    assert loaded.pid == 1234
    assert loaded.bytes_out == 42
    assert loaded.state == TerminalState.RUNNING.value


def test_initial_state_is_starting(store):
    t = store.add(_make_terminal())
    assert t.state == TerminalState.STARTING.value


@pytest.mark.parametrize(
    "from_state,to_state",
    [
        (TerminalState.STARTING, TerminalState.RUNNING),
        (TerminalState.STARTING, TerminalState.FAILED),
        (TerminalState.STARTING, TerminalState.CANCELLED),
        (TerminalState.RUNNING, TerminalState.SUCCEEDED),
        (TerminalState.RUNNING, TerminalState.FAILED),
        (TerminalState.RUNNING, TerminalState.CANCELLED),
        (TerminalState.RUNNING, TerminalState.TIMED_OUT),
        (TerminalState.RUNNING, TerminalState.STREAM_LOST),
    ],
)
def test_valid_transitions(store, from_state, to_state):
    t = store.add(_make_terminal())
    if from_state is not TerminalState.STARTING:
        _force_state(store, t.id, from_state.value)
    updated = store.update_state(t.id, to_state.value)
    assert updated.state == to_state.value
    assert updated.updated_at >= t.created_at


@pytest.mark.parametrize(
    "from_state,to_state",
    [
        (TerminalState.STARTING, TerminalState.SUCCEEDED),
        (TerminalState.SUCCEEDED, TerminalState.RUNNING),
        (TerminalState.FAILED, TerminalState.RUNNING),
        (TerminalState.CANCELLED, TerminalState.RUNNING),
        (TerminalState.TIMED_OUT, TerminalState.RUNNING),
        (TerminalState.STREAM_LOST, TerminalState.RUNNING),
        (TerminalState.SUCCEEDED, TerminalState.FAILED),
    ],
)
def test_invalid_transition_raises(store, from_state, to_state):
    t = store.add(_make_terminal())
    if from_state is not TerminalState.STARTING:
        _force_state(store, t.id, from_state.value)
    with pytest.raises(TerminalStateTransitionError):
        store.update_state(t.id, to_state.value)


def test_terminal_states_are_terminal(store):
    for terminal in (
        TerminalState.SUCCEEDED,
        TerminalState.FAILED,
        TerminalState.CANCELLED,
        TerminalState.TIMED_OUT,
        TerminalState.STREAM_LOST,
    ):
        t = store.add(_make_terminal())
        _force_state(store, t.id, terminal.value)
        for target in (TerminalState.RUNNING, TerminalState.STARTING):
            with pytest.raises(TerminalStateTransitionError):
                store.update_state(t.id, target.value)


def test_resolved_at_set_on_terminal_transition(store):
    t = store.add(_make_terminal())
    store.update_state(t.id, TerminalState.RUNNING.value)
    assert store.get(t.id).resolved_at is None
    store.update_state(t.id, TerminalState.SUCCEEDED.value)
    assert store.get(t.id).resolved_at is not None


def test_list_active_excludes_terminal(store):
    a = store.add(_make_terminal(cmd="a"))
    b = store.add(_make_terminal(cmd="b"))
    _force_state(store, b.id, TerminalState.SUCCEEDED.value)
    c = store.add(_make_terminal(cmd="c"))
    store.update_state(c.id, TerminalState.RUNNING.value)
    ids = {t.id for t in store.list_active()}
    assert a.id in ids
    assert c.id in ids
    assert b.id not in ids


def test_list_for_parent_filters_correctly(store):
    a = store.add(_make_terminal(parent_session_id="p1"))
    b = store.add(_make_terminal(parent_session_id="p1"))
    c = store.add(_make_terminal(parent_session_id="p2"))
    p1 = {t.id for t in store.list_for_parent("p1")}
    assert p1 == {a.id, b.id}
    assert {t.id for t in store.list_for_parent("p2")} == {c.id}


def test_list_all_returns_every_terminal(store):
    a = store.add(_make_terminal(cmd="a"))
    b = store.add(_make_terminal(cmd="b"))
    _force_state(store, b.id, TerminalState.SUCCEEDED.value)
    ids = {t.id for t in store.list_all()}
    assert ids == {a.id, b.id}


def test_update_persists_arbitrary_fields(store, store_path):
    t = store.add(_make_terminal())
    store.update(t.id, pid=999, bytes_out=128, lines_out=4, last_line="ok", exit_code=0)
    reloaded = TerminalSessionStore(store_path).get(t.id)
    assert reloaded.pid == 999
    assert reloaded.bytes_out == 128
    assert reloaded.lines_out == 4
    assert reloaded.last_line == "ok"
    assert reloaded.exit_code == 0


def test_update_unknown_field_raises(store):
    t = store.add(_make_terminal())
    with pytest.raises(ValueError):
        store.update(t.id, totally_made_up_field=1)


def test_update_unknown_terminal_raises(store):
    with pytest.raises(KeyError):
        store.update("term-does-not-exist", pid=1)


def test_update_state_unknown_terminal_raises(store):
    with pytest.raises(KeyError):
        store.update_state("term-nope", TerminalState.RUNNING.value)


def test_concurrent_state_updates_serialise(store):
    t = store.add(_make_terminal())
    store.update_state(t.id, TerminalState.RUNNING.value)
    barrier = threading.Barrier(8)
    successes = []
    failures = []

    def attempt():
        barrier.wait()
        try:
            store.update_state(t.id, TerminalState.SUCCEEDED.value)
            successes.append(1)
        except TerminalStateTransitionError:
            failures.append(1)

    threads = [threading.Thread(target=attempt) for _ in range(8)]
    for t_ in threads:
        t_.start()
    for t_ in threads:
        t_.join()

    assert len(successes) == 1
    assert len(failures) == 7
    assert store.get(t.id).state == TerminalState.SUCCEEDED.value


def test_atomic_save_via_tmpfile(store, store_path, tmp_path):
    """The store writes via a .tmp sibling and os.replace, so a half-written file
    never replaces the canonical JSON."""
    store.add(_make_terminal(cmd="atomic test"))
    # The .tmp file should not linger after a successful save.
    assert not (store_path.with_suffix(".tmp")).exists()
    assert store_path.exists()
    import json

    data = json.loads(store_path.read_text())
    assert "terminals" in data
    assert len(data["terminals"]) == 1


def _force_state(store, terminal_id, target_state):
    """Walk the state machine via valid transitions to reach target_state."""
    paths = {
        TerminalState.RUNNING.value: [TerminalState.RUNNING.value],
        TerminalState.SUCCEEDED.value: [TerminalState.RUNNING.value, TerminalState.SUCCEEDED.value],
        TerminalState.FAILED.value: [TerminalState.FAILED.value],
        TerminalState.CANCELLED.value: [TerminalState.CANCELLED.value],
        TerminalState.TIMED_OUT.value: [TerminalState.RUNNING.value, TerminalState.TIMED_OUT.value],
        TerminalState.STREAM_LOST.value: [TerminalState.RUNNING.value, TerminalState.STREAM_LOST.value],
    }
    for step in paths[target_state]:
        store.update_state(terminal_id, step)
