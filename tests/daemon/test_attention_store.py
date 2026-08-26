"""Needs-attention records: one per waiting thing, durable across a restart."""

import pytest
from tsugite_daemon.attention_store import (
    OWNER_SESSION,
    SOURCE_ASK,
    SOURCE_DELIVERY,
    SOURCE_JOB,
    AttentionStore,
)


@pytest.fixture
def store(tmp_path):
    return AttentionStore(tmp_path / "attention.json")


def _open(store, owner_id="s1", source=SOURCE_ASK, ref_id="ask-1", kind="needs_answer", **kw):
    return store.open(
        owner_kind=OWNER_SESSION,
        owner_id=owner_id,
        source=source,
        ref_id=ref_id,
        kind=kind,
        **kw,
    )


def test_an_open_record_makes_its_owner_need_the_user(store):
    record = _open(store)

    assert [r.id for r in store.open_records("s1")] == [record.id]


def test_a_session_with_nothing_open_has_no_records(store):
    assert store.open_records("s1") == []


def test_reporting_the_same_ref_twice_opens_one_record(store):
    first = _open(store)

    assert _open(store) is None
    assert [r.id for r in store.open_records("s1")] == [first.id]


def test_two_sources_on_one_owner_are_two_records(store):
    _open(store, source=SOURCE_ASK, ref_id="ask-1")
    _open(store, source=SOURCE_DELIVERY, ref_id="del-1", kind="needs_ack")

    assert len(store.open_records("s1")) == 2


def test_the_same_ref_id_from_two_sources_stays_distinct(store):
    a = _open(store, source=SOURCE_ASK, ref_id="x1")
    b = _open(store, source=SOURCE_JOB, ref_id="x1", kind="stuck")

    assert a.id != b.id
    assert len(store.open_records("s1")) == 2


def test_clearing_a_record_removes_it_from_the_open_set(store):
    record = _open(store)

    assert [r.id for r in store.clear_ref(SOURCE_ASK, "ask-1")] == [record.id]
    assert store.open_records("s1") == []


def test_clearing_an_unknown_ref_is_not_an_error(store):
    assert store.clear_ref(SOURCE_ASK, "nope") == []


def test_clearing_by_ref_clears_without_knowing_the_record_id(store):
    """A job leaving `stuck` knows its own id, never the attention record's."""
    _open(store, source=SOURCE_JOB, ref_id="job-7", kind="stuck")

    cleared = store.clear_ref(SOURCE_JOB, "job-7")

    assert len(cleared) == 1
    assert store.open_records("s1") == []


def test_clearing_by_ref_leaves_other_sources_alone(store):
    _open(store, source=SOURCE_JOB, ref_id="job-7", kind="stuck")
    _open(store, source=SOURCE_ASK, ref_id="ask-1")

    store.clear_ref(SOURCE_JOB, "job-7")

    assert [r.source for r in store.open_records("s1")] == [SOURCE_ASK]


def test_clearing_an_owner_clears_everything_it_holds(store):
    _open(store, source=SOURCE_ASK, ref_id="ask-1")
    _open(store, source=SOURCE_DELIVERY, ref_id="del-1", kind="needs_ack")

    cleared = store.clear_owner("s1")

    assert len(cleared) == 2
    assert store.open_records("s1") == []


def test_clearing_an_owner_can_be_narrowed_to_one_source(store):
    """Boot recovery drops asks whose blocked call died with the daemon, and must
    not touch the deliveries that outlive it."""
    _open(store, source=SOURCE_ASK, ref_id="ask-1")
    _open(store, source=SOURCE_DELIVERY, ref_id="del-1", kind="needs_ack")

    store.clear_owner("s1", source=SOURCE_ASK)

    assert [r.source for r in store.open_records("s1")] == [SOURCE_DELIVERY]


def test_records_are_scoped_per_owner(store):
    _open(store, owner_id="s1")
    _open(store, owner_id="s2", ref_id="ask-2")

    assert len(store.open_records("s1")) == 1
    assert len(store.open_records("s2")) == 1


def test_open_records_without_an_owner_spans_every_owner(store):
    _open(store, owner_id="s1")
    _open(store, owner_id="s2", ref_id="ask-2")

    assert len(store.open_records()) == 2


def test_reopening_a_cleared_ref_opens_a_new_record(store):
    """A job that goes stuck, is retried, and goes stuck again needs the user twice."""
    first = _open(store, source=SOURCE_JOB, ref_id="job-7", kind="stuck")
    store.clear_ref(SOURCE_JOB, "job-7")

    second = _open(store, source=SOURCE_JOB, ref_id="job-7", kind="stuck")

    assert second.id != first.id
    assert [r.id for r in store.open_records("s1")] == [second.id]


def test_open_records_survive_a_restart(tmp_path):
    path = tmp_path / "attention.json"
    first = AttentionStore(path)
    record = first.open(
        owner_kind=OWNER_SESSION,
        owner_id="s1",
        source=SOURCE_DELIVERY,
        ref_id="del-1",
        kind="needs_ack",
    )

    reopened = AttentionStore(path)

    survivors = reopened.open_records("s1")
    assert [r.id for r in survivors] == [record.id]
    assert survivors[0].source == SOURCE_DELIVERY


def test_a_clear_survives_a_restart(tmp_path):
    path = tmp_path / "attention.json"
    first = AttentionStore(path)
    first.open(
        owner_kind=OWNER_SESSION,
        owner_id="s1",
        source=SOURCE_DELIVERY,
        ref_id="del-2",
        kind="needs_ack",
    )
    first.clear_ref(SOURCE_DELIVERY, "del-2")

    assert AttentionStore(path).open_records("s1") == []


def test_clearing_stale_asks_closes_only_asks(store):
    _open(store, source=SOURCE_ASK, ref_id="ask-1")
    _open(store, source=SOURCE_DELIVERY, ref_id="del-1", kind="needs_ack")

    cleared = store.clear_stale_asks()

    assert [r.source for r in cleared] == [SOURCE_ASK]
    assert [r.source for r in store.open_records("s1")] == [SOURCE_DELIVERY]
