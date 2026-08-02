"""Seed and read history sessions through the backend.

Replaces the per-file `SessionStorage.create(session_path=...)` seeding these
tests used when JSONL was a selectable backend. Sessions now go through
`get_history_backend()`, so the tests exercise the backend that actually ships.
"""


def seed_history_session(session_id, *, agent="test", model="m", parent_session=None, events=()):
    """Create a history session and record `events` into it.

    Replaces the `SessionStorage.create(session_path=...)` seeding the tests used
    when JSONL was a backend. `events` is an iterable of (type, data) pairs.
    """
    from tsugite.history import get_history_backend

    storage = get_history_backend().create(
        agent_name=agent,
        model=model,
        session_id=session_id,
        parent_session=parent_session,
    )
    for event_type, data in events:
        storage.record(event_type, **data)
    return storage


def load_history_session(session_id):
    """Read a seeded session back through the backend."""
    from tsugite.history import get_history_backend

    return get_history_backend().load(session_id)
