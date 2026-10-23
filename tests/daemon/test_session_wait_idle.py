import asyncio

import pytest
from tsugite_daemon.session_runner import SessionRunner
from tsugite_daemon.session_store import Session, SessionStatus, SessionStore


class _Adapter:
    def __init__(self):
        self.release = asyncio.Event()
        self.error = None

    async def handle_message(self, user_id, message, channel_context, custom_logger=None):
        await self.release.wait()
        if self.error:
            raise self.error
        return "done"


@pytest.fixture
def store(tmp_path):
    store = SessionStore(tmp_path / "state" / "store.json")
    store.agents.ensure_default(tmp_path / "main", "default")
    return store


@pytest.fixture
def adapter():
    return _Adapter()


@pytest.fixture
def runner(store, adapter):
    return SessionRunner(store, adapter)


@pytest.fixture
def chat(store):
    return store.create_session(Session(id="", user_id="u"))


async def _waiting(runner, session_id):
    waiter = asyncio.create_task(runner.wait_idle(session_id))
    await asyncio.sleep(0.01)
    assert not waiter.done()
    return waiter


async def _woke(waiter):
    await asyncio.wait_for(waiter, 1)


async def _background(runner, adapter, error=None):
    adapter.error = error
    session = runner.start_session(Session(id="", prompt="go", title="bg"))
    await asyncio.sleep(0)
    return session.id


@pytest.mark.asyncio
async def test_idle_session_returns_at_once(runner, chat):
    await asyncio.wait_for(runner.wait_idle(chat.id), 0.1)


@pytest.mark.asyncio
async def test_wakes_when_an_interactive_turn_ends(runner, store, chat):
    store.begin_turn(chat.id)
    waiter = await _waiting(runner, chat.id)

    store.end_turn(chat.id)

    await _woke(waiter)


@pytest.mark.asyncio
async def test_wakes_when_compaction_hands_the_turn_on(runner, store, chat):
    store.begin_turn(chat.id)
    waiter = await _waiting(runner, chat.id)

    store.end_turn(chat.id, notify_listeners=False)

    await _woke(waiter)


@pytest.mark.parametrize("error", [None, RuntimeError("boom")])
@pytest.mark.asyncio
async def test_wakes_when_a_background_run_ends(runner, store, adapter, error):
    session_id = await _background(runner, adapter, error)
    waiter = await _waiting(runner, session_id)

    adapter.release.set()

    await _woke(waiter)
    assert store.get_session(session_id).status != SessionStatus.RUNNING.value


@pytest.mark.asyncio
async def test_wakes_when_a_background_run_is_cancelled(runner, adapter):
    session_id = await _background(runner, adapter)
    waiter = await _waiting(runner, session_id)

    runner.cancel_session(session_id)

    await _woke(waiter)


@pytest.mark.asyncio
async def test_wakes_when_the_queue_drains(runner, store, chat):
    store.queue_message(chat.id, {"id": "q-1", "text": "hi"})
    release = asyncio.Event()

    async def sender(session_id, entry):
        await release.wait()
        return True

    runner.set_queued_message_sender(sender)
    runner.drain_queued_messages(chat.id)
    waiter = await _waiting(runner, chat.id)

    release.set()

    await _woke(waiter)


@pytest.mark.asyncio
async def test_wakes_when_the_last_queued_message_is_dropped(runner, store, chat):
    store.queue_message(chat.id, {"id": "q-1", "text": "hi"})
    waiter = await _waiting(runner, chat.id)

    store.drop_queued_message(chat.id, "q-1")

    await _woke(waiter)


@pytest.mark.asyncio
async def test_concurrent_waiters_all_wake(runner, store, chat):
    store.begin_turn(chat.id)
    first = await _waiting(runner, chat.id)
    second = await _waiting(runner, chat.id)

    store.end_turn(chat.id)

    await _woke(first)
    await _woke(second)


@pytest.mark.asyncio
async def test_wakes_when_compaction_moves_the_turn_to_the_successor(runner, store, chat):
    store.begin_turn(chat.id)
    waiter = await _waiting(runner, chat.id)

    store.compact_session(chat.id)

    await _woke(waiter)
