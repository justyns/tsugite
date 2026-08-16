"""An agent editing a document somebody else has open.

The contract is an ordering: force-save the session, take what was typed, write
it, edit, rotate the key, tell the page once the run of edits settles. Most of
what follows records every step into one list and asserts on that list, because
the order is the whole contract.
"""

import asyncio
import json

import httpx
import pytest
import pytest_asyncio
import tsugite_onlyoffice.sessions as sessions_module
from onlyoffice_helpers import (
    DOWNLOAD_URL,
    PLAIN_DOCUMENT,
    FakeCommands,
    build_docx,
    callback_body,
    post_callback,
    serve_downloads,
)
from tsugite_onlyoffice import tools
from tsugite_onlyoffice.command_service import NO_SUCH_SESSION, NOTHING_TO_SAVE
from tsugite_onlyoffice.documents import OutsideDocumentsError, document_key
from tsugite_onlyoffice.docx import Document

DOCUMENT = "review.docx"
# The same document as DOCUMENT, spelled the way an agent that has just listed a
# directory spells it.
SPELLING = "./review.docx"
OTHER = "reports/q1.docx"
LIVE_KEY = "keyfromtheeditorsession"
OTHER_KEY = "keyfromtheothersession"


class FakeEventBus:
    """Records what the adapter broadcast, in place of the SSE broadcaster."""

    def __init__(self, recorded):
        self.recorded = recorded

    def emit(self, event_type, data=None):
        self.recorded.append(("event", {"type": event_type, **(data or {})}))


def delivering(adapter):
    """A command answer that resolves the save without any of it reaching disk."""

    async def answer(_command, key):
        # Which document is on this key, following a turn that rotated it.
        relative = next(name for name, state in adapter.sessions._documents.items() if state.key == key)
        adapter.sessions.deliver(relative)

    return answer


def answering(recorded, http_server, relative, status, url=DOWNLOAD_URL):
    """A command answer that POSTs the callback the document server would send."""

    async def answer(_command, key):
        recorded.append(("callback", status))
        await post_callback(http_server, relative, callback_body(status, key, url=url))

    return answer


def make_live(adapter, relative, key):
    """Put a document into the state a `status 1` callback would have left it in."""
    adapter.sessions.session_started(relative, key)


def recording_work(recorded, label="edit", text="Reviewed."):
    """A unit of agent work that records that it ran."""

    def work(document):
        recorded.append((label, document.path.name))
        document.insert(1, f" {text}")

    return work


def steps(recorded, without=("event",)):
    return [step for step, _detail in recorded if step not in without]


def command_service(recorded, *codes):
    """A transport that answers each CommandService post with the next error code.

    The real `CommandClient` runs over this, rather than `FakeCommands`, for the
    tests where the code the document server spends is the thing under test.
    """
    answers = iter(codes)

    def handle(request):
        recorded.append(("command", json.loads(request.content)["key"]))
        return httpx.Response(200, json={"error": next(answers)})

    return handle


async def announced(recorded, timeout=2.0):
    """Wait out the trailing-edge announce and return every swap it broadcast.

    A test that read `recorded` straight after its turn would see the run of edits
    still in progress rather than the one swap it settles into.
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not any(step == "event" for step, _detail in recorded):
        if loop.time() > deadline:
            raise AssertionError(f"no announce arrived within {timeout}s")
        await asyncio.sleep(0.005)
    # A second swap would land one more debounce out, and one swap per run is the point.
    await asyncio.sleep(sessions_module.ANNOUNCE_DEBOUNCE * 2)
    return [detail for step, detail in recorded if step == "event"]


@pytest.fixture
def fast_announce(monkeypatch):
    """Shorten the announce debounce, so a test waits on the swap and not on the clock.

    Only for tests whose turns are separated by a wait of their own. A run of turns
    back to back has to pin the debounce out of reach instead, or the run's own speed
    becomes part of the assertion.
    """
    monkeypatch.setattr(sessions_module, "ANNOUNCE_DEBOUNCE", 0.05)


@pytest.fixture
def turn(adapter, documents_dir, http_server):
    """An adapter with two real documents, a fake CommandService and a fake bus.

    Ordered after `http_server` on purpose: attaching the plugin's routes is what
    hands the adapter the daemon's real broadcaster, and it would overwrite the
    fake bus if it ran second.
    """
    build_docx(documents_dir / DOCUMENT, PLAIN_DOCUMENT)
    recorded = []
    adapter.commands = FakeCommands(calls=recorded, answer=delivering(adapter))
    adapter.event_bus = FakeEventBus(recorded)
    return recorded


# ── the order ──


@pytest.mark.asyncio
async def test_the_turn_edits_what_was_typed_and_not_what_was_on_disk(
    turn, adapter, http_server, documents_dir, typed_bytes
):
    """Editing bytes the session has not handed back yet is how keystrokes get lost."""
    make_live(adapter, DOCUMENT, LIVE_KEY)

    serve_downloads(adapter, typed_bytes)
    adapter.commands.answer = answering(turn, http_server, DOCUMENT, 2)

    await adapter.sessions.agent_turn(DOCUMENT, recording_work(turn))
    text = Document.open(documents_dir / DOCUMENT).text()
    assert "typed while the session was open Reviewed." in text


@pytest.mark.asyncio
async def test_a_session_that_closes_with_nothing_to_save_releases_the_parked_turn(
    turn, adapter, http_server, documents_dir
):
    """The human closes the tab while the turn is parked on its force-save.

    That closes with nothing typed since the last save, so status 4 arrives with no
    payload to fetch, and a turn that only treats a written-back save as the end of
    the wait stalls until it times out.
    """
    make_live(adapter, DOCUMENT, LIVE_KEY)
    adapter.commands.answer = answering(turn, http_server, DOCUMENT, 4, url=None)

    await adapter.sessions.agent_turn(DOCUMENT, recording_work(turn))

    assert steps(turn, without=()) == ["forcesave", "callback", "edit"]
    assert "Reviewed." in Document.open(documents_dir / DOCUMENT).text()


@pytest.mark.asyncio
async def test_a_turn_whose_session_is_already_gone_still_edits(turn, adapter, documents_dir, monkeypatch):
    """The daemon believes a session is live until a callback says otherwise, and a restart
    or a missed callback leaves that belief behind. The force-save then reports no such
    session, which leaves the file on disk already current, so it is an answer and not a
    failure."""
    monkeypatch.setattr(sessions_module, "SAVE_TIMEOUT", 0.05)
    make_live(adapter, DOCUMENT, LIVE_KEY)
    adapter.commands = FakeCommands(calls=turn, nothing_to_do={"forcesave"})

    await adapter.sessions.agent_turn(DOCUMENT, recording_work(turn))
    assert "Reviewed." in Document.open(documents_dir / DOCUMENT).text()


@pytest.mark.asyncio
async def test_a_turn_on_a_document_nobody_has_open_just_edits_it(turn, adapter):
    """No session means nothing to end, wait for, or refresh."""
    await adapter.sessions.agent_turn(DOCUMENT, recording_work(turn))
    assert steps(turn, without=()) == ["edit"]


# ── the key and the event ──


@pytest.mark.asyncio
async def test_the_key_rotates_off_the_one_the_live_session_is_holding(turn, adapter, documents_dir):
    """The document server caches by key, so the new bytes need a key of their own before the
    page can be sent to them. The live session stays on the one it opened with."""
    make_live(adapter, DOCUMENT, LIVE_KEY)
    await adapter.sessions.agent_turn(DOCUMENT, recording_work(turn))

    state = adapter.sessions._state(DOCUMENT)
    assert state.key != LIVE_KEY
    assert state.key == document_key(DOCUMENT, documents_dir / DOCUMENT, state.generation)


@pytest.mark.asyncio
async def test_the_event_names_the_document_and_the_key_the_page_should_load(
    turn, adapter, client, headers, fast_announce
):
    """The page swaps onto the key the event names, so it has to be the one /config now hands out."""
    make_live(adapter, DOCUMENT, LIVE_KEY)
    await adapter.sessions.agent_turn(DOCUMENT, recording_work(turn))

    (event,) = await announced(turn)
    issued = client.get(f"/api/plugins/onlyoffice/config?path={DOCUMENT}", headers=headers).json()
    assert event["path"] == DOCUMENT
    assert event["key"] == issued["config"]["document"]["key"]


@pytest.mark.asyncio
async def test_a_run_of_edits_announces_one_swap_and_not_one_per_edit(turn, adapter, monkeypatch):
    """Resolving every comment on a document is a run of turns back to back, and a swap per
    turn takes the document out from under whoever is reading it, once per edit."""
    # A debounce no run can outlast, then the trailing edge fired by hand. Racing a
    # short one against 17 real turns makes the run's own speed part of the assertion:
    # any pause longer than the debounce - a slow disk, a GC, a loaded CI box - lets
    # the timer fire mid-run and the test reads it as a coalescing failure.
    monkeypatch.setattr(sessions_module, "ANNOUNCE_DEBOUNCE", 30)
    make_live(adapter, DOCUMENT, LIVE_KEY)
    for number in range(17):
        await adapter.sessions.agent_turn(DOCUMENT, recording_work(turn, f"edit-{number}"))

    state = adapter.sessions._state(DOCUMENT)
    assert state.announce_handle is not None, "the run of edits left no announce armed"
    state.announce_handle.cancel()
    adapter.sessions._announce_now(state)

    events = [detail for step, detail in turn if step == "event"]
    assert len(events) == 1, f"{len(events)} swaps for one run of edits"
    assert events[0]["key"] == state.key, "the swap lands on what the run left behind"


@pytest.mark.asyncio
async def test_stopping_the_adapter_drops_an_announce_that_has_not_fired_yet(turn, adapter, fast_announce):
    """A timer that outlives the adapter swaps the page onto a key nothing is left to serve."""
    await adapter.start()
    make_live(adapter, DOCUMENT, LIVE_KEY)
    await adapter.sessions.agent_turn(DOCUMENT, recording_work(turn))
    assert adapter.sessions._state(DOCUMENT).announce_handle is not None

    await adapter.stop()

    assert adapter.sessions._state(DOCUMENT).announce_handle is None
    await asyncio.sleep(sessions_module.ANNOUNCE_DEBOUNCE * 3)
    assert steps(turn, without=()) == ["forcesave", "edit"], "the announce went out anyway"


@pytest.mark.asyncio
async def test_a_second_edit_after_the_key_rotated_is_not_a_failure(turn, adapter, documents_dir):
    """The first turn rotates the key; the live editor is still on the one before it. The second
    turn's force-save therefore names a key no session is on, which the document server spends an
    error code on and which holds nothing the file on disk does not already have."""
    make_live(adapter, DOCUMENT, LIVE_KEY)
    # The real CommandClient, so the error codes go through the client that reads them.
    adapter.commands = None
    serve_downloads(adapter, command_service(turn, NOTHING_TO_SAVE, NO_SUCH_SESSION))

    await adapter.sessions.agent_turn(DOCUMENT, recording_work(turn, "first", "One."))
    await adapter.sessions.agent_turn(DOCUMENT, recording_work(turn, "second", "Two."))

    keys = [detail for step, detail in turn if step == "command"]
    assert keys[0] == LIVE_KEY
    assert keys[1] != LIVE_KEY, "the second turn force-saves the key the first one rotated to"
    text = Document.open(documents_dir / DOCUMENT).text()
    assert "One." in text and "Two." in text


# ── two sessions at once ──


@pytest.mark.asyncio
async def test_the_session_a_turn_moved_past_closing_does_not_silence_the_one_that_replaced_it(
    turn, adapter, http_server, fast_announce
):
    """A version swap is two sessions at once, for as long as the old one takes to close.

    The tab reopens on the key the turn announced while the document server is still
    holding the session that turn moved past. That close arrives seconds later, and a
    document that can only remember one session reads it as the document going quiet:
    the next turn then force-saves nothing, rotates nothing and announces nothing, and
    the session nobody retired saves its pre-edit bytes over the edit.
    """
    make_live(adapter, DOCUMENT, LIVE_KEY)
    await adapter.sessions.agent_turn(DOCUMENT, recording_work(turn, "first", "One."))
    (first_swap,) = await announced(turn)
    rotated = first_swap["key"]

    await post_callback(http_server, DOCUMENT, callback_body(1, rotated, url=None))
    await post_callback(http_server, DOCUMENT, callback_body(2, LIVE_KEY))

    await adapter.sessions.agent_turn(DOCUMENT, recording_work(turn, "second", "Two."))

    state = adapter.sessions._state(DOCUMENT)
    assert steps(turn) == ["forcesave", "first", "forcesave", "second"], "the second turn skipped its force-save"
    assert state.key != rotated, "the second turn left the tab on the key it just replaced"
    swaps = await announced(turn)
    assert [swap["key"] for swap in swaps] == [rotated, state.key], "the second turn never told the tab"


@pytest.mark.asyncio
async def test_a_download_already_in_flight_cannot_land_on_top_of_the_turn_that_overtook_it(
    turn, adapter, http_server, documents_dir, typed_bytes
):
    """A save is a download the callback holds no lock across, and a whole turn fits inside it.

    The session's own save is fetched before the turn starts and written after it
    finishes, so a handler that decides once, on the way in, whether those bytes are
    still current writes the pre-edit ones back over the edit.
    """
    make_live(adapter, DOCUMENT, LIVE_KEY)
    fetching, release = asyncio.Event(), asyncio.Event()

    async def parked(_request):
        fetching.set()
        await release.wait()
        return httpx.Response(200, content=typed_bytes)

    serve_downloads(adapter, parked)
    # The turn's own force-save finds nothing left to save, because the save this
    # test parks is the one the session had already started.
    adapter.commands = FakeCommands(calls=turn, nothing_to_do={"forcesave"})
    saving = asyncio.create_task(post_callback(http_server, DOCUMENT, callback_body(6, LIVE_KEY)))
    await asyncio.wait_for(fetching.wait(), 2)

    await adapter.sessions.agent_turn(DOCUMENT, recording_work(turn))
    release.set()
    await saving

    assert "Reviewed." in Document.open(documents_dir / DOCUMENT).text()


@pytest.mark.asyncio
async def test_a_session_that_opens_mid_turn_is_still_told_where_the_new_bytes_are(turn, adapter, fast_announce):
    """The turn reads liveness at the top and the human opens the document after that.

    There was no session to force-save when it started, but by the time it lands there
    is a tab holding the bytes it replaced, and the announce is the only thing that
    moves that tab off them.
    """

    def work(document):
        turn.append(("edit", document.path.name))
        adapter.sessions.session_started(DOCUMENT, LIVE_KEY)
        document.insert(1, " Reviewed.")

    await adapter.sessions.agent_turn(DOCUMENT, work)

    (swap,) = await announced(turn)
    assert swap["key"] == adapter.sessions._state(DOCUMENT).key
    assert steps(turn) == ["edit"], "there was nothing to force-save when the turn started"


@pytest.mark.asyncio
async def test_a_close_on_a_superseded_key_retires_that_key_and_leaves_the_live_one(turn, adapter, http_server):
    make_live(adapter, DOCUMENT, LIVE_KEY)
    await adapter.sessions.agent_turn(DOCUMENT, recording_work(turn))
    rotated = adapter.sessions._state(DOCUMENT).key
    await post_callback(http_server, DOCUMENT, callback_body(1, rotated, url=None))

    state = adapter.sessions._state(DOCUMENT)
    generation = state.generation
    await post_callback(http_server, DOCUMENT, callback_body(2, LIVE_KEY))

    assert state.live_keys == {rotated}
    assert state.generation == generation + 1, "the closed session's key stays retired"


@pytest.mark.asyncio
async def test_a_turn_whose_session_named_no_key_force_saves_the_one_it_can_re_derive(turn, adapter, documents_dir):
    """A callback that names no session leaves nothing to force-save by.

    Deriving the key from the file reproduces what the session opened with, as long
    as nobody saved in between and the tab opened at the generation on record.
    """
    adapter.sessions.session_started(DOCUMENT, None)
    assert adapter.sessions._state(DOCUMENT).key is None
    # What the session opened on, before the turn's own edit moves it.
    opened_on = document_key(DOCUMENT, documents_dir / DOCUMENT, 0)

    # `delivering` finds the document by the key on record, which this document
    # does not have.
    async def answer(_command, _key):
        adapter.sessions.deliver(DOCUMENT)

    adapter.commands.answer = answer
    await adapter.sessions.agent_turn(DOCUMENT, recording_work(turn))

    (saved,) = [detail for step, detail in turn if step == "forcesave"]
    assert saved == opened_on


# ── knowing whether anyone is in there ──


@pytest.mark.asyncio
async def test_the_callback_is_what_tells_the_daemon_a_document_is_open(adapter, http_server, typed_bytes):
    serve_downloads(adapter, typed_bytes)
    await post_callback(http_server, "notes.docx", callback_body(1, LIVE_KEY))
    assert adapter.sessions._state("notes.docx").live is True
    await post_callback(http_server, "notes.docx", callback_body(2, LIVE_KEY))
    assert adapter.sessions._state("notes.docx").live is False


def test_a_status_1_adopts_the_key_it_names_when_there_is_none_on_record(adapter):
    """A restart forgets the key it handed out; the callback from that tab still carries it."""
    adapter.sessions.session_started(DOCUMENT, LIVE_KEY)
    assert adapter.sessions._state(DOCUMENT).key == LIVE_KEY


def test_a_status_1_on_an_older_key_does_not_rewind_the_key_a_turn_rotated_to(adapter):
    """The tab that opened before the rotation reports itself live after it, and adopting
    that key would send the next reader back to the bytes the edit replaced."""
    rotated = "the-key-the-turn-rotated-to"
    adapter.sessions.session_started(DOCUMENT, LIVE_KEY)
    adapter.sessions._state(DOCUMENT).key = rotated

    adapter.sessions.session_started(DOCUMENT, LIVE_KEY)
    assert adapter.sessions._state(DOCUMENT).key == rotated


@pytest.mark.asyncio
async def test_a_session_closed_with_no_changes_also_clears_the_flag(adapter, http_server):
    await post_callback(http_server, "notes.docx", callback_body(1, LIVE_KEY))
    await post_callback(http_server, "notes.docx", callback_body(4, LIVE_KEY, url=None))
    assert adapter.sessions._state("notes.docx").live is False


# ── serialization ──


@pytest.mark.asyncio
async def test_two_turns_on_one_document_do_not_interleave(turn, adapter):
    """Interleaved turns would each rotate the key the other is about to hand out."""
    make_live(adapter, DOCUMENT, LIVE_KEY)
    await asyncio.gather(
        adapter.sessions.agent_turn(DOCUMENT, recording_work(turn, "edit-a")),
        adapter.sessions.agent_turn(DOCUMENT, recording_work(turn, "edit-b")),
    )
    assert steps(turn) in (
        ["forcesave", "edit-a", "forcesave", "edit-b"],
        ["forcesave", "edit-b", "forcesave", "edit-a"],
    )


@pytest.mark.asyncio
async def test_two_turns_on_one_document_do_not_interleave_under_two_spellings(turn, adapter):
    """`doc_read("./review.docx")` and `doc_replace("review.docx")` are two turns on one file.

    Keyed on whatever the caller typed, they take a lock each and neither sees the
    session the other found: they interleave over the same bytes, and the one that
    spelled it the other way force-saves nothing before editing them.
    """
    make_live(adapter, DOCUMENT, LIVE_KEY)
    await asyncio.gather(
        adapter.sessions.agent_turn(DOCUMENT, recording_work(turn, "edit-a")),
        adapter.sessions.agent_turn(SPELLING, recording_work(turn, "edit-b")),
    )
    assert steps(turn) in (
        ["forcesave", "edit-a", "forcesave", "edit-b"],
        ["forcesave", "edit-b", "forcesave", "edit-a"],
    )


@pytest.mark.asyncio
async def test_turns_on_different_documents_do_not_wait_for_each_other(turn, adapter):
    make_live(adapter, DOCUMENT, LIVE_KEY)
    make_live(adapter, OTHER, OTHER_KEY)
    await asyncio.gather(
        adapter.sessions.agent_turn(DOCUMENT, recording_work(turn, "edit-a")),
        adapter.sessions.agent_turn(OTHER, recording_work(turn, "edit-b")),
    )
    assert steps(turn)[:2] == ["forcesave", "forcesave"], "one document's turn waited for the other's"
    assert sorted(steps(turn)[2:]) == ["edit-a", "edit-b"]


# ── one document, however it is spelled ──


@pytest.mark.asyncio
async def test_a_turn_spelling_the_path_differently_still_sees_the_open_session(turn, adapter):
    """The tab reported itself open under the spelling `/config` was asked for, and the
    agent types its own. A turn that misses that session edits the bytes on disk while
    somebody is still typing into the ones the document server holds."""
    make_live(adapter, DOCUMENT, LIVE_KEY)

    await adapter.sessions.agent_turn(SPELLING, recording_work(turn))

    assert steps(turn, without=()) == ["forcesave", "edit"]


@pytest.mark.asyncio
async def test_a_turn_spelled_another_way_announces_the_path_the_page_opened_on(turn, adapter, fast_announce):
    """A tab takes its path from `GET /docs`, which lists one spelling of each document,
    and drops any event naming another. The agent's own spelling is not the tab's."""
    make_live(adapter, DOCUMENT, LIVE_KEY)
    await adapter.sessions.agent_turn(SPELLING, recording_work(turn))

    (swap,) = await announced(turn)
    assert swap["path"] == DOCUMENT


def test_two_spellings_of_one_document_open_on_one_key(adapter, documents_dir):
    """The document server caches by key, so a second key on one file is the same bytes
    cached twice, and the turn that rotates one of them leaves the other on what it
    replaced."""
    path = documents_dir / "notes.docx"
    keys = {
        adapter.sessions.open_key(spelling, path)
        for spelling in ("notes.docx", "./notes.docx", "reports/../notes.docx")
    }
    assert len(keys) == 1, f"{len(keys)} keys for one document"


def test_a_path_that_escapes_the_documents_dir_gets_no_session_state(adapter):
    """Reading a path as one document rather than three must not read it out of the jail."""
    with pytest.raises(OutsideDocumentsError):
        adapter.sessions._state("../outside.docx")


# ── failure ──


@pytest.mark.asyncio
async def test_a_save_that_never_arrives_times_out_and_changes_nothing(turn, adapter, documents_dir, monkeypatch):
    monkeypatch.setattr(sessions_module, "SAVE_TIMEOUT", 0.05)
    make_live(adapter, DOCUMENT, LIVE_KEY)
    adapter.commands.answer = None
    before = (documents_dir / DOCUMENT).read_bytes()

    with pytest.raises(RuntimeError) as raised:
        await adapter.sessions.agent_turn(DOCUMENT, recording_work(turn))

    assert DOCUMENT in str(raised.value)
    assert (documents_dir / DOCUMENT).read_bytes() == before
    assert steps(turn, without=()) == ["forcesave"]
    assert adapter.sessions._state(DOCUMENT).waiters == []


# ── the tools ──


@pytest_asyncio.fixture
async def started(adapter):
    await adapter.start()
    yield
    await adapter.stop()


@pytest.mark.asyncio
async def test_a_tool_edit_on_an_open_document_goes_through_a_turn(turn, adapter, fast_announce, started):
    make_live(adapter, DOCUMENT, LIVE_KEY)
    result = await asyncio.to_thread(tools.doc_replace, path=DOCUMENT, target="Costs", replacement="Expenses")
    await announced(turn)

    assert result["replaced"] == 1
    assert steps(turn, without=()) == ["forcesave", "event"]


@pytest.mark.asyncio
async def test_a_tool_read_of_an_open_document_takes_the_live_bytes(turn, adapter, http_server, typed_bytes, started):
    """Reading the file on disk would miss everything typed since the last save."""
    make_live(adapter, DOCUMENT, LIVE_KEY)

    serve_downloads(adapter, typed_bytes)
    adapter.commands.answer = answering(turn, http_server, DOCUMENT, 6)
    result = await asyncio.to_thread(tools.doc_read, path=DOCUMENT)

    assert "typed while the session was open" in result["text"]
    assert steps(turn, without=()) == ["forcesave", "callback"]


@pytest.mark.asyncio
async def test_a_tool_read_of_a_document_nobody_has_open_asks_for_nothing(turn, adapter, started):
    result = await asyncio.to_thread(tools.doc_read, path=DOCUMENT)

    assert "Quarterly review" in result["text"]
    assert steps(turn, without=()) == []


@pytest.mark.asyncio
async def test_a_tool_edit_a_live_session_never_answers_tells_the_agent_what_to_check(
    turn, adapter, monkeypatch, started
):
    """That message is the agent's whole diagnostic when a session stops answering."""
    monkeypatch.setattr(sessions_module, "SAVE_TIMEOUT", 0.05)
    make_live(adapter, DOCUMENT, LIVE_KEY)
    adapter.commands.answer = None
    result = await asyncio.to_thread(tools.doc_replace, path=DOCUMENT, target="Costs", replacement="Expenses")

    assert DOCUMENT in result["error"]
    assert "/health" in result["error"]


def test_a_tool_edit_before_the_adapter_started_refuses_rather_than_writing(turn, adapter, documents_dir):
    """Writing without the turn coordination is how a live session's keystrokes get lost."""
    before = (documents_dir / DOCUMENT).read_bytes()
    tools.set_onlyoffice_runtime(adapter.sessions)
    try:
        result = tools.doc_replace(path=DOCUMENT, target="pilot", replacement="trial")
    finally:
        tools.set_onlyoffice_runtime(None)

    assert "not running" in result["error"]
    assert (documents_dir / DOCUMENT).read_bytes() == before
