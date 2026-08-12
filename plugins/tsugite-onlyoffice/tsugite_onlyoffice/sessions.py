"""Coordinating an agent's edits with whatever the document server is doing.

The adapter's HTTP routes stay a thin dispatcher over this, and the turn protocol
stays unit-testable without standing up a document server.

Two calling conventions meet here. `open_document` and `edit_document` are sync
and run on a tool executor thread; everything else is async and runs on the daemon
loop, because only that loop hears the callback a save arrives as.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Awaitable, Callable

from tsugite.tools import call_on_loop
from tsugite_onlyoffice.config import OnlyOfficeConfig
from tsugite_onlyoffice.documents import canonical, document_key, resolve_existing
from tsugite_onlyoffice.docx import Document

if TYPE_CHECKING:
    from tsugite_onlyoffice.command_service import CommandClient

logger = logging.getLogger(__name__)

# How long a run of edits has to stay quiet before the announce goes out. Long
# enough that back-to-back tool calls collapse into one swap, short enough that a
# single edit still feels immediate next to the turn that produced it.
ANNOUNCE_DEBOUNCE = 0.75

# How long a requested save has to come back as a callback. It is a round trip
# through the document server plus a download, on a document a human is typing in.
SAVE_TIMEOUT = 30.0

# A whole turn is that save plus the edit itself, waited on from a tool thread.
TURN_TIMEOUT = SAVE_TIMEOUT + 60.0


def _apply_edit(path: Path, work: Callable[[Document], object]) -> object:
    document = Document.open(path)
    result = work(document)
    document.save()
    return result


@dataclass
class _DocumentState:
    """What the daemon knows about one document's editing sessions.

    Attributes:
        relative: The document this is about, in the one spelling every caller's
            path is reduced to. Everything keyed on the document is keyed on it,
            the document key included.
        key: The document key the last config handed out, which is the key a new
            editor joins on and the key a command names the session by.
        live_keys: The keys sessions currently have the document open under. A
            version swap runs two of them at once, for as long as the document
            server takes to close the one an agent turn moved past, so a single
            flag cannot say who is in there: the close of the old session would
            declare the one that replaced it dead. The callback is the only place
            any of this can be learned.
        generation: How many sessions on this document have already ended. Keys
            are minted with it, so one an ended session retired cannot come back.
        waiters: Reads parked on the next save of this document.
        lock: Held for a whole agent turn, so two turns cannot rotate the key
            out from under each other.
        announce_handle: The pending trailing-edge announce, if a run of edits is
            still going. Cancelled and re-armed by each edit, so a bulk operation
            costs the reader one swap rather than one per edit.
    """

    relative: str
    key: str | None = None
    live_keys: set[str | None] = field(default_factory=set)
    generation: int = 0
    waiters: list[asyncio.Future] = field(default_factory=list)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    announce_handle: asyncio.TimerHandle | None = None

    @property
    def live(self) -> bool:
        """Whether any session has the document open."""
        return bool(self.live_keys)


class DocumentSessions:
    """The editing sessions one document server is running, and the turns over them."""

    def __init__(
        self,
        config: OnlyOfficeConfig,
        commands: Callable[[], CommandClient],
        announce: Callable[[str, str], None],
    ):
        self.config = config
        self.loop: asyncio.AbstractEventLoop | None = None
        self._commands = commands
        self._announce = announce
        self._documents: dict[str, _DocumentState] = {}

    # ── session state ──

    def _state(self, relative: str) -> _DocumentState:
        """The state for a document, created on first mention and kept afterwards.

        Canonicalized here, which is the one seam every caller's path passes
        through, so two spellings of one file cannot end up with a lock and a
        liveness record each.

        Kept, rather than reclaimed once the session ends, because the generation
        is the only record that a key was ever retired: forget it and the next
        config mints that same key straight back.

        Raises:
            OutsideDocumentsError: The path resolves outside the documents directory.
        """
        name = canonical(self.config.documents_dir, relative)
        if name not in self._documents:
            self._documents[name] = _DocumentState(name)
        return self._documents[name]

    def open_key(self, relative: str, path: Path) -> str:
        """The key a new editor should open this document under, remembered as issued.

        A live session keeps the key it opened with, so a second tab joins it rather
        than starting a second session on bytes the first one is still holding, and
        a force-save moving the file underneath does not split them.

        Anything else mints, because the document server will not reopen a key whose
        session has ended, and a session can end without touching the file at all.
        Re-deriving one of those hands the tab a key that cannot load, and so does
        every reopen after it.
        """
        state = self._state(relative)
        if not (state.live and state.key):
            state.key = document_key(state.relative, path, state.generation)
        return state.key

    def session_started(self, relative: str, key: str | None) -> None:
        """Note that somebody has the document open, under the key they opened it on.

        The key is only adopted as the current one when there is nothing on record,
        which is what a restart leaves behind. A tab that opened before an agent
        turn rotated the key can report itself live after it, and adopting that key
        would send the next reader back to the bytes the edit replaced.
        """
        state = self._state(relative)
        state.live_keys.add(key)
        if state.key is None:
            state.key = key

    def session_ended(self, relative: str, key: str | None) -> None:
        """Retire one session's key, leaving any other session on the document open.

        The generation moves on every end, whether or not that key was one we had
        on record, because its whole job is that a key a session retired can never
        be minted again.
        """
        state = self._state(relative)
        state.live_keys.discard(key)
        state.generation += 1

    def _schedule_announce(self, state: _DocumentState) -> None:
        """Announce once a run of edits has stopped, rather than once per edit.

        A bulk operation - resolving every comment, say - is many turns back to
        back, and announcing each one swaps the document out from under whoever is
        reading it, repeatedly. Re-arming a trailing-edge timer collapses the run
        into the single swap the reader wants, and because the timer reads the key
        when it fires rather than when it was armed, that swap lands on whatever
        the last edit left behind.
        """
        if state.announce_handle is not None:
            state.announce_handle.cancel()
        state.announce_handle = asyncio.get_running_loop().call_later(ANNOUNCE_DEBOUNCE, self._announce_now, state)

    def _announce_now(self, state: _DocumentState) -> None:
        """Fire the trailing-edge announce for a document whose edits have settled."""
        state.announce_handle = None
        self._announce(state.relative, state.key)

    def cancel_pending_announces(self) -> None:
        """Drop every armed announce, for a daemon that is going away."""
        for state in self._documents.values():
            if state.announce_handle is not None:
                state.announce_handle.cancel()
                state.announce_handle = None

    def is_current_key(self, relative: str, key: str | None) -> bool:
        """Whether a callback's key is the one this document is currently on.

        A session that outlives an agent turn keeps editing bytes the turn has
        already replaced, so its save has to be refused rather than written back
        over the edit. Anything we cannot compare against - no state, no key on
        record after a restart, no key in the body - is trusted, because refusing
        those would refuse the genuine save that follows a restart.
        """
        state = self._documents.get(canonical(self.config.documents_dir, relative))
        if state is None or not state.key or not key:
            return True
        return key == state.key

    def deliver(self, relative: str) -> None:
        """Tell whatever is waiting on this document that its save landed."""
        for waiter in self._state(relative).waiters:
            if not waiter.done():
                waiter.set_result(None)

    # ── the runtime the @onlyoffice tools call ──

    @property
    def author(self) -> str:
        """The name an agent's comments and replies are written under."""
        return self.config.agent_name

    def document_path(self, relative: str) -> Path:
        """Resolve a tool-supplied path to a document that exists.

        Raises:
            ValueError: The path escapes the documents directory, or there is no
                such document.
        """
        return resolve_existing(self.config.documents_dir, relative)

    def _turn_loop(self) -> asyncio.AbstractEventLoop:
        """The daemon loop a turn runs on.

        Raises:
            RuntimeError: The adapter has not started, so there is no loop to
                coordinate the turn on and no way to hear a save land.
        """
        if self.loop is None:
            raise RuntimeError("the onlyoffice adapter is not running, so it cannot reach the document server")
        return self.loop

    def open_document(self, relative: str) -> Document:
        """Open a document for reading, taking a live session's unsaved typing with it."""
        return call_on_loop(self._turn_loop(), self.read_turn, relative, timeout=TURN_TIMEOUT)

    def edit_document(self, relative: str, work: Callable[[Document], object]) -> object:
        """Apply an edit to a document and write it back."""
        return call_on_loop(self._turn_loop(), self.agent_turn, relative, work, timeout=TURN_TIMEOUT)

    # ── the turn ──

    async def read_turn(self, relative: str) -> Document:
        """Read a document, force-saving first when somebody has it open."""
        path = self.document_path(relative)
        state = self._state(relative)
        async with state.lock:
            if state.live:
                await self.read_live(relative)
            # Unzipping every part of a large document is hundreds of milliseconds
            # in which the daemon would answer nobody, SSE subscribers included.
            return await asyncio.to_thread(Document.open, path)

    async def agent_turn(self, relative: str, work: Callable[[Document], object]) -> object:
        """Apply an agent's edit around whatever the document server is doing.

        The session is force-saved rather than ended, so it survives the turn and
        the page swaps versions in place instead of rebuilding the editor. The
        force-save lands everything typed so far, so nothing is lost; what it
        leaves behind is a session holding pre-edit bytes, which is what
        `is_current_key` in the callback handler exists to make harmless.

        Raises:
            RuntimeError: The force-saved session's save never came back.
        """
        path = self.document_path(relative)
        state = self._state(relative)
        async with state.lock:
            # There is nothing to force-save from a session that opened after this
            # line, so the save is decided on what was open when the turn started
            # and the swap on what is open when it lands.
            if state.live:
                key = self._session_key(path, state)
                await self._await_save(relative, state, self._commands().forcesave(key))
            result = await asyncio.to_thread(_apply_edit, path, work)
            if state.live:
                state.key = document_key(state.relative, path, state.generation)
                self._schedule_announce(state)
            else:
                # The counterpart to `_announce`'s log: same symptom, other cause.
                logger.info("onlyoffice edited %s with no live session to tell", relative)
            return result

    async def read_live(self, relative: str) -> None:
        """Bring what a live editing session holds, keystrokes included, onto disk.

        A forced save is the only way to see typing the document server has not
        written back yet, and its result arrives as a callback rather than as an
        answer, so the read parks until that callback lands and writes the file.
        The session survives it, which is why a read does not have to end one.

        Raises:
            RuntimeError: No save came back in time.
        """
        path = self.document_path(relative)
        state = self._state(relative)
        key = self._session_key(path, state)
        await self._await_save(relative, state, self._commands().forcesave(key))

    def _session_key(self, path: Path, state: _DocumentState) -> str:
        """The key to name the live session by in a command.

        A status 1 carries the key its session opened on and `session_started`
        adopts it, so the only session left with nothing on record is one whose
        callback named no key at all. Re-deriving reproduces what it opened with,
        as long as nobody saved since and it opened at the generation on record.
        """
        return state.key or document_key(state.relative, path, state.generation)

    async def _await_save(self, relative: str, state: _DocumentState, command: Awaitable[bool]) -> None:
        """Issue a command and park until the save it triggers reaches the callback.

        Raises:
            RuntimeError: No save came back in time.
        """
        waiter = asyncio.get_running_loop().create_future()
        # Parked before the command goes out, or a callback that beats this line
        # resolves nothing and the wait below runs to its timeout.
        state.waiters.append(waiter)
        try:
            if not await command:
                # The document server had nothing to do, so no callback is coming
                # and the file on disk is already everything the session held.
                return
            await asyncio.wait_for(waiter, SAVE_TIMEOUT)
        except asyncio.TimeoutError:
            raise RuntimeError(
                f"onlyoffice asked the document server to force-save {relative}, and no save came back within "
                f"{SAVE_TIMEOUT:.0f}s. Check that the document is open in an editor and that /health reports "
                "the document server as reachable."
            ) from None
        finally:
            state.waiters.remove(waiter)
