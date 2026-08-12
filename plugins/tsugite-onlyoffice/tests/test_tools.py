"""The @onlyoffice agent tools.

These are the only part of the plugin an agent touches, so what they register
as, what they refuse, and what they hand back are asserted here rather than
through the document helpers underneath.
"""

import asyncio
import inspect
import threading

import pytest
from onlyoffice_helpers import PLAIN_DOCUMENT, build_docx
from tsugite_onlyoffice import tools
from tsugite_onlyoffice.docx import Document

import tsugite.tools as registry
from tsugite.tools import expand_tool_specs

TOOL_NAMES = ["doc_comment", "doc_insert", "doc_read", "doc_replace", "doc_reply", "doc_resolve"]
DOCUMENT = "review.docx"

# Enough to satisfy every tool's signature; each call takes the subset it declares.
ARGUMENTS = {"anchor": "Costs held flat.", "text": "note", "target": "pilot", "replacement": "trial", "comment_id": "1"}


def call(tools, name, **overrides):
    """Call a tool by name, filling in whichever arguments it declares."""
    fn = getattr(tools, name)
    supplied = {**ARGUMENTS, **overrides}
    return fn(**{key: value for key, value in supplied.items() if key in inspect.signature(fn).parameters})


@pytest.fixture
def daemon_tools():
    """Register the daemon-only tools, and put the registry back afterwards."""
    registry.set_daemon_mode(True)
    try:
        yield registry
    finally:
        registry.set_daemon_mode(False)


@pytest.fixture
def wired(adapter, documents_dir):
    """The tools with a started adapter behind them and a real docx to work on.

    A tool runs on an executor thread and hands the turn to the daemon's loop, so
    the fixture puts a loop in a thread of its own and starts the adapter on it.
    That is the only arrangement the tools support, and it is the daemon's.
    """
    build_docx(documents_dir / DOCUMENT, PLAIN_DOCUMENT)
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    asyncio.run_coroutine_threadsafe(adapter.start(), loop).result()
    try:
        yield tools
    finally:
        asyncio.run_coroutine_threadsafe(adapter.stop(), loop).result()
        loop.call_soon_threadsafe(loop.stop)
        thread.join()
        loop.close()


# ── registration ──


def test_the_tools_register_under_the_onlyoffice_category(daemon_tools):
    assert daemon_tools.get_tools_by_category("onlyoffice") == TOOL_NAMES


def test_the_tools_are_daemon_only():
    """Outside the daemon there is no runtime to reach, so they must not be offered."""
    assert not set(TOOL_NAMES) & set(registry.list_tools())


def test_a_strict_agent_may_reference_the_category_where_the_plugin_is_absent():
    """@onlyoffice expands to nothing rather than raising when nothing registered it."""
    assert expand_tool_specs(["@onlyoffice"], strict=True) == []


def test_the_adapter_wires_the_runtime_on_start_and_drops_it_on_stop(adapter):
    assert not tools.runtime_available()
    asyncio.run(adapter.start())
    try:
        assert tools.runtime_available()
    finally:
        asyncio.run(adapter.stop())
    assert not tools.runtime_available()


# ── refusals ──


@pytest.mark.parametrize("name", TOOL_NAMES)
def test_a_tool_called_outside_daemon_mode_says_so(name):
    assert "not available" in call(tools, name, path=DOCUMENT)["error"]


@pytest.mark.parametrize("name", TOOL_NAMES)
def test_every_tool_refuses_a_path_outside_the_documents_dir(wired, name):
    assert "escapes the documents directory" in call(wired, name, path="../outside.docx")["error"]


def test_a_missing_document_is_reported_as_an_error(wired):
    assert "no such document" in wired.doc_read(path="absent.docx")["error"]


def test_an_anchor_that_does_not_appear_is_reported_as_an_error(wired):
    result = wired.doc_insert(path=DOCUMENT, anchor="no such sentence", text="x")
    assert "does not appear" in result["error"]


# ── reading ──


def test_doc_read_numbers_the_paragraphs_so_they_can_be_used_as_anchors(wired):
    lines = wired.doc_read(path=DOCUMENT)["text"].splitlines()
    assert lines[:2] == ["1. Quarterly review", "2. The pilot shipped on time."]


def test_doc_read_carries_no_comments_for_a_document_that_has_none(wired):
    assert wired.doc_read(path=DOCUMENT)["comments"] == []


# ── editing ──


def test_doc_insert_lands_the_text_next_to_its_anchor(wired):
    assert wired.doc_insert(path=DOCUMENT, anchor="The pilot shipped on time.", text=" Twice.")["inserted"]
    assert "2. The pilot shipped on time. Twice." in wired.doc_read(path=DOCUMENT)["text"]


def test_a_numbered_anchor_selects_that_paragraph(wired):
    wired.doc_insert(path=DOCUMENT, anchor="3", text=" Barely.")
    assert "3. Costs held flat. Barely." in wired.doc_read(path=DOCUMENT)["text"]


def test_doc_replace_reports_how_many_occurrences_it_changed(wired):
    assert wired.doc_replace(path=DOCUMENT, target="pilot", replacement="trial")["replaced"] == 1
    assert "The trial shipped on time." in wired.doc_read(path=DOCUMENT)["text"]


def test_an_edit_is_on_disk_for_the_next_reader(wired, documents_dir):
    """The document server reads the file, not this process, so the save has to have happened."""
    wired.doc_replace(path=DOCUMENT, target="Costs held flat.", replacement="Costs rose slightly.")
    assert "Costs rose slightly." in Document.open(documents_dir / DOCUMENT).text()


# ── comments ──


def test_a_comment_thread_reads_back_with_its_structure(wired):
    first = wired.doc_comment(path=DOCUMENT, anchor="Costs held flat.", text="Where is the number?")
    reply = wired.doc_reply(path=DOCUMENT, comment_id=first["comment_id"], text="Added in the appendix.")
    wired.doc_resolve(path=DOCUMENT, comment_id=first["comment_id"])

    comments = wired.doc_read(path=DOCUMENT)["comments"]
    assert [c["id"] for c in comments] == [first["comment_id"], reply["comment_id"]]
    assert comments[1]["parent"] == first["comment_id"]
    assert comments[0]["parent"] is None
    assert comments[0]["anchor"] == "Costs held flat."
    assert comments[0]["text"] == "Where is the number?"
    assert comments[0]["resolved"] is True


def test_comments_are_authored_as_the_agent(wired, adapter):
    wired.doc_comment(path=DOCUMENT, anchor="1", text="Title looks fine.")
    (comment,) = wired.doc_read(path=DOCUMENT)["comments"]
    assert comment["author"] == adapter.config.agent_name


def test_replying_to_an_unknown_comment_is_reported_as_an_error(wired):
    assert "no comment with id" in wired.doc_reply(path=DOCUMENT, comment_id="99", text="hi")["error"]
