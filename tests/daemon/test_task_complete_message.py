"""The <background_task_complete> card survives the round trip into a session.

When the originating session is idle, the card is delivered through
reply_to_session -> handle_message, so it is stored as the session's user_input
text and the web UI peels it back off with split_injected_context. A result
carrying the closing tag used to cut the card short, spilling the remainder into
the chat as though the user had typed it.
"""

import xml.etree.ElementTree as ET

from tsugite_daemon.adapters.scheduler_adapter import build_task_complete_message

from tsugite.history.ui_events import split_injected_context


def test_card_is_well_formed_with_awkward_result():
    msg = build_task_complete_message("task-1", 0, "check A & B", "found <div> & 3 items")

    parsed = ET.fromstring(msg)
    assert parsed.get("id") == "task-1"
    assert parsed.findtext("prompt") == "check A & B"
    assert parsed.findtext("result").strip() == "found <div> & 3 items"


def test_result_containing_the_closing_tag_does_not_truncate_the_card():
    result = "the runtime closes it with </background_task_complete> at the end"
    msg = build_task_complete_message("task-2", 1, "explain the format", result)

    blocks, rest = split_injected_context(msg)

    assert len(blocks) == 1
    assert blocks[0]["tag"] == "background_task_complete"
    assert blocks[0]["id"] == "task-2"
    # Nothing leaks past the card into what renders as the user's own words.
    assert rest == ""


def test_long_prompt_is_summarized():
    msg = build_task_complete_message("task-3", 0, "x" * 300, "done")
    assert "…" in ET.fromstring(msg).findtext("prompt")
    assert "x" * 300 not in msg
