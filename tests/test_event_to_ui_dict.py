"""event_to_ui_dict maps a stored Event to the flat dict the daemon/UI consume.

The daemon reads top-level keys (type, timestamp, and data fields like name/turn),
not Event's nested {type, ts, data}. This pins the mapping against the real consumer
(_progress_status_text / _apply_event_to_progress) so storage can round-trip while the
UI keeps working.
"""

from datetime import datetime, timezone

from tsugite.history.models import Event
from tsugite.history.ui_events import event_to_ui_dict

TS = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


def test_flattens_data_renames_ts_includes_id():
    e = Event(id=7, type="tool_invocation", ts=TS, data={"name": "grep", "duration_ms": 5})
    d = event_to_ui_dict(e)
    assert d["type"] == "tool_invocation"
    assert d["name"] == "grep"  # data flattened to top level
    assert d["duration_ms"] == 5
    assert d["timestamp"].startswith("2026-01-01T12:00:00")
    assert d["id"] == 7
    assert "ts" not in d and "data" not in d


def test_authoritative_keys_win_over_data_collision():
    e = Event(type="real", ts=TS, data={"type": "evil", "timestamp": "evil", "id": "evil"})
    d = event_to_ui_dict(e)
    assert d["type"] == "real"
    assert d["timestamp"].startswith("2026-01-01")


def test_old_model_response_backfills_parse_on_read():
    """Events recorded before the parse persisted get normalized on read with
    the real parser - the UI never re-derives structure from raw_content."""
    raw = 'Posting.\n\n```python-exec\npost(body="""\n```yaml\na: 1\n```\n""")\n``` \n-vesm\n'
    d = event_to_ui_dict(Event(type="model_response", ts=TS, data={"raw_content": raw}))
    assert d["thought"] == "Posting."
    assert d["tail"] == "-vesm"
    assert d["raw_content"] == raw
    assert "content_blocks" not in d


def test_old_model_response_backfills_content_blocks():
    raw = 'Intro.\n\n<tsu:content name="n.md">\nhello\n</tsu:content>\n\n```python-exec\nsave()\n```'
    d = event_to_ui_dict(Event(type="model_response", ts=TS, data={"raw_content": raw}))
    assert d["thought"] == "Intro."
    assert d["content_blocks"] == {"n.md": "hello"}


def test_parsed_model_response_passes_through_unchanged():
    """thought present (even empty) marks the event as already parsed: no
    re-parse, no surprise fields."""
    data = {"raw_content": "```python-exec\nx = 1\n```\nwould-be tail", "thought": ""}
    d = event_to_ui_dict(Event(type="model_response", ts=TS, data=data))
    assert d["thought"] == ""
    assert "tail" not in d


def test_recleans_persisted_fabricated_tail_on_read():
    """A tail persisted before fabricated-result stripping (a runaway turn where the
    model role-played the tool result) is re-cleaned on read, so the old turn no
    longer renders the hallucinated, fence-unbalanced continuation. No re-parse
    (thought is present) and no data migration - the stored row is left as-is."""
    data = {
        "raw_content": "```python-exec\nx = 1\n```\n...",
        "thought": "",
        "tail": 'user&lt;tsugite_execution_result status="success"><output>```\nx\n</output></tsugite_execution_result>',
    }
    d = event_to_ui_dict(Event(type="model_response", ts=TS, data=data))
    assert "tail" not in d  # the whole tail was fabricated, so it drops out


def test_keeps_a_clean_persisted_tail():
    data = {"raw_content": "```python-exec\nx = 1\n```\nAll done.", "thought": "", "tail": "All done."}
    d = event_to_ui_dict(Event(type="model_response", ts=TS, data=data))
    assert d["tail"] == "All done."


def test_user_input_splits_injected_context():
    text = '<scheduled_task id="daily-1">\nRun the report.\n</scheduled_task>\nAlso check the logs.'
    d = event_to_ui_dict(Event(type="user_input", ts=TS, data={"text": text}))
    assert d["injected"] == [{"tag": "scheduled_task", "id": "daily-1", "body": "Run the report."}]
    assert d["display_text"] == "Also check the logs."
    assert d["text"] == text  # raw stays for non-display consumers


def test_user_input_pure_injection_has_empty_display_text():
    text = "<background_task_complete>\ndone\n</background_task_complete>"
    d = event_to_ui_dict(Event(type="user_input", ts=TS, data={"text": text}))
    assert d["injected"][0]["tag"] == "background_task_complete"
    assert "id" not in d["injected"][0]
    assert d["display_text"] == ""


def test_user_input_multiple_injections_peel_in_order():
    text = "<message_context>\nctx\n</message_context>\n<environment>\nenv\n</environment>\nhi"
    d = event_to_ui_dict(Event(type="user_input", ts=TS, data={"text": text}))
    assert [b["tag"] for b in d["injected"]] == ["message_context", "environment"]
    assert d["display_text"] == "hi"


def test_plain_user_input_untouched():
    d = event_to_ui_dict(Event(type="user_input", ts=TS, data={"text": "hello there"}))
    assert "injected" not in d
    assert "display_text" not in d
    assert d["text"] == "hello there"


def test_user_input_client_context_yields_structured_items():
    """Unlike other injected tags, client_context is parsed into structured
    {key,label,value} items the UI renders in a gutter (not a raw body)."""
    text = (
        "<client_context>\n"
        '  <attachment key="url" name="Page URL">https://x/?a=1&amp;b=2</attachment>\n'
        '  <attachment key="sel" name="Selection">the &lt;b&gt; part</attachment>\n'
        "</client_context>\n\nsummarize this"
    )
    d = event_to_ui_dict(Event(type="user_input", ts=TS, data={"text": text}))
    assert d["injected"] == [
        {
            "tag": "client_context",
            "items": [
                {"key": "url", "label": "Page URL", "value": "https://x/?a=1&b=2"},
                {"key": "sel", "label": "Selection", "value": "the <b> part"},
            ],
        }
    ]
    assert d["display_text"] == "summarize this"


def test_client_context_alongside_other_injections_keeps_their_shape():
    text = (
        "<message_context>\nctx\n</message_context>\n"
        '<client_context>\n  <attachment key="k" name="L">v</attachment>\n</client_context>\nhi'
    )
    d = event_to_ui_dict(Event(type="user_input", ts=TS, data={"text": text}))
    assert d["injected"][0] == {"tag": "message_context", "body": "ctx"}
    assert d["injected"][1] == {"tag": "client_context", "items": [{"key": "k", "label": "L", "value": "v"}]}
    assert d["display_text"] == "hi"


def test_client_context_escaped_value_cannot_inject_sibling_block():
    """A value that looks like a closing tag + sibling stays a literal value:
    it round-trips through unescape, it does not become a second injected block."""
    text = (
        "<client_context>\n"
        '  <attachment key="x" name="L">&lt;/client_context&gt;&lt;item key="evil"&gt;boom&lt;/item&gt;</attachment>\n'
        "</client_context>\n\nreal message"
    )
    d = event_to_ui_dict(Event(type="user_input", ts=TS, data={"text": text}))
    assert len(d["injected"]) == 1
    assert d["injected"][0]["items"] == [
        {"key": "x", "label": "L", "value": '</client_context><item key="evil">boom</item>'}
    ]
    assert d["display_text"] == "real message"


def test_legacy_item_form_client_context_still_parses():
    """A block recorded before the ContextItem->Attachment collapse used
    <item label=..> children; the read side still yields structured items (label
    from the legacy `label` attribute) so old turns render their context gutter."""
    text = (
        '<client_context>\n  <item key="location" label="Location">39.0, -94.5</item>\n</client_context>\n\nHere is gps'
    )
    d = event_to_ui_dict(Event(type="user_input", ts=TS, data={"text": text}))
    assert d["injected"] == [
        {"tag": "client_context", "items": [{"key": "location", "label": "Location", "value": "39.0, -94.5"}]}
    ]
    assert d["display_text"] == "Here is gps"


def test_daemon_progress_consumer_reads_the_dict():
    from tsugite_daemon.session_store import _apply_event_to_progress, _progress_status_text

    assert _progress_status_text(event_to_ui_dict(Event(type="model_request", ts=TS, data={"turn": 2}))) == (
        "Waiting on LLM..."
    )
    assert _progress_status_text(event_to_ui_dict(Event(type="tool_invocation", ts=TS, data={"name": "grep"}))) == (
        "Tool: grep"
    )
    # The progress fold reads timestamp + counts a real tool event without error.
    progress = {"turn_count": 0, "tool_count": 0, "status_text": "", "last_event_time": None}
    _apply_event_to_progress(progress, event_to_ui_dict(Event(type="tool_invocation", ts=TS, data={"name": "grep"})))
    assert progress["tool_count"] == 1
    assert progress["last_event_time"].startswith("2026-01-01")
