"""Reconstruct LLM-bound messages from a session event log.

The agent loop appends events as they happen. Before each model call we walk
the log and build the messages array the provider expects. Raw `model_response`
text is sent back verbatim (no re-rendering from parsed pieces) so parser bugs
can't corrupt what the model sees as its own past output.
"""

import re
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional
from xml.sax.saxutils import escape

from tsugite.renderer import format_prompt_ts, parse_iso_utc

from .models import Event

# Pre-#479 history stored an executed turn with a bare ```python fence. Replaying it
# verbatim makes the model imitate that fence on its next turn, where it now no-ops
# (bare blocks aren't executed). When an old model_response was actually executed, we
# promote its (first) bare fence to ```python-exec so the replayed context matches the
# current convention. The negative lookahead leaves already-migrated turns untouched.
_LEGACY_EXEC_FENCE = re.compile(r"(?m)^```python(?!-exec)([ \t]*\r?\n)")


def _promote_exec_fence(raw: str) -> str:
    return _LEGACY_EXEC_FENCE.sub(r"```python-exec\1", raw, count=1)


def _response_was_executed(events: List[Event], idx: int) -> bool:
    """True if the model_response at ``idx`` is followed by a code_execution before the
    next turn boundary (a later model_response / user_input)."""
    for event in events[idx + 1 :]:
        if event.type == "code_execution":
            return True
        if event.type in ("model_response", "user_input"):
            return False
    return False


def _format_event_ts(ts: Optional[datetime]) -> str:
    """Format an event timestamp as `YYYY-MM-DD HH:MM TZ` for prompt prefixes.

    Stays absolute (no relative phrase) so prefixes on past messages remain
    byte-stable across turns and the prompt cache keeps hitting.
    """
    return format_prompt_ts(ts) if ts else ""


def last_index_of(events: List[Event], type_: str) -> Optional[int]:
    """Index of the last event of `type_`, or None if absent."""
    for i in range(len(events) - 1, -1, -1):
        if events[i].type == type_:
            return i
    return None


def events_to_messages(events: Iterable[Event]) -> List[Dict[str, Any]]:
    """Walk events and return the full message history to send next.

    Session-owning providers that hold their own prior conversation trim this down
    themselves; they receive the full history via ``set_context``.
    """
    events = list(events)

    cutoff = last_index_of(events, "compaction")
    post_compaction = events[cutoff + 1 :] if cutoff is not None else events

    messages: List[Dict[str, Any]] = []
    if cutoff is not None:
        messages.append({"role": "user", "content": _compaction_user_block(events[cutoff])})
        messages.append(
            {"role": "assistant", "content": "I've reviewed our previous conversation and I'm ready to continue."}
        )

    for idx, event in enumerate(post_compaction):
        executed = event.type == "model_response" and _response_was_executed(post_compaction, idx)
        rendered = _event_to_message(event, executed=executed)
        if rendered:
            messages.append(rendered)

    return messages


def _event_to_message(event: Event, executed: bool = False) -> Optional[Dict[str, Any]]:
    if event.type == "user_input":
        text = event.data.get("text", "")
        ts_str = _format_event_ts(event.ts)
        content = f"[{ts_str}] {text}" if ts_str else text
        return {"role": "user", "content": content}
    if event.type == "model_response":
        raw = event.data.get("raw_content", "")
        if executed:
            raw = _promote_exec_fence(raw)
        return {"role": "assistant", "content": raw}
    if event.type == "code_execution":
        return {"role": "user", "content": _execution_xml(event.data, event.ts)}
    if event.type == "format_error":
        return {"role": "user", "content": _format_error_xml(event.data)}
    return None


def _execution_xml(data: Dict[str, Any], ts: Optional[datetime] = None) -> str:
    """Build the <tsugite_execution_result> envelope from event data."""
    from tsugite.core.executor import MAX_EXECUTION_OUTPUT_KB

    output = data.get("output") or ""
    error = data.get("error")
    duration_ms = data.get("duration_ms")

    # Mirror the live ExecutionResult.to_xml truncation: the full output is stored in the
    # event, but the live turn only showed the first MAX_EXECUTION_OUTPUT_KB. Replaying it
    # in full would re-inflate context and diverge byte-for-byte from what the model saw.
    truncated = False
    max_bytes = MAX_EXECUTION_OUTPUT_KB * 1024
    if len(output) > max_bytes:
        output = output[:max_bytes]
        truncated = True

    status = "error" if error else "success"
    attrs = f'status="{status}"'
    if duration_ms:
        attrs += f' duration_ms="{duration_ms}"'
    if truncated:
        attrs += ' truncated="true"'
    ts_str = _format_event_ts(ts)
    if ts_str:
        attrs += f' ts="{ts_str}"'

    parts = [f"<tsugite_execution_result {attrs}>", f"<output>{escape(output)}</output>"]
    if error:
        parts.append(f"<error>{escape(error)}</error>")
    parts.append("</tsugite_execution_result>")
    return "\n".join(parts)


def _format_error_xml(data: Dict[str, Any]) -> str:
    reason = data.get("reason", "")
    msg = (
        "Format Error: " + reason + ". You must respond with exactly ONE ```python-exec "
        "code block per response. Combine all code into a single block."
    )
    return (
        '<tsugite_execution_result status="error">\n'
        "<output></output>\n"
        f"<error>{escape(msg)}</error>\n"
        "</tsugite_execution_result>"
    )


def _compaction_user_block(event: Event) -> str:
    summary = event.data.get("summary", "")
    intro = _compaction_intro_line(event)
    return (
        "<previous_conversation>\n"
        f"{intro}\n"
        "Continue from where this conversation left off. "
        "Pay attention to file paths, decisions, and incomplete work mentioned below.\n\n"
        f"{summary}\n"
        "</previous_conversation>"
    )


def _compaction_intro_line(event: Event) -> str:
    """First line of the compaction block, naming the time period and when the
    compaction itself happened. Falls back to the legacy generic phrasing when
    pre-existing JSONLs lack the range fields.
    """
    range_start = parse_iso_utc(event.data.get("range_start"))
    range_end = parse_iso_utc(event.data.get("range_end"))
    compacted_at = event.ts

    if range_start and range_end:
        start_str = range_start.strftime("%Y-%m-%d %H:%M")
        end_str = range_end.strftime("%Y-%m-%d %H:%M")
        when_str = compacted_at.strftime("%Y-%m-%d %H:%M") if compacted_at else None
        if when_str:
            return f"Summary of conversation from {start_str} to {end_str} (compacted on {when_str})."
        return f"Summary of conversation from {start_str} to {end_str}."

    return "The following is a summary of our earlier conversation, which was compacted to save context space."
