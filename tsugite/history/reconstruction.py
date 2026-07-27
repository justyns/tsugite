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


def reconstruct_raw_turns(events: Iterable[Event]) -> List[Dict[str, Any]]:
    """Rebuild each model call's request messages and raw response from an event log.

    A ``model_request`` is recorded right before each provider call, so the
    messages the model saw that call are :func:`events_to_messages` of the log
    prefix up to it (a prefix reconstructs the conversation as-of that point,
    matching resume). One entry per ``model_request``, in log order.

    Each entry carries an ``index`` (a monotonic 1-based count) as its stable
    identity. The event's own ``turn`` is a per-run step counter that resets to 0
    every user message, so it is kept only as a secondary hint - it is neither
    unique nor a pairing key. The response is the ``model_response`` that follows
    the request in the log (before the next request), so pairing survives the
    repeated turn numbers; a call that never produced one carries null.

    ``new_messages`` is what the call added to the prompt over the previous call
    (consecutive prompts only append), so a surface can show just the delta
    instead of re-rendering the whole conversation every entry. Across a
    compaction the prompt resets to a summary and shares no prefix; then
    ``reset_before`` is true and ``new_messages`` is the whole request.

    Returns plain dicts so any surface can serialize them: the per-call slicing
    lives here in core, next to the reconstruction it builds on, not in a caller.
    """
    events = list(events)

    entries: List[Dict[str, Any]] = []
    prev_request: Optional[List[Dict[str, Any]]] = None
    for i, event in enumerate(events):
        if event.type != "model_request":
            continue
        request = events_to_messages(events[:i])
        new_messages, reset_before = _request_delta(prev_request, request)
        entries.append(
            {
                "index": len(entries) + 1,
                "turn": event.data.get("turn"),
                "provider": event.data.get("provider"),
                "model": event.data.get("model"),
                "request": request,
                "new_messages": new_messages,
                "reset_before": reset_before,
                "response": _response_after(events, i),
            }
        )
        prev_request = request
    return entries


def _response_after(events: List[Event], req_idx: int) -> Optional[Dict[str, Any]]:
    """The ``model_response`` paired with the ``model_request`` at ``req_idx``: the
    first one before the next ``model_request``. Pairs by log position rather than
    the ``turn`` field, which is a per-run step counter and repeats across user
    messages - a turn-keyed lookup would collide and mispair early calls."""
    for event in events[req_idx + 1 :]:
        if event.type == "model_request":
            return None
        if event.type == "model_response":
            return {"raw_content": event.data.get("raw_content", "")}
    return None


def _request_delta(
    prev: Optional[List[Dict[str, Any]]], cur: List[Dict[str, Any]]
) -> tuple[List[Dict[str, Any]], bool]:
    """Messages ``cur`` added over ``prev`` (what this call appended to the prompt),
    plus whether the prior prefix was dropped. Consecutive calls only append, so
    the delta is ``cur`` past the shared head - except across a compaction, where
    ``cur`` resets to a summary and shares no prefix; then the whole prompt is new
    and the flag is true so a surface can mark the reset."""
    if prev is None:
        return cur, False
    if cur[: len(prev)] == prev:
        return cur[len(prev) :], False
    return cur, True


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
