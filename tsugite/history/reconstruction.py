"""Reconstruct LLM-bound messages from a session event log.

The agent loop appends events as they happen. Before each model call we walk
the log and build the messages array the provider expects. Raw `model_response`
text is sent back verbatim (no re-rendering from parsed pieces) so parser bugs
can't corrupt what the model sees as its own past output.
"""

from typing import Any, Dict, Iterable, List, Optional
from xml.sax.saxutils import escape

from .models import Event


def last_index_of(events: List[Event], type_: str) -> Optional[int]:
    """Index of the last event of `type_`, or None if absent."""
    for i in range(len(events) - 1, -1, -1):
        if events[i].type == type_:
            return i
    return None


def events_to_messages(events: Iterable[Event], provider: Optional[str] = None) -> List[Dict[str, Any]]:
    """Walk events and return the messages list to send next.

    For stateless providers (OpenAI/Anthropic) returns the full message
    history. For session-owning providers (Claude Code) returns only the
    unsent tail since the provider holds the prior conversation.
    """
    events = list(events)

    cutoff = last_index_of(events, "compaction")
    post_compaction = events[cutoff + 1 :] if cutoff is not None else events

    messages: List[Dict[str, Any]] = []
    if cutoff is not None:
        summary = events[cutoff].data.get("summary", "")
        messages.append({"role": "user", "content": _compaction_user_block(summary)})
        messages.append(
            {"role": "assistant", "content": "I've reviewed our previous conversation and I'm ready to continue."}
        )

    for event in post_compaction:
        rendered = _event_to_message(event)
        if rendered:
            messages.append(rendered)

    if provider == "claude_code":
        return _claude_code_tail(messages, post_compaction, cutoff)

    return messages


def _event_to_message(event: Event) -> Optional[Dict[str, Any]]:
    if event.type == "user_input":
        text = event.data.get("text", "")
        return {"role": "user", "content": text}
    if event.type == "model_response":
        raw = event.data.get("raw_content", "")
        return {"role": "assistant", "content": raw}
    if event.type == "code_execution":
        return {"role": "user", "content": _execution_xml(event.data)}
    if event.type == "format_error":
        return {"role": "user", "content": _format_error_xml(event.data)}
    return None


def _execution_xml(data: Dict[str, Any]) -> str:
    """Build the <tsugite_execution_result> envelope from event data."""
    output = data.get("output") or ""
    error = data.get("error")
    duration_ms = data.get("duration_ms")

    status = "error" if error else "success"
    attrs = f'status="{status}"'
    if duration_ms:
        attrs += f' duration_ms="{duration_ms}"'

    parts = [f"<tsugite_execution_result {attrs}>", f"<output>{escape(output)}</output>"]
    if error:
        parts.append(f"<error>{escape(error)}</error>")
    parts.append("</tsugite_execution_result>")
    return "\n".join(parts)


def _format_error_xml(data: Dict[str, Any]) -> str:
    reason = data.get("reason", "")
    msg = (
        "Format Error: " + reason + ". You must respond with exactly ONE ```python "
        "code block per response. Combine all code into a single block."
    )
    return (
        '<tsugite_execution_result status="error">\n'
        "<output></output>\n"
        f"<error>{escape(msg)}</error>\n"
        "</tsugite_execution_result>"
    )


def _compaction_user_block(summary: str) -> str:
    return (
        "<previous_conversation>\n"
        "The following is a summary of our earlier conversation, "
        "which was compacted to save context space.\n"
        "Continue from where this conversation left off. "
        "Pay attention to file paths, decisions, and incomplete work mentioned below.\n\n"
        f"{summary}\n"
        "</previous_conversation>"
    )


def _claude_code_tail(
    messages: List[Dict[str, Any]],
    post_compaction: List[Event],
    compaction_cutoff: Optional[int],
) -> List[Dict[str, Any]]:
    """Return only the messages Claude Code hasn't seen yet.

    Claude Code maintains its own session, so we send only what's been added
    since its last `model_response`. After compaction we send the synthetic
    summary block plus any newer events.
    """
    if compaction_cutoff is not None:
        # After compaction, hand Claude Code the summary + everything since.
        return messages

    last_resp = last_index_of(post_compaction, "model_response")
    if last_resp is None:
        return messages

    tail: List[Dict[str, Any]] = []
    for event in post_compaction[last_resp + 1 :]:
        m = _event_to_message(event)
        if m:
            tail.append(m)
    return tail
