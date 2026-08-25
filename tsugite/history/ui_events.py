"""Map a stored Event to the flat dict the daemon/UI layers consume.

History stores Event(type, ts, data); the daemon's progress/SSE code reads top-level
keys (``type``, ``timestamp``, and data fields like ``name``/``turn``). This is the one
adapter between the two shapes - keep daemon reads going through it so storage changes
can't silently break the UI.

It also normalizes events on read so clients never have to parse raw model
output themselves: model_response events recorded before the parse persisted
get thought/content_blocks/tail backfilled with the real parser, and
user_input events get their runtime context injections split into structured
blocks.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

from .models import Event

# Tags the daemon prepends to a user message as context injections (scheduled
# task results, environment/message context). Split off the visible message so
# UIs can fold them instead of rendering them as the user's own words.
# ``environment`` appears only in older histories but must keep normalizing.
_INJECTED_TAGS = ("message_context", "environment", "background_task_complete", "scheduled_task", "client_context")
_INJECTED_RES = {tag: re.compile(rf"<{tag}(\s[^>]*)?>(.*?)</{tag}>", re.DOTALL) for tag in _INJECTED_TAGS}
# Single quotes because quoteattr falls back to them when the value holds a `"`.
_ID_RE = re.compile(r"""id=(?:"([^"]*)"|'([^']*)')""")


def split_injected_context(text: str) -> Tuple[List[Dict[str, str]], str]:
    """Peel leading injection tags off a user message.

    Returns (blocks, rest) where each block is {tag, id?, body} and rest is
    the text the person actually typed ("" for a pure injection turn).
    """
    blocks: List[Dict[str, str]] = []
    body = text or ""
    progress = True
    while progress:
        progress = False
        body = body.lstrip()
        for tag, tag_re in _INJECTED_RES.items():
            m = tag_re.match(body)
            if m:
                block = {"tag": tag}
                id_m = _ID_RE.search(m.group(1) or "")
                if id_m:
                    block["id"] = id_m.group(1) if id_m.group(1) is not None else id_m.group(2)
                block["body"] = (m.group(2) or "").strip()
                blocks.append(block)
                body = body[m.end() :]
                progress = True
                break
    return blocks, body.strip()


def _normalize(out: Dict[str, Any], event_type: str) -> None:
    if event_type == "model_response":
        if "thought" not in out:
            # Recorded before the parse persisted: backfill with the real parser.
            # (thought is always stored on new events, even when empty, exactly so
            # this check can distinguish parsed from unparsed.)
            from tsugite.core.agent import parse_response_text

            parsed = parse_response_text(out.get("raw_content") or "")
            out["thought"] = parsed.thought
            if parsed.content_blocks:
                out["content_blocks"] = parsed.content_blocks
            if parsed.tail:
                out["tail"] = parsed.tail
        elif out.get("tail"):
            # A tail persisted before fabricated-result stripping can still carry a
            # hallucinated tool-result continuation (unbalanced fences that garble
            # the turn); re-clean it on read so old turns render right, no migration.
            from tsugite.core.agent import strip_fabricated_result_tail

            cleaned = strip_fabricated_result_tail(out["tail"])
            if cleaned:
                out["tail"] = cleaned
            else:
                out.pop("tail", None)
    elif event_type == "user_input":
        # The UI gutter's context items come from the structured client_context the
        # recorder wrote, never from the folded <client_context> XML: that block is
        # prompt rendering, peeled off the visible text and otherwise left alone.
        stored = out.pop("client_context", None)
        blocks, rest = split_injected_context(out.get("text") or "")
        if blocks:
            out["injected"] = [
                {"tag": "client_context", "items": stored}
                if stored is not None and b.get("tag") == "client_context"
                else b
                for b in blocks
            ]
            out["display_text"] = rest


def event_to_ui_dict(event: Event) -> Dict[str, Any]:
    """Flatten an Event into a UI dict: data at top level, plus type/timestamp/id.

    ``timestamp`` uses the event's own ``isoformat()`` (preserving source precision) to
    match the long-standing daemon flat-dict shape; storage columns keep ``iso_utc``.
    """
    out: Dict[str, Any] = dict(event.data)
    _normalize(out, event.type)
    # Authoritative keys win over any same-named data field.
    out["type"] = event.type
    out["timestamp"] = event.ts.isoformat()
    if event.id is not None:
        out["id"] = event.id
    return out
