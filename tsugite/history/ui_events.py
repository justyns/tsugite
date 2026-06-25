"""Map a stored Event to the flat dict the daemon/UI layers consume.

History stores Event(type, ts, data); the daemon's progress/SSE code reads top-level
keys (``type``, ``timestamp``, and data fields like ``name``/``turn``). This is the one
adapter between the two shapes - keep daemon reads going through it so storage changes
can't silently break the UI.
"""

from __future__ import annotations

from typing import Any, Dict

from .models import Event


def event_to_ui_dict(event: Event) -> Dict[str, Any]:
    """Flatten an Event into a UI dict: data at top level, plus type/timestamp/id.

    ``timestamp`` uses the event's own ``isoformat()`` (preserving source precision) to
    match the long-standing daemon flat-dict shape; storage columns keep ``iso_utc``.
    """
    out: Dict[str, Any] = dict(event.data)
    # Authoritative keys win over any same-named data field.
    out["type"] = event.type
    out["timestamp"] = event.ts.isoformat()
    if event.id is not None:
        out["id"] = event.id
    return out
