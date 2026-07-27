"""Example context providers - the producer kinds in one module.

A context provider contributes structured ``{key, label, value}`` items that get
folded into the agent's context and shown in the web UI's context gutter (the
same path the browser's own providers, e.g. location, ride). This module
registers one of each kind at import time:

  - MENU provider: shows in the composer's "add context" menu. On pick the daemon
    runs ``capture`` server-side. A provider that also defines ``choices`` first
    offers a submenu and passes the chosen value to ``capture`` as ``arg``.
  - DETECTOR: ``detect`` scans the outgoing message server-side at send time and
    attaches an item for anything it recognizes (a URL, a ticket id, ...).
  - AUTOCOMPLETE SOURCE: ``autocomplete_prefix`` + ``search`` make ``@demo <query>``
    fetch matching options as the user types; picking one runs ``capture``.

Wire it up with its own entry point (alongside the tool/hook/adapter ones); the
import runs the register_context_provider() calls below:

    [project.entry-points."tsugite.context_providers"]
    example = "tsugite_example_plugin.context"
"""

from __future__ import annotations

import re
from typing import Optional

from tsugite.attachments.base import Attachment
from tsugite.context import (
    ContextChoice,
    ContextProvider,
    register_context_provider,
)

# ─────────────────────────────────────────────────────────────────────────────
# 1. MENU PROVIDER - appears in the composer's "add context" menu. This one also
#    defines `choices`, so picking it opens a submenu; the picked
#    ContextChoice.value arrives as `capture`'s `arg`. Omit `choices` for a
#    provider that captures directly on pick (its `arg` is then None).
# ─────────────────────────────────────────────────────────────────────────────

# Static demo data. A real provider would derive its choices from live state
# instead (open files, running jobs, recent commits, ...).
_SNIPPETS = {
    "mit": ("MIT license", "Permission is hereby granted, free of charge, to any person ..."),
    "coc": ("Code of conduct", "Be kind. Assume good faith. Harassment is not tolerated."),
}


def snippet_choices(context: dict) -> list[ContextChoice]:
    """Submenu options shown before capture.

    `context` carries {session_id, user_id, agent, workspace_dir}; a live
    provider would use it to scope its options to the current session. `value`
    is what comes back to `capture` as `arg`; `label` is what the user sees.
    """
    return [ContextChoice(value=key, label=label) for key, (label, _body) in _SNIPPETS.items()]


def capture_snippet(arg: str | None, context: dict) -> list[Attachment]:
    """Turn the picked choice into context item(s).

    Return [] for anything you do not recognize - never raise for a bad pick.
    `key` is the stable id (dedupe / chip / gutter-row key); keep it unique per
    distinct item.
    """
    snippet = _SNIPPETS.get(arg or "")
    if snippet is None:
        return []
    label, body = snippet
    return [Attachment.context(key=f"snippet:{arg}", label=label, value=body)]


register_context_provider(
    ContextProvider(
        key="example_snippet",
        label="Canned snippet",
        icon="sparkle",
        choices=snippet_choices,
        capture=capture_snippet,
    )
)


# ─────────────────────────────────────────────────────────────────────────────
# 2. DETECTOR - `detect` scans every outgoing message and attaches an item for
#    each mention it recognizes. Keep it fast and total: a detector must never
#    raise (a raising detector is logged and skipped by the runner) and should
#    return [] when it finds nothing. This one recognizes `#tag` hashtags.
# ─────────────────────────────────────────────────────────────────────────────
_HASHTAG_RE = re.compile(r"#(\w+)")


def detect_hashtags(message: str, context: dict) -> list[Attachment]:
    """Attach a canned note for each `#tag` in the message (de-duplicated,
    order preserved). A real detector might look the tag up in a wiki or tracker."""
    items: list[Attachment] = []
    for tag in dict.fromkeys(_HASHTAG_RE.findall(message or "")):
        items.append(Attachment.context(key=f"hashtag:{tag}", label=f"#{tag}", value=f"The user mentioned #{tag}."))
    return items


register_context_provider(
    ContextProvider(key="example_hashtag", label="Hashtag", icon="sparkle", detect=detect_hashtags)
)


# ─────────────────────────────────────────────────────────────────────────────
# 3. AUTOCOMPLETE SOURCE - declares `autocomplete_prefix` + `search`, so typing
#    `@demo <query>` in the composer scopes the popover to this provider and
#    fetches `search` server-side as the user types; picking a result captures it
#    through `capture`. This is the worked example a real integration (a Jira
#    plugin doing `@jira auth`) mirrors: swap the static dict for a live query.
#    `menu=False` keeps it out of the add-context menu (it is reached by its
#    prefix, not the menu).
# ─────────────────────────────────────────────────────────────────────────────
_DEMO_ENTRIES = {
    "roadmap": "Q3 roadmap: ship the pluggable @ autocomplete.",
    "runbook": "Runbook: restart the daemon, then tail the log.",
    "retro": "Retro notes: keep the pick/attach path, do not reinvent it.",
}


def demo_search(context: dict, query: str) -> list[ContextChoice]:
    """Options matching the typed query (case-insensitive substring).

    `context` carries {session_id, user_id, agent, workspace_dir}; a live source
    would use it to scope results to the session. An empty query lists everything.
    `value` is what `capture` receives as `arg`; `label` is what the user sees.
    """
    q = (query or "").strip().lower()
    return [ContextChoice(value=key, label=key) for key in _DEMO_ENTRIES if q in key]


def capture_demo(arg: Optional[str], context: dict) -> list[Attachment]:
    """Turn a picked result into a context item, or [] for an unknown value."""
    body = _DEMO_ENTRIES.get(arg or "")
    if body is None:
        return []
    return [Attachment.context(key=f"demo:{arg}", label=f"demo/{arg}", value=body)]


register_context_provider(
    ContextProvider(
        key="example_demo",
        label="Demo docs",
        icon="sparkle",
        autocomplete_prefix="demo",
        search=demo_search,
        capture=capture_demo,
        menu=False,
    )
)
