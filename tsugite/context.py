"""Context providers: structured ``{key, label, value}`` items attached to a chat
message, folded into the agent's context and rendered in the web UI's context
gutter (the same ``<client_context>`` path the browser's own providers, e.g.
location, ride).

Two producer kinds, both plugin-extensible through the ``tsugite.context_providers``
entry point (a module-only entry point whose import registers the provider):

  - menu provider: appears in the composer's "add context" menu. On pick the
    daemon runs ``capture`` server-side. A provider that also defines ``choices``
    first offers a submenu and passes the chosen value to ``capture`` as ``arg``.
    A capture provider may set ``menu=False`` to stay out of the menu and be run
    only by an explicit UI action (a button or a reference paste).
  - detector: ``detect`` scans the outgoing message server-side at send time and
    attaches items for anything it recognizes (a URL, a JIRA-1234 ticket).

A single provider may define either or both. The item shape is deliberately the
same one the frontend sends as ``context_metadata``, so server-produced items
merge with client-captured ones and reuse the whole fold/record/render path with
no read-side changes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Optional

from tsugite.attachments.base import Attachment

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ContextChoice:
    """A pickable option a menu provider offers before capture. ``value`` is
    passed back to the provider's ``capture`` as ``arg``."""

    value: str
    label: str


# capture(arg, context) -> context attachments. ``arg`` is the picked
# ContextChoice.value, or None when the provider has no choices (direct capture).
CaptureFn = Callable[[Optional[str], dict], "list[Attachment]"]
# choices(context) -> submenu options for a menu provider.
ChoicesFn = Callable[[dict], "list[ContextChoice]"]
# search(context, query) -> options matching a typed query. The query-aware twin
# of ``choices``: a provider that declares ``autocomplete_prefix`` + ``search``
# becomes an ``@<prefix> <query>`` autocomplete source in the composer.
SearchFn = Callable[[dict, str], "list[ContextChoice]"]
# detect(message, context) -> context attachments for mentions found in the message.
DetectFn = Callable[[str, dict], "list[Attachment]"]


@dataclass
class ContextProvider:
    """A plugin-contributed source of context items.

    ``key`` is the stable id (the ``context_metadata`` key, the chip/testid
    suffix, the dedupe key). Give a menu provider ``capture`` (and optionally
    ``choices`` for a submenu); give a detector ``detect``; a provider may have
    both. Set ``picker`` when the choices are large/searchable so the composer
    opens the searchable picker overlay instead of the inline submenu. Set
    ``menu=False`` for a capture the UI triggers only by an explicit action (a
    button or a reference paste), so it stays out of the add-context menu while
    remaining reachable through the capture endpoint.

    Declare ``autocomplete_prefix`` + ``search`` to also become a composer
    ``@<prefix> <query>`` autocomplete source: typing that prefix scopes the ``@``
    popover to this provider and fetches ``search`` server-side as the user types;
    picking a result captures it through the same ``capture`` path. Such a source
    typically sets ``menu=False`` (it is reached by its prefix, not the menu).
    """

    key: str
    label: str
    icon: str = "sparkle"
    capture: Optional[CaptureFn] = None
    choices: Optional[ChoicesFn] = None
    detect: Optional[DetectFn] = None
    picker: bool = False
    menu: bool = True
    autocomplete_prefix: Optional[str] = None
    search: Optional[SearchFn] = None

    @property
    def in_menu(self) -> bool:
        """A provider shows in the composer menu iff it can capture on pick and it
        hasn't opted out (``menu=False`` for an explicit-action-only capture)."""
        return self.capture is not None and self.menu

    @property
    def is_autocomplete_source(self) -> bool:
        """A provider is a composer ``@<prefix>`` autocomplete source iff it
        declares both a prefix and a query-aware ``search``."""
        return self.autocomplete_prefix is not None and self.search is not None


_registry: dict[str, ContextProvider] = {}
_loaded = False


def register_context_provider(provider: ContextProvider) -> None:
    """Register a provider (idempotent by ``key``; the last registration wins).

    Plugins call this at import time from a ``tsugite.context_providers`` module.
    """
    if provider.key in _registry:
        logger.debug("Context provider '%s' re-registered", provider.key)
    _registry[provider.key] = provider


def reset_context_providers() -> None:
    """Clear the registry and the load flag (tests)."""
    global _loaded
    _registry.clear()
    _loaded = False


def ensure_loaded() -> None:
    """Register the built-in providers, then import the
    ``tsugite.context_providers`` entry-point modules once, so every
    ``register_context_provider`` call has run before the registry is read.

    Built-ins register first (a called function, so they re-register after a
    ``reset_context_providers()`` where an import side effect would not) and
    plugins load after, so a plugin may override a built-in by key.
    """
    global _loaded
    if _loaded:
        return
    _loaded = True
    from tsugite.builtin_context import register_builtin_providers

    register_builtin_providers()
    try:
        from tsugite.plugins import GROUP_CONTEXT_PROVIDERS, load_module_only_plugins

        load_module_only_plugins(GROUP_CONTEXT_PROVIDERS)
    except Exception as e:  # never let plugin discovery break a read
        logger.warning("Loading context-provider plugins failed: %s", e)


def get_context_providers() -> list[ContextProvider]:
    """Every registered provider (menu and detector)."""
    ensure_loaded()
    return list(_registry.values())


def get_context_provider(key: str) -> Optional[ContextProvider]:
    ensure_loaded()
    return _registry.get(key)


def _clean(items: object) -> list[Attachment]:
    """Keep only well-formed context attachments (an Attachment with a non-empty
    ``key`` and text content). Context producers set ``key`` via
    ``Attachment.context``; a bare Attachment without one is treated as malformed
    here (uploads take a different path and never reach this filter)."""
    if not isinstance(items, list):
        return []
    return [it for it in items if isinstance(it, Attachment) and it.key and it.value]


def run_capture(key: str, arg: Optional[str], context: dict) -> list[Attachment]:
    """Run a menu provider's ``capture``. Propagates the provider's exception so
    the endpoint can surface it (the user picked this deliberately); returns [] if
    the provider is unknown or has no capture."""
    provider = get_context_provider(key)
    if not provider or not provider.capture:
        return []
    return _clean(provider.capture(arg, context))


def get_choices(key: str, context: dict) -> list[ContextChoice]:
    """A menu provider's current submenu options, or [] if it has none."""
    provider = get_context_provider(key)
    if not provider or not provider.choices:
        return []
    result = provider.choices(context)
    return [c for c in result if isinstance(c, ContextChoice)] if isinstance(result, list) else []


def run_search(key: str, query: str, context: dict) -> list[ContextChoice]:
    """An autocomplete source's matches for ``query``, or [] when the provider is
    unknown or is not a search source. The query-aware twin of ``get_choices``,
    kept separate so ``choices`` stays the menu-submenu path and ``search`` the
    typeahead path; neither disturbs the other. A plugin's ``search`` that raises
    is contained (logged, empty) so a flaky third-party source can't 500 the
    typeahead on a keystroke."""
    provider = get_context_provider(key)
    if not provider or not provider.search:
        return []
    try:
        result = provider.search(context, query)
    except Exception as e:  # noqa: BLE001 - a plugin's failure must not break the composer typeahead
        logger.warning("context provider %r search failed: %s", key, e)
        return []
    return [c for c in result if isinstance(c, ContextChoice)] if isinstance(result, list) else []


def collect_detected_items(message: str, context: dict) -> list[Attachment]:
    """Run every detector against the outgoing message and collect their items.

    Detectors must never break a send: a raising or misbehaving detector is
    logged and skipped, and only well-formed items are kept.
    """
    items: list[Attachment] = []
    for provider in get_context_providers():
        if not provider.detect:
            continue
        try:
            items.extend(_clean(provider.detect(message, context)))
        except Exception as e:
            logger.warning("Context detector '%s' failed: %s", provider.key, e)
    return items
