"""Context-provider registry contract.

Locks the behavior daemon routes and the send-flow detect-merge rely on:
menu vs detector classification, capture error propagation, and the
well-formedness filtering that keeps a misbehaving provider from poisoning a
send. The registry is process-global, so every test runs against a clean,
plugin-free registry (``ensure_loaded`` is stubbed so entry-point discovery
never leaks real providers into a unit test).
"""

from __future__ import annotations

import pytest

from tsugite import context as ctx
from tsugite.attachments.base import Attachment
from tsugite.context import (
    ContextChoice,
    ContextProvider,
    collect_detected_items,
    get_choices,
    get_context_provider,
    get_context_providers,
    register_context_provider,
    reset_context_providers,
    run_capture,
    run_search,
)


@pytest.fixture(autouse=True)
def _clean_registry(monkeypatch):
    monkeypatch.setattr(ctx, "ensure_loaded", lambda: None)
    reset_context_providers()
    yield
    reset_context_providers()


def _menu_provider(key="snippet", **kw):
    return ContextProvider(
        key=key, label="Snippet", capture=lambda arg, c: [Attachment.context(key, "Snippet", "x")], **kw
    )


def test_to_metadata_shape():
    assert Attachment.context("url", "URL", "https://x").to_metadata() == {
        "key": "url",
        "label": "URL",
        "value": "https://x",
    }


def test_register_and_get():
    p = _menu_provider()
    register_context_provider(p)
    assert get_context_provider("snippet") is p
    assert p in get_context_providers()


def test_register_last_wins():
    first = _menu_provider()
    second = _menu_provider()
    register_context_provider(first)
    register_context_provider(second)
    assert get_context_provider("snippet") is second
    assert [p for p in get_context_providers() if p.key == "snippet"] == [second]


def test_get_unknown_returns_none():
    assert get_context_provider("nope") is None


def test_in_menu_reflects_capture():
    assert _menu_provider().in_menu is True
    assert ContextProvider(key="webpage", label="Web page", detect=lambda m, c: []).in_menu is False


def test_menu_false_keeps_a_capture_provider_out_of_the_menu():
    # A capture triggered only by an explicit action opts out of the add-context
    # menu while staying capturable through run_capture.
    provider = ContextProvider(key="session", label="Session", capture=lambda a, c: [], menu=False)
    assert provider.in_menu is False


def test_picker_defaults_false():
    assert _menu_provider().picker is False
    assert _menu_provider(picker=True).picker is True


def test_run_capture_returns_cleaned_items():
    def capture(arg, c):
        return [
            Attachment.context("a", "A", "va"),
            Attachment.context("", "empty key", "v"),
            Attachment.context("b", "B", ""),
            {"key": "c", "value": "not an item"},
            Attachment.context("d", "D", "vd"),
        ]

    register_context_provider(ContextProvider(key="k", label="K", capture=capture))
    items = run_capture("k", None, {})
    assert [(it.key, it.value) for it in items] == [("a", "va"), ("d", "vd")]


def test_run_capture_unknown_or_no_capture_returns_empty():
    register_context_provider(ContextProvider(key="det", label="Det", detect=lambda m, c: []))
    assert run_capture("missing", None, {}) == []
    assert run_capture("det", None, {}) == []


def test_run_capture_propagates_provider_error():
    def boom(arg, c):
        raise RuntimeError("capture blew up")

    register_context_provider(ContextProvider(key="boom", label="Boom", capture=boom))
    with pytest.raises(RuntimeError, match="capture blew up"):
        run_capture("boom", None, {})


def test_run_capture_passes_arg_and_ctx():
    seen: dict = {}

    def capture(arg, c):
        seen["arg"] = arg
        seen["ctx"] = c
        return [Attachment.context("t", "T", "v")]

    register_context_provider(ContextProvider(key="t", label="T", capture=capture))
    run_capture("t", "term-7", {"session_id": "s1"})
    assert seen == {"arg": "term-7", "ctx": {"session_id": "s1"}}


def test_get_choices_returns_options_or_empty():
    register_context_provider(
        ContextProvider(
            key="term",
            label="Terminal",
            capture=lambda arg, c: [],
            choices=lambda c: [ContextChoice("t1", "One"), ContextChoice("t2", "Two")],
        )
    )
    register_context_provider(_menu_provider(key="nochoice"))
    assert [(c.value, c.label) for c in get_choices("term", {})] == [("t1", "One"), ("t2", "Two")]
    assert get_choices("nochoice", {}) == []
    assert get_choices("unknown", {}) == []


def test_get_choices_filters_non_choices():
    register_context_provider(
        ContextProvider(
            key="term",
            label="Terminal",
            capture=lambda arg, c: [],
            choices=lambda c: [ContextChoice("t1", "One"), {"value": "bad"}, "nope"],
        )
    )
    assert [c.value for c in get_choices("term", {})] == ["t1"]


def _search_provider(key="jira", **kw):
    def search(context, query):
        tickets = {"auth login flow": "PROJ-1", "auth logout": "PROJ-2", "billing": "PROJ-3"}
        return [ContextChoice(value=v, label=k) for k, v in tickets.items() if query.lower() in k]

    return ContextProvider(
        key=key,
        label="Jira",
        capture=lambda arg, c: [Attachment.context(key, "Jira", str(arg))],
        autocomplete_prefix="jira",
        search=search,
        menu=False,
        **kw,
    )


def test_is_autocomplete_source_needs_prefix_and_search():
    assert _search_provider().is_autocomplete_source is True
    # A prefix without search (or vice versa) is not a usable source.
    assert ContextProvider(key="x", label="X", autocomplete_prefix="x").is_autocomplete_source is False
    assert ContextProvider(key="y", label="Y", search=lambda c, q: []).is_autocomplete_source is False
    assert _menu_provider().is_autocomplete_source is False


def test_run_search_returns_query_matches():
    register_context_provider(_search_provider())
    assert [(c.value, c.label) for c in run_search("jira", "auth", {})] == [
        ("PROJ-1", "auth login flow"),
        ("PROJ-2", "auth logout"),
    ]


def test_run_search_empty_query_returns_all():
    register_context_provider(_search_provider())
    assert {c.value for c in run_search("jira", "", {})} == {"PROJ-1", "PROJ-2", "PROJ-3"}


def test_run_search_unknown_or_no_search_returns_empty():
    register_context_provider(_menu_provider(key="plain"))
    assert run_search("missing", "q", {}) == []
    assert run_search("plain", "q", {}) == []


def test_run_search_contains_a_raising_provider():
    def boom(context, query):
        raise RuntimeError("jira api down")

    register_context_provider(ContextProvider(key="flaky", label="Flaky", autocomplete_prefix="flaky", search=boom))
    # A plugin's search that raises is contained (not a 500 on a keystroke); the
    # typeahead just shows nothing.
    assert run_search("flaky", "q", {}) == []


def test_run_search_filters_non_choices():
    register_context_provider(
        ContextProvider(
            key="jira",
            label="Jira",
            autocomplete_prefix="jira",
            search=lambda c, q: [ContextChoice("PROJ-1", "One"), {"value": "bad"}, "nope"],
        )
    )
    assert [c.value for c in run_search("jira", "", {})] == ["PROJ-1"]


def test_run_search_passes_context_and_query():
    seen: dict = {}

    def search(context, query):
        seen["ctx"] = context
        seen["query"] = query
        return []

    register_context_provider(ContextProvider(key="jira", label="Jira", autocomplete_prefix="jira", search=search))
    run_search("jira", "auth", {"session_id": "s1"})
    assert seen == {"ctx": {"session_id": "s1"}, "query": "auth"}


def test_collect_detected_items_keeps_only_wellformed():
    def detect(message, c):
        return [
            Attachment.context("good", "Good", "v"),
            Attachment.context("", "empty key", "v"),
            Attachment.context("emptyval", "Empty", ""),
            {"key": "dict", "value": "not an item"},
            "totally wrong",
        ]

    register_context_provider(ContextProvider(key="d", label="D", detect=detect))
    items = collect_detected_items("hello", {})
    assert [it.key for it in items] == ["good"]


def test_collect_detected_items_swallows_raising_detector():
    register_context_provider(
        ContextProvider(key="boom", label="Boom", detect=lambda m, c: (_ for _ in ()).throw(ValueError("nope")))
    )
    register_context_provider(
        ContextProvider(key="ok", label="Ok", detect=lambda m, c: [Attachment.context("ok", "Ok", "v")])
    )
    items = collect_detected_items("hi", {})
    assert [it.key for it in items] == ["ok"]


def test_collect_detected_items_empty_without_detectors():
    register_context_provider(_menu_provider())
    assert collect_detected_items("no detectors here", {}) == []


def test_collect_detected_items_passes_message_and_ctx():
    seen: dict = {}

    def detect(message, c):
        seen["message"] = message
        seen["ctx"] = c
        return []

    register_context_provider(ContextProvider(key="d", label="D", detect=detect))
    collect_detected_items("scan me", {"session_id": "s9"})
    assert seen == {"message": "scan me", "ctx": {"session_id": "s9"}}
