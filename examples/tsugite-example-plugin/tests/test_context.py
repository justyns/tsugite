"""Tests for the example plugin's context providers (menu + detector)."""

from tsugite_example_plugin import context as ctx

from tsugite.context import get_context_provider, reset_context_providers


def test_snippet_choices_lists_static_options():
    values = {c.value for c in ctx.snippet_choices({})}
    assert values == {"mit", "coc"}


def test_capture_snippet_returns_item():
    items = ctx.capture_snippet("mit", {})
    assert len(items) == 1
    assert items[0].key == "snippet:mit"
    assert items[0].label == "MIT license"
    assert items[0].value


def test_capture_snippet_unknown_returns_empty():
    assert ctx.capture_snippet("nope", {}) == []
    assert ctx.capture_snippet(None, {}) == []


def test_detect_hashtags_attaches_one_item_per_tag():
    items = ctx.detect_hashtags("ship #alpha and #beta and #alpha again", {})
    assert [i.key for i in items] == ["hashtag:alpha", "hashtag:beta"]
    assert items[0].label == "#alpha"


def test_detect_hashtags_none_found():
    assert ctx.detect_hashtags("nothing tagged here", {}) == []


def test_demo_search_filters_by_query():
    values = {c.value for c in ctx.demo_search({}, "run")}
    assert values == {"runbook"}
    # An empty query lists every entry.
    assert {c.value for c in ctx.demo_search({}, "")} == {"roadmap", "runbook", "retro"}


def test_demo_search_no_match_is_empty():
    assert ctx.demo_search({}, "nonexistent") == []


def test_capture_demo_returns_item():
    items = ctx.capture_demo("runbook", {})
    assert len(items) == 1
    assert items[0].key == "demo:runbook"
    assert items[0].value


def test_capture_demo_unknown_returns_empty():
    assert ctx.capture_demo("nope", {}) == []
    assert ctx.capture_demo(None, {}) == []


def test_registers_all_providers():
    import importlib

    reset_context_providers()
    importlib.reload(ctx)

    menu = get_context_provider("example_snippet")
    assert menu is not None and menu.in_menu and menu.choices is not None

    detector = get_context_provider("example_hashtag")
    assert detector is not None and detector.detect is not None

    # The autocomplete source is a search provider kept out of the add-context menu.
    demo = get_context_provider("example_demo")
    assert demo is not None and demo.is_autocomplete_source and demo.autocomplete_prefix == "demo"
    assert demo.in_menu is False
