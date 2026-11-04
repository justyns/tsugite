"""Tests for the plugin link preview resolver registry."""

import logging
import re

import pytest

from tsugite.link_previews import (
    attributing_to,
    link_resolver,
    link_resolver_patterns,
    register_link_resolver,
    registered_link_resolvers,
    resolver_for,
)


def _resolve(url):
    return {"label": url}


class TestRegister:
    def test_registers_the_descriptor(self):
        register_link_resolver(name="pulls", hosts=["Forge.Example.Test"], path=r"/pulls/\d+$", resolve=_resolve)

        assert registered_link_resolvers() == [
            {
                "plugin": "",
                "name": "pulls",
                "hosts": ["forge.example.test"],
                "path": r"/pulls/\d+$",
                "resolve": _resolve,
            }
        ]

    def test_an_invalid_path_pattern_fails_at_registration(self):
        with pytest.raises(re.error):
            register_link_resolver(name="bad", hosts=["example.test"], path="(", resolve=_resolve)

    def test_the_decorator_returns_the_function_unchanged(self):
        @link_resolver(name="demo", hosts=["example.test"])
        def resolve(url):
            return None

        assert registered_link_resolvers()[0]["resolve"] is resolve

    def test_patterns_leave_out_the_callable(self):
        register_link_resolver(name="demo", hosts=["example.test"], resolve=_resolve)

        assert link_resolver_patterns() == [{"plugin": "", "name": "demo", "hosts": ["example.test"], "path": None}]


class TestResolverFor:
    def test_matches_host_and_path(self):
        register_link_resolver(name="pulls", hosts=["forge.example.test"], path=r"/pulls/\d+$", resolve=_resolve)

        assert resolver_for("https://FORGE.example.test/octo-org/demo/pulls/7")["name"] == "pulls"
        assert resolver_for("https://forge.example.test/octo-org/demo/issues/7") is None
        assert resolver_for("https://other.example.test/octo-org/demo/pulls/7") is None

    def test_without_a_path_any_path_on_the_host_matches(self):
        register_link_resolver(name="any", hosts=["example.test"], resolve=_resolve)

        assert resolver_for("http://example.test/")["name"] == "any"

    def test_only_http_urls_match(self):
        register_link_resolver(name="any", hosts=["example.test"], resolve=_resolve)

        assert resolver_for("ftp://example.test/x") is None
        assert resolver_for("javascript://example.test/x") is None

    def test_the_first_registered_applies_with_a_warning(self, caplog):
        register_link_resolver(name="first", hosts=["example.test"], resolve=_resolve)
        register_link_resolver(name="second", hosts=["example.test"], path="/x", resolve=_resolve)

        with caplog.at_level(logging.WARNING, logger="tsugite.link_previews"):
            assert resolver_for("https://example.test/x")["name"] == "first"

        assert "second" in caplog.text


class TestAttribution:
    def test_stamps_registrations_with_the_plugin_name(self):
        with attributing_to("forge"):
            register_link_resolver(name="pulls", hosts=["example.test"], resolve=_resolve)

        assert [r["plugin"] for r in registered_link_resolvers()] == ["forge"]

    def test_a_block_that_raises_leaves_nothing_behind(self):
        with pytest.raises(RuntimeError), attributing_to("broken"):
            register_link_resolver(name="pulls", hosts=["example.test"], resolve=_resolve)
            raise RuntimeError("boom")

        assert registered_link_resolvers() == []
