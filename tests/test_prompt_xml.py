"""Unit tests for the prompt_xml element builder."""

import xml.etree.ElementTree as ET

import pytest

from tsugite.prompt_xml import El, Raw, render_fragments


class TestAttributes:
    def test_values_are_quoted(self):
        assert El("t", attrs={"a": "x"}).render() == '<t a="x"></t>'

    def test_special_characters_are_escaped(self):
        out = El("t", attrs={"a": "R&D <x>"}).render()
        assert out == '<t a="R&amp;D &lt;x&gt;"></t>'
        assert ET.fromstring(out).get("a") == "R&D <x>"

    def test_double_quote_switches_to_single_quotes(self):
        out = El("t", attrs={"a": 'has"quote'}).render()
        assert ET.fromstring(out).get("a") == 'has"quote'

    def test_none_values_are_dropped(self):
        assert El("t", attrs={"a": "1", "b": None, "c": "2"}).render() == '<t a="1" c="2"></t>'

    def test_insertion_order_is_preserved(self):
        assert El("t", attrs={"z": "1", "a": "2"}).render() == '<t z="1" a="2"></t>'

    def test_non_string_values_are_stringified(self):
        assert El("t", attrs={"n": 42, "b": True}).render() == '<t n="42" b="True"></t>'


class TestBodies:
    def test_str_child_is_escaped(self):
        out = El("t", ["a < b & c"], inline=True).render()
        assert out == "<t>a &lt; b &amp; c</t>"
        assert ET.fromstring(out).text == "a < b & c"

    def test_raw_child_is_verbatim(self):
        assert El("t", [Raw("<div> & co")], inline=True).render() == "<t><div> & co</t>"

    def test_empty_element(self):
        assert El("t").render() == "<t></t>"

    def test_void_element_self_closes(self):
        assert El("t", attrs={"a": "1"}, void=True).render() == '<t a="1" />'

    def test_void_element_without_attrs(self):
        assert El("t", void=True).render() == "<t />"


class TestNesting:
    def test_block_children_on_own_lines(self):
        out = El("outer", [El("a", ["1"], inline=True), El("b", ["2"], inline=True)]).render()
        assert out == "<outer>\n<a>1</a>\n<b>2</b>\n</outer>"

    def test_indent_applies_per_level(self):
        out = El("outer", [El("a", ["1"], inline=True)]).render(indent="  ")
        assert out == "<outer>\n  <a>1</a>\n</outer>"

    def test_nested_indent_compounds(self):
        inner = El("inner", [El("leaf", ["x"], inline=True)])
        assert El("outer", [inner]).render(indent="  ") == (
            "<outer>\n  <inner>\n    <leaf>x</leaf>\n  </inner>\n</outer>"
        )

    def test_none_children_are_dropped(self):
        assert El("t", [El("a", ["1"], inline=True), None]).render() == "<t>\n<a>1</a>\n</t>"

    def test_mixed_raw_and_element_children(self):
        out = El("t", [Raw("prefix"), El("a", ["1"], inline=True)]).render()
        assert out == "<t>\nprefix\n<a>1</a>\n</t>"


class TestFragments:
    def test_renders_multiple_roots(self):
        out = render_fragments([El("a", ["1"], inline=True), El("b", ["2"], inline=True)])
        assert out == "<a>1</a>\n<b>2</b>"

    def test_drops_none(self):
        assert render_fragments([El("a", ["1"], inline=True), None]) == "<a>1</a>"

    def test_empty_list(self):
        assert render_fragments([]) == ""


class TestAttributeFastPath:
    """_quote skips quoteattr for clean values; it must never disagree with it."""

    @pytest.mark.parametrize("value", ["plain", "a&b", "a<b", 'a"b', "a\nb", "a\tb", "a\rb", "it's", ""])
    def test_matches_quoteattr(self, value):
        from xml.sax.saxutils import quoteattr

        from tsugite.prompt_xml import _quote

        assert _quote(value) == quoteattr(value)

    def test_matches_quoteattr_on_random_strings(self):
        import random
        import string
        from xml.sax.saxutils import quoteattr

        from tsugite.prompt_xml import _quote

        rng = random.Random(7)
        for _ in range(5000):
            s = "".join(rng.choice(string.printable) for _ in range(rng.randint(0, 12)))
            assert _quote(s) == quoteattr(s), repr(s)


class TestWellFormedness:
    @pytest.mark.parametrize(
        "value",
        ["plain", "a & b", "<script>", 'q"q', "it's", "mixed <&>\"'", ""],
    )
    def test_escaped_bodies_and_attrs_always_parse(self, value):
        out = El("t", [value], attrs={"a": value}, inline=True).render()
        parsed = ET.fromstring(out)
        assert parsed.get("a") == value
        assert (parsed.text or "") == value
