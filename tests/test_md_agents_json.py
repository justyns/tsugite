"""Unit tests for JSON directive argument extraction in md_agents.

Covers `find_json_object_in_string` and `extract_and_parse_json_args`. The key
correctness case is braces that live inside a JSON string value — the current
brace-counting implementation miscounts them; a proper JSON decoder does not.
"""

import pytest

from tsugite.md_agents import extract_and_parse_json_args, find_json_object_in_string


class TestFindJsonObjectInString:
    def test_flat_object(self):
        text = 'args={"path": "test.txt"}'
        start, end = find_json_object_in_string(text, "args=")
        assert text[start:end] == '{"path": "test.txt"}'

    def test_nested_object(self):
        text = 'args={"a": {"b": {"c": 1}}}'
        start, end = find_json_object_in_string(text, "args=")
        assert text[start:end] == '{"a": {"b": {"c": 1}}}'

    def test_trailing_content_after_json(self):
        text = 'args={"path": "x"} assign="data"'
        start, end = find_json_object_in_string(text, "args=")
        assert text[start:end] == '{"path": "x"}'

    def test_leading_content_before_keyword(self):
        text = 'name="read_file" args={"path": "x"}'
        start, end = find_json_object_in_string(text, "args=")
        assert text[start:end] == '{"path": "x"}'

    def test_brace_inside_string_value(self):
        """A `}` inside a JSON string must not be treated as the object terminator."""
        text = 'args={"msg": "}"}'
        start, end = find_json_object_in_string(text, "args=")
        assert text[start:end] == '{"msg": "}"}'

    def test_both_braces_inside_string_value(self):
        text = 'args={"tmpl": "{{ name }}"}'
        start, end = find_json_object_in_string(text, "args=")
        assert text[start:end] == '{"tmpl": "{{ name }}"}'

    def test_escaped_quote_inside_string(self):
        text = 'args={"msg": "she said \\"hi\\""}'
        start, end = find_json_object_in_string(text, "args=")
        assert text[start:end] == '{"msg": "she said \\"hi\\""}'

    def test_missing_keyword_raises(self):
        with pytest.raises(ValueError, match="not found"):
            find_json_object_in_string('name="read_file"', "args=")

    def test_no_opening_brace_after_keyword_raises(self):
        with pytest.raises(ValueError, match="No JSON object"):
            find_json_object_in_string('args="not json"', "args=")

    def test_unmatched_braces_raises(self):
        with pytest.raises(ValueError, match="[Uu]nmatched|[Ii]nvalid"):
            find_json_object_in_string('args={"unclosed": true', "args=")

    def test_empty_object(self):
        text = "args={}"
        start, end = find_json_object_in_string(text, "args=")
        assert text[start:end] == "{}"


class TestExtractAndParseJsonArgs:
    def test_simple(self):
        assert extract_and_parse_json_args('args={"path": "test.txt"}', "read_file") == {"path": "test.txt"}

    def test_with_surrounding_attributes(self):
        raw = 'name="read_file" args={"path": "x"} assign="data"'
        assert extract_and_parse_json_args(raw, "read_file") == {"path": "x"}

    def test_nested(self):
        raw = 'args={"url": "http://e", "headers": {"auth": "t"}}'
        assert extract_and_parse_json_args(raw, "fetch_json") == {
            "url": "http://e",
            "headers": {"auth": "t"},
        }

    def test_brace_inside_string_value_does_not_truncate(self):
        raw = 'args={"msg": "}"} assign="out"'
        assert extract_and_parse_json_args(raw, "echo") == {"msg": "}"}

    def test_jinja_template_in_value(self):
        raw = 'args={"tmpl": "{{ name }}"}'
        assert extract_and_parse_json_args(raw, "render") == {"tmpl": "{{ name }}"}

    def test_missing_args_raises(self):
        with pytest.raises(ValueError, match="missing required 'args'"):
            extract_and_parse_json_args('name="read_file"', "read_file")

    def test_malformed_json_raises(self):
        with pytest.raises(ValueError, match="Invalid JSON|must be a JSON object"):
            extract_and_parse_json_args("args={malformed}", "test_tool")
