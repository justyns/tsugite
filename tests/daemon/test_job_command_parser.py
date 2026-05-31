"""Tests for the /job slash command's AC + flag parsing helpers."""

from tsugite.daemon.commands import _parse_acceptance_criteria


def test_empty_returns_empty_list():
    assert _parse_acceptance_criteria(None) == []
    assert _parse_acceptance_criteria("") == []
    assert _parse_acceptance_criteria([]) == []


def test_plain_pipe_separated_strings_default_to_llm_kind():
    out = _parse_acceptance_criteria("tests pass|PR open")
    assert out == [
        {"text": "tests pass", "kind": "llm"},
        {"text": "PR open", "kind": "llm"},
    ]


def test_ac_kind_inferred_from_double_colon_syntax():
    out = _parse_acceptance_criteria("tests pass::test|button renders::ui|curl returns 200::cmd")
    assert out == [
        {"text": "tests pass", "kind": "test"},
        {"text": "button renders", "kind": "ui"},
        {"text": "curl returns 200", "kind": "cmd"},
    ]


def test_unknown_kind_falls_back_to_llm():
    out = _parse_acceptance_criteria("does the right thing::garbage")
    assert out == [{"text": "does the right thing", "kind": "llm"}]


def test_dict_entries_pass_through():
    out = _parse_acceptance_criteria([{"text": "x", "kind": "ui"}])
    assert out == [{"text": "x", "kind": "ui"}]


def test_json_array_parsed_with_kind_suffix():
    out = _parse_acceptance_criteria('["tests pass::test", "PR open"]')
    assert out == [
        {"text": "tests pass", "kind": "test"},
        {"text": "PR open", "kind": "llm"},
    ]
