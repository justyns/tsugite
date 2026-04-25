"""The LLM sometimes writes a triple-quoted Python string containing a
markdown code fence (real newlines, literal triple backticks). The current
parser terminates the code block at the first `\\n```` it sees, which lands
inside the string and produces an unterminated-triple-quote SyntaxError.

Reproduces the #201 re-run failure: the original Python code was valid; the
parser is the thing that mangled it.
"""

import ast

import pytest

from tsugite.core.agent import TsugiteAgent


def _agent():
    return TsugiteAgent(
        model_string="openai:gpt-4o-mini",
        tools=[],
        instructions="",
        max_turns=5,
    )


@pytest.mark.asyncio
async def test_parse_triple_quoted_body_with_real_newlines_and_inner_fences():
    llm_response = (
        "Posting a comment.\n\n"
        "```python\n"
        'body = """example:\n'
        "\n"
        "```\n"
        "inner\n"
        "```\n"
        "\n"
        'end.\n'
        '"""\n'
        "post_comment(body)\n"
        "```"
    )

    parsed = _agent()._parse_response_from_text(llm_response)

    assert "post_comment(body)" in parsed.code, (
        f"Parser truncated at an inner backtick. Got: {parsed.code!r}"
    )
    ast.parse(parsed.code)


@pytest.mark.asyncio
async def test_parse_triple_single_quoted_body_with_inner_fences():
    llm_response = (
        "```python\n"
        "body = '''\n"
        "example:\n"
        "```\n"
        "inner\n"
        "```\n"
        "end\n"
        "'''\n"
        "print(body)\n"
        "```"
    )

    parsed = _agent()._parse_response_from_text(llm_response)
    assert "print(body)" in parsed.code, f"Got: {parsed.code!r}"
    ast.parse(parsed.code)


@pytest.mark.asyncio
async def test_existing_single_quoted_escape_case_still_works():
    """Regression guard for the case covered by the older parser: fences
    inside a single-quoted string with `\\n` escape sequences.
    """
    llm_response = (
        "```python\n"
        'comment_body = "See example:\\n\\n```python\\nx = 1\\n```\\nthat is all."\n'
        "post_comment(comment_body)\n"
        "```"
    )
    parsed = _agent()._parse_response_from_text(llm_response)
    assert "post_comment(comment_body)" in parsed.code
    ast.parse(parsed.code)


@pytest.mark.asyncio
async def test_multiple_top_level_blocks_still_counted():
    """Picking the right close fence must not change the multi-block count
    the agent relies on for its warning.
    """
    llm_response = (
        "```python\n"
        'x = """\n```\ny\n```\n"""\n'
        "```\n\n"
        "some prose\n\n"
        "```python\n"
        "final_answer(x)\n"
        "```"
    )
    parsed = _agent()._parse_response_from_text(llm_response)
    assert parsed.num_code_blocks == 2, f"got {parsed.num_code_blocks}"
