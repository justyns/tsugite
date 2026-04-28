"""Tests for sanitize_for_summary: pre-summarization input hygiene.

Compaction feeds reconstructed events to an LLM. Two recurring noise sources
bloat the summary input:

1. Inlined scaffolding blocks (<attachment>, <context>, <skill_content>) that
   may end up in tool outputs or model responses.
2. Oversized <tsugite_execution_result> bodies (e.g. a web search dumping 4KB
   of results) that take up most of the chunk budget without adding equivalent
   summarization value.

The sanitizer elides scaffolding and truncates oversized tool outputs.
"""

from tsugite.daemon.memory import COMBINE_SYSTEM_PROMPT, SUMMARIZE_SYSTEM_PROMPT, sanitize_for_summary

MODEL = "openai:gpt-4o-mini"


class TestSanitizeAttachmentElision:
    def test_strips_attachment_block_keeps_surrounding_text(self):
        msg = {
            "role": "user",
            "content": (
                "Here is the file:\n"
                '<attachment name="AGENTS.md" type="text">\n'
                "Big workspace doc with hundreds of lines of text...\n"
                "Even more text here.\n"
                "</attachment>\n"
                "Please review."
            ),
        }
        out = sanitize_for_summary([msg], model=MODEL)
        c = out[0]["content"]
        assert "Big workspace doc" not in c
        assert "Please review." in c
        assert "Here is the file:" in c
        assert "elided" in c
        assert "AGENTS.md" in c

    def test_strips_context_block(self):
        msg = {
            "role": "user",
            "content": "<context>\nlots of context\nmore context\n</context>\nthe ask",
        }
        out = sanitize_for_summary([msg], model=MODEL)
        assert "lots of context" not in out[0]["content"]
        assert "the ask" in out[0]["content"]

    def test_strips_skill_content_block(self):
        msg = {
            "role": "user",
            "content": ('<skill_content name="webapp-testing">\nLong skill body...\n</skill_content>\nTail text.'),
        }
        out = sanitize_for_summary([msg], model=MODEL)
        assert "Long skill body" not in out[0]["content"]
        assert "Tail text." in out[0]["content"]
        assert "webapp-testing" in out[0]["content"]

    def test_multiple_blocks_all_elided(self):
        msg = {
            "role": "user",
            "content": (
                '<attachment name="A.md" type="text">aaa</attachment>\n'
                "between\n"
                '<attachment name="B.md" type="text">bbb</attachment>\n'
                "after"
            ),
        }
        out = sanitize_for_summary([msg], model=MODEL)
        c = out[0]["content"]
        assert "aaa" not in c
        assert "bbb" not in c
        assert "between" in c
        assert "after" in c
        assert c.count("elided") == 2


class TestSanitizeExecutionResultTruncation:
    def _wrap(self, body: str) -> str:
        return (
            '<tsugite_execution_result status="success" duration_ms="42">\n'
            f"<output>{body}</output>\n"
            "</tsugite_execution_result>"
        )

    def test_small_output_passes_through_unchanged(self):
        msg = {"role": "user", "content": self._wrap("short output")}
        out = sanitize_for_summary([msg], model=MODEL, per_message_token_budget=1000)
        assert out[0]["content"] == msg["content"]

    def test_oversized_output_truncated_preserves_wrapper(self):
        big = "x" * 20000
        msg = {"role": "user", "content": self._wrap(big)}
        out = sanitize_for_summary([msg], model=MODEL, per_message_token_budget=200)
        c = out[0]["content"]
        assert "<tsugite_execution_result" in c
        assert "</tsugite_execution_result>" in c
        assert "<output>" in c
        assert "</output>" in c
        assert big not in c
        assert "truncated" in c
        assert len(c) < 5000

    def test_oversized_truncation_includes_head_and_tail(self):
        body = "HEADHEADHEAD" + ("y" * 20000) + "TAILTAILTAIL"
        msg = {"role": "user", "content": self._wrap(body)}
        out = sanitize_for_summary([msg], model=MODEL, per_message_token_budget=200)
        c = out[0]["content"]
        assert "HEADHEADHEAD" in c
        assert "TAILTAILTAIL" in c

    def test_error_block_with_small_output_unchanged(self):
        wrapped = (
            '<tsugite_execution_result status="error">\n'
            "<output></output>\n"
            "<error>boom</error>\n"
            "</tsugite_execution_result>"
        )
        msg = {"role": "user", "content": wrapped}
        out = sanitize_for_summary([msg], model=MODEL, per_message_token_budget=1000)
        assert out[0]["content"] == msg["content"]


class TestSanitizeNonInterference:
    def test_plain_user_message_unchanged(self):
        msg = {"role": "user", "content": "what time is it"}
        out = sanitize_for_summary([msg], model=MODEL)
        assert out[0] == msg

    def test_plain_assistant_message_unchanged(self):
        msg = {"role": "assistant", "content": "the time is 5 PM"}
        out = sanitize_for_summary([msg], model=MODEL)
        assert out[0] == msg

    def test_empty_list(self):
        assert sanitize_for_summary([], model=MODEL) == []

    def test_non_string_content_passes_through(self):
        msg = {"role": "user", "content": [{"type": "text", "text": "hello"}]}
        out = sanitize_for_summary([msg], model=MODEL)
        assert out[0] == msg


class TestSanitizeIdempotence:
    def test_idempotent_on_attachment(self):
        msg = {
            "role": "user",
            "content": '<attachment name="X.md" type="text">junk</attachment>\nkeep',
        }
        once = sanitize_for_summary([msg], model=MODEL)
        twice = sanitize_for_summary(once, model=MODEL)
        assert once == twice

    def test_idempotent_on_truncated_output(self):
        big = "x" * 20000
        msg = {
            "role": "user",
            "content": (
                f'<tsugite_execution_result status="success">\n<output>{big}</output>\n</tsugite_execution_result>'
            ),
        }
        once = sanitize_for_summary([msg], model=MODEL, per_message_token_budget=200)
        twice = sanitize_for_summary(once, model=MODEL, per_message_token_budget=200)
        assert once == twice


class TestSanitizeCapitalAttachmentForm:
    """The `<Attachment: name>...</Attachment: name>` format used by
    `_format_attachment` (capital A, colon, name) must also be elided.
    """

    def test_strips_capital_attachment_block(self):
        msg = {
            "role": "user",
            "content": (
                "Here is the file:\n"
                "<Attachment: AGENTS.md>\n"
                "Big workspace doc with hundreds of lines of text...\n"
                "</Attachment: AGENTS.md>\n"
                "Please review."
            ),
        }
        out = sanitize_for_summary([msg], model=MODEL)
        c = out[0]["content"]
        assert "Big workspace doc" not in c
        assert "Please review." in c
        assert "Here is the file:" in c

    def test_strips_capital_attachment_with_simple_name(self):
        msg = {
            "role": "user",
            "content": "<Attachment: README>\ncontent here\n</Attachment: README>\nrest",
        }
        out = sanitize_for_summary([msg], model=MODEL)
        assert "content here" not in out[0]["content"]
        assert "rest" in out[0]["content"]


class TestSanitizeAttachmentBasenameElision:
    """Tool outputs that read known attachment files (e.g. read_note('MEMORY.md'))
    leak the file's verbatim contents into `code_execution` events. These bypass
    the size-based truncation when small.

    When `attachment_basenames` is provided, sanitize elides any
    <tsugite_execution_result> block referencing those basenames regardless of
    size.
    """

    def _wrap(self, body: str, code_hint: str = "") -> str:
        return (
            '<tsugite_execution_result status="success" duration_ms="42">\n'
            f"{code_hint}"
            f"<output>{body}</output>\n"
            "</tsugite_execution_result>"
        )

    def test_small_tool_output_referencing_attachment_is_elided(self):
        # Small enough to slip past the size budget
        body = "# AGENTS.md\nshort file content here"
        msg = {
            "role": "user",
            "content": self._wrap(
                body, code_hint="<code>read_note('AGENTS.md')</code>\n"
            ),
        }
        out = sanitize_for_summary(
            [msg],
            model=MODEL,
            per_message_token_budget=10000,
            attachment_basenames={"AGENTS.md", "MEMORY.md"},
        )
        c = out[0]["content"]
        assert "short file content here" not in c
        assert "<tsugite_execution_result" in c
        assert "</tsugite_execution_result>" in c

    def test_tool_output_unrelated_to_attachments_unchanged(self):
        body = "search results for kittens"
        msg = {
            "role": "user",
            "content": self._wrap(body, code_hint="<code>web_search('kittens')</code>\n"),
        }
        out = sanitize_for_summary(
            [msg],
            model=MODEL,
            per_message_token_budget=10000,
            attachment_basenames={"AGENTS.md", "MEMORY.md"},
        )
        assert "search results for kittens" in out[0]["content"]

    def test_attachment_basenames_match_in_output_body(self):
        # Attachment basename in the output body itself, not just the code arg
        body = "# AGENTS.md\nthis is the agents config\n## section"
        msg = {"role": "user", "content": self._wrap(body)}
        out = sanitize_for_summary(
            [msg],
            model=MODEL,
            per_message_token_budget=10000,
            attachment_basenames={"AGENTS.md"},
        )
        c = out[0]["content"]
        assert "this is the agents config" not in c

    def test_no_attachment_basenames_keeps_existing_behavior(self):
        body = "# AGENTS.md\nshort file content here"
        msg = {
            "role": "user",
            "content": self._wrap(body, code_hint="<code>read_note('AGENTS.md')</code>\n"),
        }
        out_no_basenames = sanitize_for_summary([msg], model=MODEL, per_message_token_budget=10000)
        out_empty_basenames = sanitize_for_summary(
            [msg], model=MODEL, per_message_token_budget=10000, attachment_basenames=set()
        )
        assert out_no_basenames[0]["content"] == msg["content"]
        assert out_empty_basenames[0]["content"] == msg["content"]


class TestSummarizerPromptDirective:
    """Both summarizer prompts must instruct the LLM to skip auto-attached
    workspace files.
    """

    def test_summarize_prompt_mentions_attachments(self):
        assert "auto-attached" in SUMMARIZE_SYSTEM_PROMPT.lower() or "attached" in SUMMARIZE_SYSTEM_PROMPT.lower()
        assert "MEMORY.md" in SUMMARIZE_SYSTEM_PROMPT or "AGENTS.md" in SUMMARIZE_SYSTEM_PROMPT

    def test_combine_prompt_mentions_attachments(self):
        assert "auto-attached" in COMBINE_SYSTEM_PROMPT.lower() or "attached" in COMBINE_SYSTEM_PROMPT.lower()
        assert "MEMORY.md" in COMBINE_SYSTEM_PROMPT or "AGENTS.md" in COMBINE_SYSTEM_PROMPT
