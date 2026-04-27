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

from tsugite.daemon.memory import sanitize_for_summary

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
