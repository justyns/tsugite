"""Tests for content block extraction and injection."""

import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from tsugite.core.content_blocks import extract_content_blocks, write_content_blocks_to_files
from tsugite.providers.base import CompletionResponse, Usage


def _resp(content: str) -> CompletionResponse:
    return CompletionResponse(content=content, usage=Usage(total_tokens=100), cost=0.001)


class TestExtractContentBlocks:
    def test_single_content_block(self):
        text = 'Some text\n<content name="my_var">hello world</content>\nMore text'
        cleaned, blocks = extract_content_blocks(text)
        assert blocks == {"my_var": "hello world"}
        assert "<content" not in cleaned
        assert "Some text" in cleaned
        assert "More text" in cleaned

    def test_multiple_content_blocks(self):
        text = (
            '<content name="a">alpha</content>\n'
            "middle\n"
            '<content name="b">beta</content>'
        )
        cleaned, blocks = extract_content_blocks(text)
        assert blocks == {"a": "alpha", "b": "beta"}
        assert "middle" in cleaned

    def test_multiline_content(self):
        text = '<content name="code">\ndef foo():\n    return 42\n</content>'
        cleaned, blocks = extract_content_blocks(text)
        assert "def foo():" in blocks["code"]
        assert "return 42" in blocks["code"]

    def test_no_content_blocks(self):
        text = "Just regular text without any content blocks"
        cleaned, blocks = extract_content_blocks(text)
        assert blocks == {}
        assert cleaned == text

    def test_content_with_backticks(self):
        text = '<content name="code">\n```python\nprint("hello")\n```\n</content>'
        cleaned, blocks = extract_content_blocks(text)
        assert "```python" in blocks["code"]
        assert 'print("hello")' in blocks["code"]

    def test_content_block_preserves_indentation(self):
        text = '<content name="py">\n    def foo():\n        pass\n</content>'
        _, blocks = extract_content_blocks(text)
        assert "    def foo():" in blocks["py"]
        assert "        pass" in blocks["py"]

    def test_empty_content_block(self):
        text = '<content name="empty"></content>'
        _, blocks = extract_content_blocks(text)
        assert blocks == {"empty": ""}

    def test_content_block_with_special_chars(self):
        text = '<content name="data">{"key": "value", "list": [1, 2, 3]}</content>'
        _, blocks = extract_content_blocks(text)
        assert blocks["data"] == '{"key": "value", "list": [1, 2, 3]}'


class TestWriteContentBlocksToFiles:
    def test_write_creates_files(self):
        blocks = {"a": "content a", "b": "content b"}
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = write_content_blocks_to_files(blocks, tmpdir)
            assert len(paths) == 2
            for name, path in paths.items():
                assert Path(path).exists()
                assert Path(path).read_text() == blocks[name]

    def test_write_empty_blocks(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = write_content_blocks_to_files({}, tmpdir)
            assert paths == {}


class TestContentBlocksIntegration:
    """Integration tests with mock LLM."""

    @pytest.mark.asyncio
    async def test_content_block_write_file(self, mock_provider, mock_litellm_response):
        with tempfile.TemporaryDirectory() as tmpdir:
            outfile = os.path.join(tmpdir, "output.py")

            mock_provider.acompletion.return_value = mock_litellm_response(
                "Thought: Writing a Python file\n\n"
                "```python\n"
                f'write_file("{outfile}", content=py_code)\n'
                'final_answer("done")\n'
                "```\n\n"
                '<content name="py_code">\n'
                "def hello():\n"
                '    """Says hello"""\n'
                '    print("hello")\n'
                "</content>"
            )

            from tsugite.core.agent import TsugiteAgent
            from tsugite.core.tools import create_tool_from_function

            def write_file(path: str, content: str) -> str:
                """Write content to a file."""
                Path(path).write_text(content)
                return f"Wrote {path}"

            tool = create_tool_from_function(write_file)
            agent = TsugiteAgent(
                model_string="openai:gpt-4o-mini",
                tools=[tool],
                instructions="",
                max_turns=3,
            )
            result = await agent.run("Write a Python file")

            assert result == "done"
            written = Path(outfile).read_text()
            assert '"""Says hello"""' in written
            assert 'print("hello")' in written

    @pytest.mark.asyncio
    async def test_content_block_edit_file(self, mock_provider, mock_litellm_response):
        with tempfile.TemporaryDirectory() as tmpdir:
            target = os.path.join(tmpdir, "target.py")
            Path(target).write_text('def foo():\n    """old docstring"""\n    pass\n')

            mock_provider.acompletion.return_value = mock_litellm_response(
                "Thought: Editing the file\n\n"
                "```python\n"
                f'result = edit_file("{target}", old_string=old, new_string=new)\n'
                "print(result)\n"
                'final_answer("edited")\n'
                "```\n\n"
                '<content name="old">\n'
                '    """old docstring"""\n'
                "</content>\n\n"
                '<content name="new">\n'
                '    """new docstring with ```backticks```"""\n'
                "</content>"
            )

            from tsugite.core.agent import TsugiteAgent
            from tsugite.core.tools import create_tool_from_function

            def edit_file(path: str, old_string: str, new_string: str) -> str:
                """Edit a file by replacing old_string with new_string."""
                content = Path(path).read_text()
                content = content.replace(old_string, new_string)
                Path(path).write_text(content)
                return f"Edited {path}"

            tool = create_tool_from_function(edit_file)
            agent = TsugiteAgent(
                model_string="openai:gpt-4o-mini",
                tools=[tool],
                instructions="",
                max_turns=3,
            )
            result = await agent.run("Edit the file")

            assert result == "edited"
            edited = Path(target).read_text()
            assert "new docstring" in edited
            assert "```backticks```" in edited

    @pytest.mark.asyncio
    async def test_content_blocks_in_memory(self, mock_provider, mock_litellm_response):
        call_count = 0

        async def mock_acompletion(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return mock_litellm_response(
                    "Thought: Setting up\n\n"
                    "```python\n"
                    "print(my_data)\n"
                    "```\n\n"
                    '<content name="my_data">\ntest content\n</content>'
                )
            else:
                return mock_litellm_response('```python\nfinal_answer("done")\n```')

        mock_provider.acompletion.side_effect = mock_acompletion

        from tsugite.core.agent import TsugiteAgent

        agent = TsugiteAgent(
            model_string="openai:gpt-4o-mini",
            tools=[],
            instructions="",
            max_turns=5,
        )
        await agent.run("test task")

        assert agent.memory.steps[0].content_blocks == {"my_data": "test content"}

        messages = agent._build_messages()
        assistant_msgs = [m for m in messages if m["role"] == "assistant"]
        assert any('<content name="my_data">' in m["content"] for m in assistant_msgs)

    @pytest.mark.asyncio
    async def test_content_block_variable_persists(self, mock_provider, mock_litellm_response):
        call_count = 0

        async def mock_acompletion(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return mock_litellm_response(
                    '```python\nprint(data)\n```\n\n<content name="data">\npersisted value\n</content>'
                )
            else:
                return mock_litellm_response("```python\nfinal_answer(data)\n```")

        mock_provider.acompletion.side_effect = mock_acompletion

        from tsugite.core.agent import TsugiteAgent

        agent = TsugiteAgent(
            model_string="openai:gpt-4o-mini",
            tools=[],
            instructions="",
            max_turns=5,
        )
        result = await agent.run("test persistence")

        assert result == "persisted value"

    @pytest.mark.asyncio
    async def test_multiple_content_blocks_same_turn(self, mock_provider, mock_litellm_response):
        with tempfile.TemporaryDirectory() as tmpdir:
            file_a = os.path.join(tmpdir, "a.txt")
            file_b = os.path.join(tmpdir, "b.txt")

            mock_provider.acompletion.return_value = mock_litellm_response(
                "```python\n"
                f'write_file("{file_a}", content=content_a)\n'
                f'write_file("{file_b}", content=content_b)\n'
                'final_answer("done")\n'
                "```\n\n"
                '<content name="content_a">\nfile a content\n</content>\n'
                '<content name="content_b">\nfile b content\n</content>'
            )

            from tsugite.core.agent import TsugiteAgent
            from tsugite.core.tools import create_tool_from_function

            def write_file(path: str, content: str) -> str:
                """Write content to a file."""
                Path(path).write_text(content)
                return f"Wrote {path}"

            tool = create_tool_from_function(write_file)
            agent = TsugiteAgent(
                model_string="openai:gpt-4o-mini",
                tools=[tool],
                instructions="",
                max_turns=3,
            )
            await agent.run("Write files")

            assert Path(file_a).read_text() == "file a content"
            assert Path(file_b).read_text() == "file b content"
