"""Tests for content block extraction and injection."""

import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from tsugite.core.content_blocks import extract_content_blocks, write_content_blocks_to_files


class TestExtractContentBlocks:
    def test_single_content_block(self):
        text = 'Some text\n<content name="my_var">hello world</content>\nMore text'
        cleaned, blocks = extract_content_blocks(text)
        assert blocks == {"my_var": "hello world"}
        assert "<content" not in cleaned
        assert "Some text" in cleaned
        assert "More text" in cleaned

    def test_single_tsu_content_block(self):
        text = '<tsu:content name="my_var">hello world</tsu:content>'
        cleaned, blocks = extract_content_blocks(text)
        assert blocks == {"my_var": "hello world"}
        assert "<tsu:content" not in cleaned

    def test_multiple_blocks(self):
        text = '<content name="a">alpha</content>\n<content name="b">beta</content>'
        cleaned, blocks = extract_content_blocks(text)
        assert blocks == {"a": "alpha", "b": "beta"}

    def test_mixed_content_and_tsu_content(self):
        text = '<content name="a">alpha</content>\n<tsu:content name="b">beta</tsu:content>'
        cleaned, blocks = extract_content_blocks(text)
        assert blocks == {"a": "alpha", "b": "beta"}

    def test_triple_double_quotes(self):
        text = '<content name="py_file">\ndef foo():\n    """docstring"""\n    pass\n</content>'
        _, blocks = extract_content_blocks(text)
        assert '"""docstring"""' in blocks["py_file"]

    def test_triple_single_quotes(self):
        text = "<content name=\"py_file\">\ndef foo():\n    '''docstring'''\n    pass\n</content>"
        _, blocks = extract_content_blocks(text)
        assert "'''docstring'''" in blocks["py_file"]

    def test_triple_backticks(self):
        text = '<content name="md_file">\n# Heading\n```python\nprint("hi")\n```\n</content>'
        _, blocks = extract_content_blocks(text)
        assert "```python" in blocks["md_file"]
        assert blocks["md_file"].endswith("```")

    def test_backslashes(self):
        text = r'<content name="code">path = "C:\\Users\\test"\nnewline = "\n"</content>'
        _, blocks = extract_content_blocks(text)
        assert "\\\\Users" in blocks["code"]

    def test_closing_content_inside_tsu_content(self):
        text = '<tsu:content name="tricky">This has </content> inside</tsu:content>'
        _, blocks = extract_content_blocks(text)
        assert blocks["tricky"] == "This has </content> inside"

    def test_no_blocks(self):
        text = "Just regular text with no blocks"
        cleaned, blocks = extract_content_blocks(text)
        assert blocks == {}
        assert cleaned == text

    def test_empty_content(self):
        text = '<content name="empty"></content>'
        _, blocks = extract_content_blocks(text)
        assert blocks == {"empty": ""}

    def test_strips_leading_trailing_newline(self):
        text = '<content name="file">\nline1\nline2\n</content>'
        _, blocks = extract_content_blocks(text)
        assert blocks["file"] == "line1\nline2"

    def test_blocks_between_code_blocks(self):
        text = (
            "Thought: writing files\n\n"
            "```python\nwrite_file('out.py', content=py_code)\n```\n\n"
            '<content name="py_code">\nprint("hello")\n</content>'
        )
        cleaned, blocks = extract_content_blocks(text)
        assert "py_code" in blocks
        assert "```python" in cleaned
        assert "write_file" in cleaned

    def test_blocks_before_code(self):
        text = (
            '<content name="data">some data</content>\n\n' "Thought: using the data\n\n" "```python\nprint(data)\n```"
        )
        cleaned, blocks = extract_content_blocks(text)
        assert blocks == {"data": "some data"}
        assert "```python" in cleaned

    def test_thought_extraction_not_affected(self):
        text = (
            "Thought: I need to write a file\n\n"
            "```python\nwrite_file('f.py', content=code)\n```\n\n"
            '<content name="code">\ndef main():\n    pass\n</content>'
        )
        cleaned, blocks = extract_content_blocks(text)
        assert "I need to write a file" in cleaned
        assert blocks["code"] == "def main():\n    pass"


class TestWriteContentBlocksToFiles:
    def test_files_written(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            blocks = {"var1": "hello", "var2": "world"}
            paths = write_content_blocks_to_files(blocks, tmpdir)
            assert set(paths.keys()) == {"var1", "var2"}
            for name, path in paths.items():
                assert os.path.exists(path)
                assert Path(path).read_text() == blocks[name]

    def test_same_content_same_filename(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            paths1 = write_content_blocks_to_files({"a": "same"}, tmpdir)
            paths2 = write_content_blocks_to_files({"b": "same"}, tmpdir)
            assert os.path.basename(paths1["a"]) == os.path.basename(paths2["b"])

    def test_different_content_different_filename(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = write_content_blocks_to_files({"a": "hello", "b": "world"}, tmpdir)
            assert os.path.basename(paths["a"]) != os.path.basename(paths["b"])

    def test_content_roundtrip_exact(self):
        content = 'def foo():\n    """Has triple quotes"""\n    x = "backslash\\n"\n    pass\n'
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = write_content_blocks_to_files({"code": content}, tmpdir)
            assert Path(paths["code"]).read_text() == content

    def test_empty_blocks(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = write_content_blocks_to_files({}, tmpdir)
            assert paths == {}


class TestContentBlocksIntegration:
    """Integration tests with mock LLM."""

    @pytest.mark.asyncio
    async def test_content_block_write_file(self, mock_litellm_response):
        """LLM returns content block + write_file call, file gets written correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            outfile = os.path.join(tmpdir, "output.py")

            with patch("litellm.acompletion", new_callable=AsyncMock) as mock_acompletion:
                mock_acompletion.return_value = mock_litellm_response(
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
    async def test_content_block_edit_file(self, mock_litellm_response):
        """Content blocks work for edit_file old_string/new_string params."""
        with tempfile.TemporaryDirectory() as tmpdir:
            target = os.path.join(tmpdir, "target.py")
            Path(target).write_text('def foo():\n    """old docstring"""\n    pass\n')

            with patch("litellm.acompletion", new_callable=AsyncMock) as mock_acompletion:
                mock_acompletion.return_value = mock_litellm_response(
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
    async def test_content_blocks_in_memory(self, mock_litellm_response):
        """Content blocks stored in memory and reconstructed in messages."""
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

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock:
            mock.side_effect = mock_acompletion

            from tsugite.core.agent import TsugiteAgent

            agent = TsugiteAgent(
                model_string="openai:gpt-4o-mini",
                tools=[],
                instructions="",
                max_turns=5,
            )
            await agent.run("test task")

        # Check that content blocks are in memory
        assert agent.memory.steps[0].content_blocks == {"my_data": "test content"}

        # Check that messages include content blocks
        messages = agent._build_messages()
        # Find the assistant message for the first step
        assistant_msgs = [m for m in messages if m["role"] == "assistant"]
        assert any('<content name="my_data">' in m["content"] for m in assistant_msgs)

    @pytest.mark.asyncio
    async def test_content_block_variable_persists(self, mock_litellm_response):
        """Content block variable accessible across turns."""
        call_count = 0

        async def mock_acompletion(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return mock_litellm_response(
                    "```python\nprint(data)\n```\n\n" '<content name="data">\npersisted value\n</content>'
                )
            else:
                return mock_litellm_response("```python\nfinal_answer(data)\n```")

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock:
            mock.side_effect = mock_acompletion

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
    async def test_multiple_content_blocks_same_turn(self, mock_litellm_response):
        """Multiple content blocks in one response all get injected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_a = os.path.join(tmpdir, "a.txt")
            file_b = os.path.join(tmpdir, "b.txt")

            with patch("litellm.acompletion", new_callable=AsyncMock) as mock_acompletion:
                mock_acompletion.return_value = mock_litellm_response(
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
