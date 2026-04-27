"""Tests for attachment management, resolution, and agent integration."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from tsugite.agent_preparation import AgentPreparer, resolve_agent_config_attachments
from tsugite.attachments import (
    add_attachment,
    get_attachment,
    list_attachments,
    remove_attachment,
    search_attachments,
)
from tsugite.attachments.base import AttachmentContentType
from tsugite.md_agents import (
    Agent,
    AgentConfig,
    AttachmentSpec,
    parse_agent,
    parse_agent_file,
    validate_agent_execution,
)
from tsugite.utils import resolve_attachments


class TestAttachmentStorage:
    """Test attachment storage and retrieval."""

    def test_add_and_get_inline_attachment(self, tmp_path, monkeypatch):
        """Test adding and retrieving an inline attachment."""
        monkeypatch.setenv("HOME", str(tmp_path))

        content = "Test attachment content"
        add_attachment("test", source="inline", content=content)

        result = get_attachment("test")
        assert result is not None
        source, retrieved_content = result
        assert source == "inline"
        assert retrieved_content == content

    def test_add_and_get_file_reference(self, tmp_path, monkeypatch):
        """Test adding and retrieving a file reference."""
        monkeypatch.setenv("HOME", str(tmp_path))

        file_path = tmp_path / "test.txt"
        file_path.write_text("File content")

        add_attachment("test", source=str(file_path))

        result = get_attachment("test")
        assert result is not None
        source, content = result
        assert source == str(file_path)
        assert content is None  # Not stored inline

    def test_get_nonexistent_attachment(self, tmp_path, monkeypatch):
        """Test getting an attachment that doesn't exist."""
        monkeypatch.setenv("HOME", str(tmp_path))

        result = get_attachment("nonexistent")
        assert result is None

    def test_update_existing_attachment(self, tmp_path, monkeypatch):
        """Test updating an existing attachment."""
        monkeypatch.setenv("HOME", str(tmp_path))

        add_attachment("test", source="inline", content="Original content")
        add_attachment("test", source="inline", content="Updated content")

        result = get_attachment("test")
        assert result is not None
        source, content = result
        assert content == "Updated content"
        assert source == "inline"

    def test_list_attachments(self, tmp_path, monkeypatch):
        """Test listing all attachments."""
        monkeypatch.setenv("HOME", str(tmp_path))

        add_attachment("att1", source="inline", content="Content 1")
        add_attachment("att2", source="inline", content="Content 2")
        add_attachment("att3", source="/path/to/file")

        attachments = list_attachments()
        assert len(attachments) == 3
        assert "att1" in attachments
        assert "att2" in attachments
        assert "att3" in attachments

    def test_remove_attachment(self, tmp_path, monkeypatch):
        """Test removing an attachment."""
        monkeypatch.setenv("HOME", str(tmp_path))

        add_attachment("test", source="inline", content="Content")

        result = remove_attachment("test")
        assert result is True
        assert get_attachment("test") is None

    def test_remove_nonexistent_attachment(self, tmp_path, monkeypatch):
        """Test removing an attachment that doesn't exist."""
        monkeypatch.setenv("HOME", str(tmp_path))

        result = remove_attachment("nonexistent")
        assert result is False

    def test_search_attachments_by_alias(self, tmp_path, monkeypatch):
        """Test searching attachments by alias."""
        monkeypatch.setenv("HOME", str(tmp_path))

        add_attachment("python_code", source="inline", content="def foo(): pass")
        add_attachment("python_docs", source="inline", content="Documentation")
        add_attachment("java_code", source="inline", content="class Foo {}")

        results = search_attachments("python")
        assert len(results) == 2
        assert "python_code" in results
        assert "python_docs" in results
        assert "java_code" not in results

    def test_search_attachments_by_source(self, tmp_path, monkeypatch):
        """Test searching attachments by source."""
        monkeypatch.setenv("HOME", str(tmp_path))

        add_attachment("code1", source="https://example.com/api.md")
        add_attachment("code2", source="docs/readme.md")
        add_attachment("code3", source="https://example.com/guide.md")

        results = search_attachments("example.com")
        assert len(results) == 2
        assert "code1" in results
        assert "code3" in results
        assert "code2" not in results

    def test_search_case_insensitive(self, tmp_path, monkeypatch):
        """Test that search is case-insensitive."""
        monkeypatch.setenv("HOME", str(tmp_path))

        add_attachment("MyCode", source="MyFile.py")

        results = search_attachments("mycode")
        assert len(results) == 1
        assert "MyCode" in results

        results = search_attachments("myfile")
        assert len(results) == 1

    def test_empty_alias_error(self, tmp_path, monkeypatch):
        """Test that empty alias raises error."""
        monkeypatch.setenv("HOME", str(tmp_path))

        with pytest.raises(ValueError, match="alias cannot be empty"):
            add_attachment("", source="inline", content="Content")

        with pytest.raises(ValueError, match="alias cannot be empty"):
            add_attachment("   ", source="inline", content="Content")

    def test_inline_without_content_error(self, tmp_path, monkeypatch):
        """Test that inline attachment without content raises error."""
        monkeypatch.setenv("HOME", str(tmp_path))

        with pytest.raises(ValueError, match="Inline attachments require content"):
            add_attachment("test", source="inline")

    def test_attachments_json_format(self, tmp_path, monkeypatch):
        """Test that attachments.json has correct structure."""
        from tsugite.attachments import get_attachments_path

        monkeypatch.setenv("HOME", str(tmp_path))

        add_attachment("inline_test", source="inline", content="Content")
        add_attachment("file_test", source="/path/to/file")

        attachments_path = get_attachments_path()
        assert attachments_path.exists()

        with open(attachments_path) as f:
            data = json.load(f)

        assert "attachments" in data

        assert data["attachments"]["inline_test"]["content"] == "Content"
        assert data["attachments"]["inline_test"]["source"] == "inline"

        assert "content" not in data["attachments"]["file_test"]
        assert data["attachments"]["file_test"]["source"] == "/path/to/file"

        for key in ("inline_test", "file_test"):
            assert "created" in data["attachments"][key]
            assert "updated" in data["attachments"][key]


class TestAttachmentResolution:
    """Test resolving attachment references to content."""

    def test_resolve_inline_attachment(self, tmp_path, monkeypatch):
        """Test resolving an inline attachment."""
        monkeypatch.setenv("HOME", str(tmp_path))

        add_attachment("mycode", source="inline", content="def hello(): pass")

        results = resolve_attachments(["mycode"])
        assert len(results) == 1
        assert results[0].name == "mycode"
        assert results[0].content == "def hello(): pass"

    def test_resolve_file_attachment(self, tmp_path, monkeypatch):
        """Test resolving a file attachment."""
        monkeypatch.setenv("HOME", str(tmp_path))

        test_file = tmp_path / "test.txt"
        test_file.write_text("File content")

        add_attachment("myfile", source=str(test_file))

        results = resolve_attachments(["myfile"])
        assert len(results) == 1
        assert results[0].name == "test.txt"
        assert results[0].content == "File content"

    def test_resolve_url_attachment(self, tmp_path, monkeypatch):
        """Test resolving a URL attachment."""
        monkeypatch.setenv("HOME", str(tmp_path))

        add_attachment("myurl", source="https://example.com/doc.md")

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_response = mock_urlopen.return_value.__enter__.return_value
            mock_response.headers.get.return_value = "text/plain"
            mock_response.read.return_value = b"URL content here"

            results = resolve_attachments(["myurl"])
            assert len(results) == 1
            assert results[0].name == "doc.md"
            assert results[0].content == "URL content here"

    def test_resolve_multiple_attachments(self, tmp_path, monkeypatch):
        """Test resolving multiple attachments."""
        monkeypatch.setenv("HOME", str(tmp_path))

        add_attachment("alias1", source="inline", content="Alias content")

        test_file = tmp_path / "file.txt"
        test_file.write_text("File content")
        add_attachment("file1", source=str(test_file))

        add_attachment("url1", source="https://example.com/doc.md")

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_response = mock_urlopen.return_value.__enter__.return_value
            mock_response.headers.get.return_value = "text/plain"
            mock_response.read.return_value = b"URL content"

            results = resolve_attachments(["alias1", "file1", "url1"])

            assert len(results) == 3
            assert results[0].name == "alias1"
            assert results[0].content == "Alias content"
            assert results[1].name == "file.txt"
            assert results[1].content == "File content"
            assert results[2].name == "doc.md"
            assert results[2].content == "URL content"

    def test_resolve_nonexistent_attachment_error(self, tmp_path, monkeypatch):
        """Test error when attachment doesn't exist."""
        monkeypatch.setenv("HOME", str(tmp_path))

        with pytest.raises(ValueError, match="Attachment not found"):
            resolve_attachments(["nonexistent"])

    def test_resolve_empty_list(self, tmp_path, monkeypatch):
        """Test resolving empty list."""
        monkeypatch.setenv("HOME", str(tmp_path))

        results = resolve_attachments([])
        assert results == []

    def test_caching_works(self, tmp_path, monkeypatch):
        """Test that caching works for file attachments."""
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))

        test_file = tmp_path / "test.txt"
        test_file.write_text("Original content")

        add_attachment("myfile", source=str(test_file))
        results1 = resolve_attachments(["myfile"])
        assert results1[0].content == "Original content"

        test_file.write_text("Modified content")

        results2 = resolve_attachments(["myfile"])
        assert results2[0].content == "Original content"  # Still cached

        results3 = resolve_attachments(["myfile"], refresh_cache=True)
        assert results3[0].content == "Modified content"

    def test_youtube_handler_integration(self, tmp_path, monkeypatch):
        """Test YouTube handler integration."""
        monkeypatch.setenv("HOME", str(tmp_path))

        add_attachment("tutorial", source="https://youtube.com/watch?v=test123")

        mock_transcript = [
            {"start": 0.0, "text": "Hello world"},
            {"start": 5.0, "text": "This is a test"},
        ]

        with patch("youtube_transcript_api.YouTubeTranscriptApi") as mock_api:
            mock_api.get_transcript.return_value = mock_transcript

            results = resolve_attachments(["tutorial"])
            assert len(results) == 1
            assert results[0].name == "youtube:test123"
            assert "[00:00] Hello world" in results[0].content
            assert "[00:05] This is a test" in results[0].content

    def test_cache_metadata_structure(self, tmp_path, monkeypatch):
        """Test that cache metadata has expected structure."""
        from tsugite.cache import get_cache_key, list_cache

        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))

        test_file = tmp_path / "test.txt"
        test_file.write_text("Test content")
        add_attachment("myfile", source=str(test_file))

        resolve_attachments(["myfile"])

        cache_entries = list_cache()
        cache_key = get_cache_key(str(test_file))

        assert cache_key in cache_entries
        cache_info = cache_entries[cache_key]
        assert "source" in cache_info
        assert "cached_at" in cache_info
        assert "size" in cache_info
        assert cache_info["source"] == str(test_file)
        assert cache_info["size"] > 0


class TestAgentAttachments:
    """Test attachment support in agent definitions."""

    def test_agent_with_attachments_field(self):
        """Test parsing agent with attachments field."""
        agent_text = """---
name: test_agent
model: openai:gpt-4o-mini
tools: []
attachments:
  - coding-standards
  - api-docs
---

Task: {{ user_prompt }}
"""
        agent = parse_agent(agent_text)
        assert agent.config.attachments == ["coding-standards", "api-docs"]

    def test_agent_without_attachments_field(self):
        """Test parsing agent without attachments field."""
        agent_text = """---
name: test_agent
model: openai:gpt-4o-mini
tools: []
---

Task: {{ user_prompt }}
"""
        agent = parse_agent(agent_text)
        assert agent.config.attachments == []

    def test_agent_with_empty_attachments(self):
        """Test parsing agent with empty attachments list."""
        agent_text = """---
name: test_agent
model: openai:gpt-4o-mini
tools: []
attachments: []
---

Task: {{ user_prompt }}
"""
        agent = parse_agent(agent_text)
        assert agent.config.attachments == []

    def test_agent_with_single_attachment(self):
        """Test parsing agent with single attachment."""
        agent_text = """---
name: test_agent
model: openai:gpt-4o-mini
tools: []
attachments:
  - style-guide
---

Task: {{ user_prompt }}
"""
        agent = parse_agent(agent_text)
        assert agent.config.attachments == ["style-guide"]

    def test_agent_attachments_in_agent_info(self, tmp_path, monkeypatch):
        """Test that attachments appear in get_agent_info."""
        from tsugite.agent_runner import get_agent_info

        monkeypatch.setenv("HOME", str(tmp_path))

        agent_text = """---
name: test_agent
model: openai:gpt-4o-mini
extends: none
tools: []
attachments:
  - coding-standards
  - security-guide
---

Task: {{ user_prompt }}
"""
        agent_file = tmp_path / "test_agent.md"
        agent_file.write_text(agent_text)

        agent_info = get_agent_info(agent_file)

        assert "attachments" in agent_info
        assert agent_info["attachments"] == ["coding-standards", "security-guide"]

    def test_agent_attachments_resolve(self, tmp_path, monkeypatch):
        """Test full integration: agent attachments resolve correctly."""
        monkeypatch.setenv("HOME", str(tmp_path))

        add_attachment("style-guide", source="inline", content="Use tabs for indentation")

        agent_text = """---
name: code_reviewer
model: openai:gpt-4o-mini
tools: []
attachments:
  - style-guide
---

You are a code reviewer.

Task: {{ user_prompt }}
"""
        agent = parse_agent(agent_text)

        resolved = resolve_attachments(agent.config.attachments)

        assert len(resolved) == 1
        assert resolved[0].name == "style-guide"


class TestAttachmentSpec:
    """Test the AttachmentSpec dict-form syntax in agent frontmatter."""

    def test_dict_form_all_fields_round_trip(self):
        """Dict spec with every field parses into an AttachmentSpec with the right values."""
        agent_text = """---
name: t
model: openai:gpt-4o-mini
attachments:
  - path: memory/topics/*.md
    mode: index
    name: topic_index
    assign: topics
    attach: false
    index_format: first_heading
    max_entries: 10
---

body
"""
        agent = parse_agent(agent_text)
        assert len(agent.config.attachments) == 1
        spec = agent.config.attachments[0]
        assert isinstance(spec, AttachmentSpec)
        assert spec.path == "memory/topics/*.md"
        assert spec.mode == "index"
        assert spec.name == "topic_index"
        assert spec.assign == "topics"
        assert spec.attach is False
        assert spec.index_format == "first_heading"
        assert spec.max_entries == 10

    def test_string_form_still_works(self):
        """Backward compat: list of strings still parses unchanged."""
        agent_text = """---
name: t
model: openai:gpt-4o-mini
attachments:
  - foo.md
  - "{{ today() }}.md"
---

body
"""
        agent = parse_agent(agent_text)
        assert agent.config.attachments == ["foo.md", "{{ today() }}.md"]

    def test_mixed_strings_and_dicts(self):
        """Mixed list of strings and dict specs."""
        agent_text = """---
name: t
model: openai:gpt-4o-mini
attachments:
  - legacy.md
  - path: dynamic.md
    assign: my_var
---

body
"""
        agent = parse_agent(agent_text)
        assert len(agent.config.attachments) == 2
        assert agent.config.attachments[0] == "legacy.md"
        assert isinstance(agent.config.attachments[1], AttachmentSpec)
        assert agent.config.attachments[1].assign == "my_var"

    def test_invalid_assign_identifier_rejected(self):
        """assign must be a valid Python identifier."""
        agent_text = """---
name: t
model: openai:gpt-4o-mini
attachments:
  - path: foo.md
    assign: 3foo
---

body
"""
        with pytest.raises(ValueError, match="valid Python identifier"):
            parse_agent(agent_text)

    def test_assign_with_dash_rejected(self):
        """assign with a dash is not a valid identifier."""
        agent_text = """---
name: t
model: openai:gpt-4o-mini
attachments:
  - path: foo.md
    assign: my-var
---

body
"""
        with pytest.raises(ValueError, match="valid Python identifier"):
            parse_agent(agent_text)

    def test_duplicate_assign_rejected(self):
        """Two specs with the same assign value raise at parse time."""
        agent_text = """---
name: t
model: openai:gpt-4o-mini
attachments:
  - path: a.md
    assign: shared
  - path: b.md
    assign: shared
---

body
"""
        with pytest.raises(ValueError, match="duplicate.*assign.*shared"):
            parse_agent(agent_text)

    def test_attach_false_without_assign_rejected(self):
        """attach: false with no assign: is meaningless."""
        agent_text = """---
name: t
model: openai:gpt-4o-mini
attachments:
  - path: foo.md
    attach: false
---

body
"""
        with pytest.raises(ValueError, match="attach.*false.*requires.*assign"):
            parse_agent(agent_text)

    def test_path_with_leading_dash_rejected(self):
        """Leading '-' on path is reserved for legacy string-form removal syntax."""
        agent_text = """---
name: t
model: openai:gpt-4o-mini
attachments:
  - path: "-foo.md"
    assign: x
---

body
"""
        with pytest.raises(ValueError, match="cannot start with '-'"):
            parse_agent(agent_text)

    def test_unknown_field_rejected(self):
        """extra='forbid' rejects unknown fields in the spec dict."""
        agent_text = """---
name: t
model: openai:gpt-4o-mini
attachments:
  - path: foo.md
    typo_field: oops
---

body
"""
        with pytest.raises(ValueError):
            parse_agent(agent_text)

    def test_default_mode_is_full(self):
        """mode defaults to 'full'."""
        agent_text = """---
name: t
model: openai:gpt-4o-mini
attachments:
  - path: foo.md
---

body
"""
        agent = parse_agent(agent_text)
        spec = agent.config.attachments[0]
        assert isinstance(spec, AttachmentSpec)
        assert spec.mode == "full"
        assert spec.attach is True
        assert spec.index_format == "first_heading"
        assert spec.max_entries == 50


class TestSpecResolution:
    """Test resolve_agent_config_attachments with the new union signature."""

    def test_dict_form_full_mode_matches_string_form(self, tmp_path):
        f = tmp_path / "foo.md"
        f.write_text("hello")

        atts_str, bindings_str = resolve_agent_config_attachments([str(f)])
        atts_spec, bindings_spec = resolve_agent_config_attachments([AttachmentSpec(path=str(f))])

        assert len(atts_str) == 1
        assert len(atts_spec) == 1
        assert atts_str[0].name == atts_spec[0].name
        assert atts_str[0].content == atts_spec[0].content == "hello"
        assert bindings_str == {}
        assert bindings_spec == {}

    def test_assign_single_file_binds_string(self, tmp_path):
        f = tmp_path / "memory.md"
        f.write_text("active task: foo")

        atts, bindings = resolve_agent_config_attachments([AttachmentSpec(path=str(f), assign="memory_content")])

        assert len(atts) == 1
        assert bindings == {"memory_content": "active task: foo"}

    def test_attach_false_with_assign_binds_without_injecting(self, tmp_path):
        f = tmp_path / "memory.md"
        f.write_text("invisible content")

        atts, bindings = resolve_agent_config_attachments(
            [AttachmentSpec(path=str(f), assign="memory_content", attach=False)]
        )

        assert atts == []
        assert bindings == {"memory_content": "invisible content"}

    def test_assign_missing_file_binds_none(self, tmp_path):
        missing = tmp_path / "does_not_exist.md"

        atts, bindings = resolve_agent_config_attachments([AttachmentSpec(path=str(missing), assign="memory")])

        assert atts == []
        assert bindings == {"memory": None}

    def test_glob_produces_alpha_sorted_attachments(self, tmp_path):
        (tmp_path / "c.md").write_text("c")
        (tmp_path / "a.md").write_text("a")
        (tmp_path / "b.md").write_text("b")

        atts, bindings = resolve_agent_config_attachments([AttachmentSpec(path=str(tmp_path / "*.md"))])

        assert len(atts) == 3
        assert [a.content for a in atts] == ["a", "b", "c"]
        assert bindings == {}

    def test_glob_with_assign_binds_list_of_dicts(self, tmp_path):
        (tmp_path / "a.md").write_text("alpha")
        (tmp_path / "b.md").write_text("beta")

        atts, bindings = resolve_agent_config_attachments([AttachmentSpec(path=str(tmp_path / "*.md"), assign="files")])

        assert len(atts) == 2
        assert "files" in bindings
        files = bindings["files"]
        assert isinstance(files, list)
        assert len(files) == 2
        assert {f["path"] for f in files} == {str(tmp_path / "a.md"), str(tmp_path / "b.md")}
        contents = sorted(f["content"] for f in files)
        assert contents == ["alpha", "beta"]

    def test_glob_with_assign_skips_binaries(self, tmp_path):
        (tmp_path / "a.md").write_text("text")
        (tmp_path / "b.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

        atts, bindings = resolve_agent_config_attachments([AttachmentSpec(path=str(tmp_path / "*"), assign="files")])

        files = bindings["files"]
        # Binary file is skipped from binding list
        assert len(files) == 1
        assert files[0]["content"] == "text"

    def test_empty_glob_binds_empty_list(self, tmp_path):
        atts, bindings = resolve_agent_config_attachments(
            [AttachmentSpec(path=str(tmp_path / "*.nonexistent"), assign="files")]
        )

        assert atts == []
        assert bindings == {"files": []}

    def test_legacy_string_returns_tuple_with_empty_bindings(self, tmp_path):

        f = tmp_path / "foo.md"
        f.write_text("legacy")

        atts, bindings = resolve_agent_config_attachments([str(f)])

        assert len(atts) == 1
        assert bindings == {}


class TestAttachmentBindingPipeline:
    """Test that assign: variables are available in agent body and instructions templates."""

    def _agent(self, body: str, attachments: list, instructions: str = "") -> Agent:
        config = AgentConfig(name="t", attachments=attachments, instructions=instructions)
        return Agent(config=config, content=body, file_path=Path("test.md"))

    def test_assign_var_available_in_agent_body(self, tmp_path):
        memory = tmp_path / "memory.md"
        memory.write_text("active task: water plants")

        agent = self._agent(
            body="Memory dump: {{ memory_content }}",
            attachments=[AttachmentSpec(path=str(memory), assign="memory_content")],
        )

        result = AgentPreparer().prepare(agent, prompt="go")
        assert "active task: water plants" in result.rendered_prompt

    def test_assign_var_available_in_instructions(self, tmp_path):
        memory = tmp_path / "memory.md"
        memory.write_text("Vikunja note")

        agent = self._agent(
            body="hello",
            attachments=[AttachmentSpec(path=str(memory), assign="memory_content")],
            instructions="{% if 'Vikunja' in memory_content %}Use vikunja skill.{% endif %}",
        )

        result = AgentPreparer().prepare(agent, prompt="go")
        assert "Use vikunja skill." in result.combined_instructions

    def test_attach_false_binding_works_without_attachment(self, tmp_path):
        memory = tmp_path / "memory.md"
        memory.write_text("invisible body")

        agent = self._agent(
            body="Memory: {{ memory_content }}",
            attachments=[AttachmentSpec(path=str(memory), assign="memory_content", attach=False)],
        )

        result = AgentPreparer().prepare(agent, prompt="go")
        assert "Memory: invisible body" in result.rendered_prompt
        assert all(att.name != "memory.md" for att in result.attachments)

    def test_collision_with_builtin_warns_and_assign_wins(self, tmp_path, caplog):
        import logging

        f = tmp_path / "today.md"
        f.write_text("FROZEN-DATE")

        agent = self._agent(
            body="value: {{ user_prompt }}",  # not the colliding one
            attachments=[AttachmentSpec(path=str(f), assign="user_prompt")],
        )

        with caplog.at_level(logging.WARNING, logger="tsugite.agent_preparation"):
            result = AgentPreparer().prepare(agent, prompt="actual_prompt_value")

        assert "FROZEN-DATE" in result.rendered_prompt
        assert any("user_prompt" in rec.message for rec in caplog.records)

    def test_collision_with_prefetch_warns_and_assign_wins(self, tmp_path, caplog, monkeypatch):
        import logging

        f = tmp_path / "shadow.md"
        f.write_text("from-attachment")

        config = AgentConfig(
            name="t",
            attachments=[AttachmentSpec(path=str(f), assign="shared_var")],
            prefetch=[{"tool": "noop", "assign": "shared_var"}],
        )
        agent = Agent(config=config, content="value: {{ shared_var }}", file_path=Path("x.md"))

        # Stub execute_prefetch to return the prefetched value
        def fake_execute_prefetch(_):
            return {"shared_var": "from-prefetch"}

        from tsugite import agent_runner

        monkeypatch.setattr(agent_runner, "execute_prefetch", fake_execute_prefetch)

        with caplog.at_level(logging.WARNING, logger="tsugite.agent_preparation"):
            result = AgentPreparer().prepare(agent, prompt="go")

        assert "from-attachment" in result.rendered_prompt
        assert any("shared_var" in rec.message for rec in caplog.records)


class TestValidationMockBindings:
    """tsu validate must accept agents that use assign: variables in their templates."""

    def test_validate_agent_with_assign_var_in_body(self, tmp_path):
        agent_file = tmp_path / "validate_assign.md"
        agent_file.write_text("""---
name: validate_assign
model: openai:gpt-4o-mini
attachments:
  - path: memory.md
    assign: memory_content
---

Memory: {{ memory_content }}
""")
        agent = parse_agent_file(agent_file)
        is_valid, msg = validate_agent_execution(agent)
        assert is_valid, f"Validation failed: {msg}"

    def test_validate_agent_with_glob_assign(self, tmp_path):
        agent_file = tmp_path / "validate_glob.md"
        agent_file.write_text("""---
name: validate_glob
model: openai:gpt-4o-mini
attachments:
  - path: "*.md"
    assign: files
---

{% for f in files %}- {{ f.path }}
{% endfor %}
""")
        agent = parse_agent_file(agent_file)
        is_valid, msg = validate_agent_execution(agent)
        assert is_valid, f"Validation failed: {msg}"

    def test_validate_agent_with_assign_var_in_instructions(self, tmp_path):
        agent_file = tmp_path / "validate_inst.md"
        agent_file.write_text("""---
name: validate_inst
model: openai:gpt-4o-mini
instructions: "memory says: {{ memory_content }}"
attachments:
  - path: memory.md
    assign: memory_content
---

body
""")
        agent = parse_agent_file(agent_file)
        is_valid, msg = validate_agent_execution(agent)
        assert is_valid, f"Validation failed: {msg}"


class TestIndexResolution:
    """Test mode: index resolution behavior."""

    def test_index_path_only_alpha_sorted(self, tmp_path):
        (tmp_path / "c.md").write_text("# C heading\n\nbody c")
        (tmp_path / "a.md").write_text("# A heading\n\nbody a")
        (tmp_path / "b.md").write_text("# B heading\n\nbody b")

        atts, bindings = resolve_agent_config_attachments(
            [AttachmentSpec(path=str(tmp_path / "*.md"), mode="index", index_format="path_only")]
        )

        assert len(atts) == 1
        idx = atts[0]
        assert idx.content_type == AttachmentContentType.TEXT
        assert idx.mode == "index"
        # Bullets in alpha order
        positions = [idx.content.find(name) for name in ["a.md", "b.md", "c.md"]]
        assert positions == sorted(positions)

    def test_index_first_heading_format(self, tmp_path):
        (tmp_path / "x.md").write_text("# Heading X\n\nbody")
        (tmp_path / "y.md").write_text("not-heading\n# Real Heading Y\n")

        atts, _ = resolve_agent_config_attachments(
            [AttachmentSpec(path=str(tmp_path / "*.md"), mode="index", index_format="first_heading")]
        )

        idx_content = atts[0].content
        assert "Heading X" in idx_content
        assert "Real Heading Y" in idx_content

    def test_index_first_line_format(self, tmp_path):
        (tmp_path / "p.md").write_text("just one line\n\n# heading later")
        (tmp_path / "q.md").write_text("first line of q")

        atts, _ = resolve_agent_config_attachments(
            [AttachmentSpec(path=str(tmp_path / "*.md"), mode="index", index_format="first_line")]
        )

        idx_content = atts[0].content
        assert "just one line" in idx_content
        assert "first line of q" in idx_content

    def test_index_frontmatter_format(self, tmp_path):
        (tmp_path / "fm.md").write_text("""---
title: Topic Title
description: A description
---

# Body heading
content
""")

        atts, _ = resolve_agent_config_attachments(
            [AttachmentSpec(path=str(tmp_path / "*.md"), mode="index", index_format="frontmatter")]
        )

        idx_content = atts[0].content
        assert "Topic Title" in idx_content

    def test_index_frontmatter_malformed_falls_back(self, tmp_path):
        (tmp_path / "broken.md").write_text("""---
title: ok
broken: yaml: nested: bad
---

body
""")

        atts, _ = resolve_agent_config_attachments(
            [AttachmentSpec(path=str(tmp_path / "*.md"), mode="index", index_format="frontmatter")]
        )

        # Falls back to path_only for malformed entry, but still emits the index
        assert len(atts) == 1
        assert "broken.md" in atts[0].content

    def test_index_max_entries_truncates(self, tmp_path, caplog):
        import logging

        for i in range(8):
            (tmp_path / f"f{i}.md").write_text(f"# heading {i}\n")

        with caplog.at_level(logging.WARNING, logger="tsugite.agent_preparation"):
            atts, _ = resolve_agent_config_attachments(
                [AttachmentSpec(path=str(tmp_path / "*.md"), mode="index", max_entries=3)]
            )

        # Only 3 file references in the index
        idx_content = atts[0].content
        included = [n for n in [f"f{i}.md" for i in range(8)] if n in idx_content]
        assert len(included) == 3
        assert any("max_entries" in rec.message or "truncat" in rec.message.lower() for rec in caplog.records)

    def test_index_empty_glob_omits_attachment(self, tmp_path):
        atts, bindings = resolve_agent_config_attachments(
            [AttachmentSpec(path=str(tmp_path / "*.nonexistent"), mode="index", assign="files")]
        )

        assert atts == []
        assert bindings == {"files": []}

    def test_index_single_concrete_file(self, tmp_path):
        f = tmp_path / "single.md"
        f.write_text("# Just one\n")

        atts, _ = resolve_agent_config_attachments(
            [AttachmentSpec(path=str(f), mode="index", index_format="first_heading")]
        )

        assert len(atts) == 1
        assert "single.md" in atts[0].content
        assert "Just one" in atts[0].content

    def test_index_with_assign_binds_list_of_dicts(self, tmp_path):
        (tmp_path / "a.md").write_text("# Heading A\n")
        (tmp_path / "b.md").write_text("# Heading B\n")

        atts, bindings = resolve_agent_config_attachments(
            [
                AttachmentSpec(
                    path=str(tmp_path / "*.md"),
                    mode="index",
                    assign="topics",
                    index_format="first_heading",
                )
            ]
        )

        assert "topics" in bindings
        topics = bindings["topics"]
        assert isinstance(topics, list)
        assert len(topics) == 2
        # Required keys per the variable shape table
        for entry in topics:
            assert "path" in entry
            assert "heading" in entry
            assert "size_bytes" in entry
            assert "mtime" in entry
        headings = sorted(t["heading"] for t in topics)
        assert headings == ["Heading A", "Heading B"]

    def test_index_attach_false_skips_attachment_keeps_binding(self, tmp_path):
        (tmp_path / "a.md").write_text("# A\n")

        atts, bindings = resolve_agent_config_attachments(
            [
                AttachmentSpec(
                    path=str(tmp_path / "*.md"),
                    mode="index",
                    assign="topics",
                    attach=False,
                )
            ]
        )

        assert atts == []
        assert "topics" in bindings
        assert len(bindings["topics"]) == 1

    def test_index_default_name_derived_from_glob(self, tmp_path):
        (tmp_path / "topics").mkdir()
        (tmp_path / "topics" / "a.md").write_text("# A\n")

        atts, _ = resolve_agent_config_attachments(
            [AttachmentSpec(path=str(tmp_path / "topics" / "*.md"), mode="index")]
        )

        # Name derives from the glob's parent directory
        assert "topics" in atts[0].name

    def test_index_explicit_name_overrides_default(self, tmp_path):
        (tmp_path / "a.md").write_text("# A\n")

        atts, _ = resolve_agent_config_attachments(
            [AttachmentSpec(path=str(tmp_path / "*.md"), mode="index", name="my_custom_index")]
        )

        assert atts[0].name == "my_custom_index"
