"""Tests for JSON Schema generation and validation."""

import json

import pytest
from pydantic import ValidationError

from tsugite.md_agents import parse_agent
from tsugite.schemas import generate_agent_schema


def test_generate_schema_has_all_fields():
    """Test that generated schema includes all AgentConfig fields."""
    schema = generate_agent_schema()

    # Check required metadata
    assert schema["$schema"] == "http://json-schema.org/draft-07/schema#"
    assert "$id" in schema
    assert schema["title"] == "Tsugite Agent Configuration"
    assert "description" in schema

    # Check all expected properties exist
    expected_fields = [
        "name",
        "description",
        "model",
        "max_turns",
        "tools",
        "prefetch",
        "attachments",
        "permissions_profile",
        "context_budget",
        "instructions",
        "mcp_servers",
        "extends",
        "reasoning_effort",
        "custom_tools",
        "initial_tasks",
        "disable_history",
        "auto_context",
        "visibility",
        "spawnable",
    ]

    properties = schema.get("properties", {})
    for field in expected_fields:
        assert field in properties, f"Field '{field}' missing from schema"

    # Check required fields
    assert schema.get("required") == ["name"]


def test_schema_enum_values():
    """Test that enum fields have correct values."""
    schema = generate_agent_schema()
    properties = schema["properties"]

    # visibility should have enum
    assert "visibility" in properties
    assert properties["visibility"]["enum"] == ["public", "private", "internal"]

    # reasoning_effort should have enum
    assert "reasoning_effort" in properties
    assert properties["reasoning_effort"]["enum"] == ["low", "medium", "high"]


def test_schema_has_examples():
    """Test that key fields have examples."""
    schema = generate_agent_schema()
    properties = schema["properties"]

    # Check that examples are provided for key fields
    assert "examples" in properties["name"]
    assert "examples" in properties["model"]
    assert "examples" in properties["tools"]
    assert "examples" in properties["extends"]


def test_valid_agent_frontmatter(tmp_path):
    """Test that valid agent frontmatter parses successfully."""
    agent_file = tmp_path / "test_agent.md"
    agent_file.write_text(
        """---
name: test_agent
model: ollama:qwen2.5-coder:7b
max_turns: 10
tools: [read_file, write_file]
visibility: public
---

Test agent prompt
"""
    )

    # Should parse without errors
    agent = parse_agent(agent_file.read_text(), agent_file)
    assert agent.config.name == "test_agent"
    assert agent.config.model == "ollama:qwen2.5-coder:7b"
    assert agent.config.max_turns == 10
    assert agent.config.tools == ["read_file", "write_file"]


def test_invalid_field_name(tmp_path):
    """Test that invalid field names are caught."""
    agent_file = tmp_path / "test_agent.md"
    agent_file.write_text(
        """---
name: test_agent
invalid_field: "this should fail"
---

Test agent
"""
    )

    # Should raise ValidationError
    with pytest.raises(ValidationError) as exc_info:
        parse_agent(agent_file.read_text(), agent_file)

    # Check error message mentions the invalid field
    assert "invalid_field" in str(exc_info.value)
    assert "Extra inputs are not permitted" in str(exc_info.value)


def test_invalid_visibility_value(tmp_path):
    """Test that invalid visibility values are caught."""
    agent_file = tmp_path / "test_agent.md"
    agent_file.write_text(
        """---
name: test_agent
visibility: invalid_value
---

Test agent
"""
    )

    # Should raise ValidationError
    with pytest.raises(ValidationError) as exc_info:
        parse_agent(agent_file.read_text(), agent_file)

    # Check error message mentions visibility
    assert "visibility" in str(exc_info.value)


def test_invalid_field_type(tmp_path):
    """Test that invalid field types are caught."""
    agent_file = tmp_path / "test_agent.md"
    agent_file.write_text(
        """---
name: test_agent
max_turns: "not_an_integer"
---

Test agent
"""
    )

    # Should raise ValidationError
    with pytest.raises(ValidationError) as exc_info:
        parse_agent(agent_file.read_text(), agent_file)

    # Check error message mentions max_turns
    assert "max_turns" in str(exc_info.value)


def test_missing_required_field(tmp_path):
    """Test that missing required fields are caught."""
    agent_file = tmp_path / "test_agent.md"
    agent_file.write_text(
        """---
model: ollama:qwen2.5-coder:7b
---

Test agent without name
"""
    )

    # Should raise ValidationError
    with pytest.raises(ValidationError) as exc_info:
        parse_agent(agent_file.read_text(), agent_file)

    # Check error message mentions name field
    assert "name" in str(exc_info.value)
    assert "required" in str(exc_info.value).lower()


def test_schema_json_serializable():
    """Test that schema can be serialized to JSON."""
    schema = generate_agent_schema()

    # Should be JSON serializable
    json_str = json.dumps(schema, indent=2)
    assert json_str

    # Should be deserializable
    parsed = json.loads(json_str)
    assert parsed == schema


def test_schema_additional_properties_false():
    """Test that schema rejects additional properties."""
    schema = generate_agent_schema()

    # Check that additionalProperties is false (strict validation)
    assert schema.get("additionalProperties") is False


def test_schema_default_values():
    """Test that schema includes correct default values."""
    schema = generate_agent_schema()
    properties = schema["properties"]

    # Check some default values
    assert properties["description"]["default"] == ""
    assert properties["max_turns"]["default"] == 5
    assert properties["disable_history"]["default"] is False
    assert properties["spawnable"]["default"] is True
    assert properties["visibility"]["default"] == "public"
