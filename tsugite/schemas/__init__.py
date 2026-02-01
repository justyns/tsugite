"""JSON Schema generation for Tsugite agent frontmatter."""

import json
from pathlib import Path
from typing import Any, Dict

from ..md_agents import AgentConfig


def generate_agent_schema() -> Dict[str, Any]:
    """Generate JSON Schema for agent frontmatter from Pydantic model.

    Returns:
        Dictionary containing the JSON Schema
    """
    # Generate base schema from Pydantic model
    schema = AgentConfig.model_json_schema()

    # Customize metadata
    schema["$schema"] = "http://json-schema.org/draft-07/schema#"
    schema["$id"] = "https://raw.githubusercontent.com/anthropics/tsugite/main/tsugite/schemas/agent.schema.json"
    schema["title"] = "Tsugite Agent Configuration"
    schema["description"] = (
        "YAML frontmatter configuration for Tsugite markdown agents. "
        "See https://github.com/anthropics/tsugite for full documentation."
    )

    # Add examples and enhance descriptions for key fields
    properties = schema.get("properties", {})

    if "name" in properties:
        properties["name"]["description"] = (
            "Agent identifier. Used for referencing with +name syntax. Example: 'researcher', 'code-reviewer'"
        )
        properties["name"]["examples"] = ["researcher", "code-reviewer", "my_agent"]

    if "model" in properties:
        properties["model"]["description"] = (
            "LLM model in format provider:model[:variant]. "
            "Examples: 'ollama:qwen2.5-coder:7b', 'openai:gpt-4o', 'anthropic:claude-3-5-sonnet'"
        )
        properties["model"]["examples"] = [
            "ollama:qwen2.5-coder:7b",
            "openai:gpt-4o",
            "anthropic:claude-3-5-sonnet",
        ]

    if "visibility" in properties:
        properties["visibility"]["description"] = (
            "Agent visibility level. 'public': can be spawned by any agent, "
            "'private': cannot be spawned unless explicitly allowed, "
            "'internal': implementation detail not meant to be spawned"
        )
        properties["visibility"]["enum"] = ["public", "private", "internal"]

    if "reasoning_effort" in properties:
        properties["reasoning_effort"]["description"] = (
            "Reasoning effort for o1/o3 models. Options: 'low', 'medium', 'high'. "
            "Only applicable to OpenAI reasoning models."
        )
        properties["reasoning_effort"]["enum"] = ["low", "medium", "high"]

    if "tools" in properties:
        properties["tools"]["description"] = (
            "List of tool names, globs (*_search), categories (@fs), or exclusions (-delete_file). "
            "Examples: ['read_file', 'write_file'], ['@fs', 'web_search']"
        )
        properties["tools"]["examples"] = [
            ["read_file", "write_file"],
            ["@fs", "web_search"],
            ["*_search", "-delete_file"],
        ]

    if "extends" in properties:
        properties["extends"]["description"] = (
            "Parent agent to inherit from. Use 'none' to opt out of default inheritance. "
            "Examples: 'default', 'none'"
        )
        properties["extends"]["examples"] = ["default", "none"]

    if "initial_tasks" in properties:
        properties["initial_tasks"]["description"] = (
            "Tasks to pre-populate when agent starts. Can be strings (default to pending/required) "
            "or objects with title, status, and optional fields."
        )
        properties["initial_tasks"]["examples"] = [
            ["Read and analyze code", "Check for security issues"],
            [
                {"title": "Core feature", "status": "pending", "optional": False},
                {"title": "Nice formatting", "optional": True},
            ],
        ]

    return schema


def save_schema(output_path: Path) -> None:
    """Generate and save the JSON Schema to a file.

    Args:
        output_path: Path where the schema JSON file should be saved
    """
    schema = generate_agent_schema()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(schema, indent=2) + "\n")
