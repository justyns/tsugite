#!/usr/bin/env python3
"""Regenerate the JSON Schema for agent frontmatter.

This script regenerates tsugite/schemas/agent.schema.json from the
AgentConfig Pydantic model. Run this whenever you modify AgentConfig
to keep the schema in sync.

Usage:
    uv run python scripts/regenerate_schema.py
"""

from pathlib import Path

# Add project root to path
import sys

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tsugite.schemas import generate_agent_schema, save_schema

if __name__ == "__main__":
    schema_path = project_root / "tsugite" / "schemas" / "agent.schema.json"

    print(f"Regenerating schema at {schema_path}...")
    save_schema(schema_path)

    schema = generate_agent_schema()
    field_count = len(schema.get("properties", {}))

    print(f"✓ Schema regenerated successfully!")
    print(f"  - {field_count} properties")
    print(f"  - Required: {schema.get('required', [])}")
    print(f"  - Location: {schema_path}")
