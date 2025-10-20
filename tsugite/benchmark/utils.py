"""Common utilities for benchmark framework."""

import json
import re
from typing import Any, Dict, List, Optional

from .config import TEST_CATEGORIES


def extract_inline_field(content: str, label: str) -> Optional[str]:
    """Extract single-line field values like Prompt or Expected Output.

    Args:
        content: Markdown content to search
        label: Field label to extract (e.g., "Prompt", "Expected Output")

    Returns:
        Extracted field value or None if not found
    """
    pattern_label = re.escape(label)

    # Try quoted format first
    quoted = re.search(rf"\*\*{pattern_label}:\*\*\s*\"([^\"]+)\"", content)
    if quoted:
        return quoted.group(1).strip()

    # Try block format
    block = re.search(rf"\*\*{pattern_label}:\*\*\s*(.+?)(?=\n\*\*|$)", content, re.DOTALL)
    if block:
        return block.group(1).strip()

    return None


def extract_block(content: str, label: str) -> Optional[str]:
    """Extract multi-line blocks introduced by a bold label.

    Args:
        content: Markdown content to search
        label: Block label (e.g., "Expected Behaviors")

    Returns:
        Block content or None if not found
    """
    pattern_label = re.escape(label)
    match = re.search(rf"\*\*{pattern_label}:\*\*\n(.*?)(?=\n\*\*|\Z)", content, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def parse_bullet_list(block: Optional[str]) -> List[str]:
    """Parse a bullet list from a markdown block.

    Args:
        block: Block containing bullet list

    Returns:
        List of bullet items (without leading "- ")
    """
    if not block:
        return []
    items = []
    for line in block.splitlines():
        line = line.strip()
        if line.startswith("- "):
            items.append(line[2:].strip())
    return items


def coerce_value(value: str) -> Any:
    """Convert string value to appropriate Python type.

    Args:
        value: String value to convert

    Returns:
        Converted value (bool, None, number, or string)
    """
    raw = value.strip()
    lowered = raw.lower()

    # Check for boolean
    if lowered in {"true", "false"}:
        return lowered == "true"

    # Check for null
    if lowered == "null":
        return None

    # Try JSON parsing first
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        pass

    # Try numeric conversion
    try:
        if "." in raw:
            return float(raw)
        return int(raw)
    except ValueError:
        # Return as string, removing quotes
        return raw.strip("'\"")


def parse_key_value_block(block: Optional[str]) -> Dict[str, Any]:
    """Parse a key-value block from markdown.

    Args:
        block: Block containing key-value pairs (bullet list format)

    Returns:
        Dictionary of parsed key-value pairs
    """
    if not block:
        return {}

    result: Dict[str, Any] = {}
    for line in block.splitlines():
        line = line.strip()
        if not line.startswith("- ") or ":" not in line:
            continue

        # Split on first colon
        key, raw_value = line[2:].split(":", 1)
        result[key.strip()] = coerce_value(raw_value)

    return result


def derive_behavior_expectations(items: List[str]) -> Dict[str, Any]:
    """Derive behavior expectations from a list of requirement items.

    Args:
        items: List of requirement strings

    Returns:
        Dictionary of expected behaviors
    """
    expectations: Dict[str, Any] = {}
    for item in items:
        lowered = item.lower()
        if "use" in lowered and "tool" in lowered:
            expectations["tool_usage"] = True
        elif "file" in lowered and "create" in lowered:
            expectations["file_created"] = True
    return expectations


def get_test_category(test_id: str) -> str:
    """Extract category from test ID.

    Args:
        test_id: Test identifier

    Returns:
        Category name (e.g., "basic", "tools", "unknown")
    """
    return TEST_CATEGORIES.get_category(test_id)


def extract_prompt_from_markdown(markdown_content: str) -> str:
    """Derive a prompt from markdown content when none is provided explicitly.

    Args:
        markdown_content: Markdown content

    Returns:
        Extracted prompt or default message
    """
    for line in markdown_content.splitlines():
        stripped = line.strip()
        # Skip empty lines and headers
        if not stripped or stripped.startswith("#"):
            continue
        return stripped
    return "Describe the required task in detail."


def normalize_code(code: str) -> str:
    """Normalize code for comparison by removing comments and extra whitespace.

    Args:
        code: Source code

    Returns:
        Normalized code
    """
    lines = []
    for line in code.split("\n"):
        # Remove comments (simplified - handles # and //)
        line = re.sub(r"#.*$", "", line)
        line = re.sub(r"//.*$", "", line)
        line = line.strip()
        if line:
            lines.append(line)

    return "\n".join(lines)


def json_similarity(obj1: Any, obj2: Any) -> float:
    """Calculate similarity between JSON objects recursively.

    Args:
        obj1: First object
        obj2: Second object

    Returns:
        Similarity score from 0.0 to 1.0
    """
    # Type mismatch
    if type(obj1) is not type(obj2):
        return 0.0

    # Dictionary comparison
    if isinstance(obj1, dict):
        if not obj1 and not obj2:
            return 1.0

        all_keys = set(obj1.keys()) | set(obj2.keys())
        if not all_keys:
            return 1.0

        key_scores = []
        for key in all_keys:
            if key in obj1 and key in obj2:
                key_scores.append(json_similarity(obj1[key], obj2[key]))
            else:
                key_scores.append(0.0)

        return sum(key_scores) / len(key_scores)

    # List comparison
    elif isinstance(obj1, list):
        if len(obj1) != len(obj2):
            return 0.5 if obj1 == obj2 else 0.0

        if not obj1:
            return 1.0

        scores = [json_similarity(a, b) for a, b in zip(obj1, obj2)]
        return sum(scores) / len(scores)

    # Primitive comparison
    else:
        return 1.0 if obj1 == obj2 else 0.0
