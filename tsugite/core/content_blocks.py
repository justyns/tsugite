"""Content block extraction for escape-free file operations.

LLMs can define file content outside code blocks using XML tags,
avoiding escaping issues with triple quotes, backticks, and backslashes.
"""

import hashlib
import os
import re
from typing import Dict, Tuple

# Matches both <content name="x">...</content> and <tsu:content name="x">...</tsu:content>
_CONTENT_BLOCK_RE = re.compile(r"<((?:tsu:)?content)\s+name=\"([^\"]+)\">(.*?)</\1>", re.DOTALL)


def extract_content_blocks(text: str) -> Tuple[str, Dict[str, str]]:
    """Extract <content> and <tsu:content> blocks from LLM response text.

    Returns:
        (cleaned_text, blocks_dict) where cleaned_text has all content blocks
        stripped and blocks_dict maps name -> raw content string.
    """
    blocks: Dict[str, str] = {}

    def _collect(match: re.Match) -> str:
        name = match.group(2)
        content = match.group(3)
        # Strip exactly one leading and one trailing newline
        if content.startswith("\n"):
            content = content[1:]
        if content.endswith("\n"):
            content = content[:-1]
        blocks[name] = content
        return ""

    cleaned = _CONTENT_BLOCK_RE.sub(_collect, text)
    return cleaned, blocks


def write_content_blocks_to_files(blocks: Dict[str, str], tmpdir: str) -> Dict[str, str]:
    """Write content blocks to content-addressed temp files.

    Returns:
        Dict mapping variable_name -> file_path.
    """
    if not blocks:
        return {}

    block_dir = os.path.join(tmpdir, "content_blocks")
    os.makedirs(block_dir, exist_ok=True)

    paths: Dict[str, str] = {}
    for name, content in blocks.items():
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:12]
        filepath = os.path.join(block_dir, f"{content_hash}.txt")
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        paths[name] = filepath

    return paths
