"""Byte-for-byte pin on every LLM-facing XML block.

Migrating a producer to `prompt_xml` must not move a single byte: `<context>`
turns carry `cache_control`, so drift both invalidates prompt caches and changes
what the model reads. A deliberate change updates the fixture in the same commit,
which puts the byte diff in front of the reviewer.

Regenerate with: TSU_REGEN_GOLDEN=1 uv run pytest tests/test_prompt_xml_golden.py
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from tests.prompt_xml_cases import all_cases

GOLDEN = Path(__file__).parent / "fixtures" / "prompt_xml_golden.txt"
MARKER = "=" * 8 + " "
JOIN = "\n\n" + MARKER


def _serialize(cases: list[tuple[str, str]]) -> str:
    return MARKER + JOIN.join(f"{name}\n{text}" for name, text in cases) + "\n"


def _parse(blob: str) -> dict[str, str]:
    # Exactly the one trailing newline _serialize added - a block whose own text
    # ends in a newline must keep it.
    body = blob[:-1] if blob.endswith("\n") else blob
    out: dict[str, str] = {}
    for chunk in body.removeprefix(MARKER).split(JOIN):
        name, _, text = chunk.partition("\n")
        out[name.strip()] = text
    return out


def test_golden_matches():
    cases = all_cases()

    if os.environ.get("TSU_REGEN_GOLDEN"):
        GOLDEN.parent.mkdir(parents=True, exist_ok=True)
        GOLDEN.write_text(_serialize(cases))
        pytest.skip("regenerated golden fixture")

    assert GOLDEN.exists(), "golden fixture missing; regenerate with TSU_REGEN_GOLDEN=1"
    expected = _parse(GOLDEN.read_text())

    assert [n for n, _ in cases] == list(expected), "case set changed; regenerate the golden fixture"
    for name, text in cases:
        assert text == expected[name], f"{name} drifted from the golden fixture"


def test_every_case_is_named_once():
    names = [n for n, _ in all_cases()]
    assert len(names) == len(set(names))
