"""Reconstruction must truncate code-execution output the same way the live turn did,
so the replayed observation is byte-stable and doesn't re-inflate context on continuation."""

from tsugite.core.executor import MAX_EXECUTION_OUTPUT_KB
from tsugite.history.reconstruction import _execution_xml


def test_execution_xml_truncates_large_output_like_live():
    max_bytes = MAX_EXECUTION_OUTPUT_KB * 1024
    big = "A" * (max_bytes + 10000)  # well over the limit
    xml = _execution_xml({"output": big, "duration_ms": 5})

    assert 'truncated="true"' in xml
    # Exactly the first max_bytes of output survive; nothing beyond.
    assert ("A" * max_bytes) in xml
    assert ("A" * (max_bytes + 1)) not in xml


def test_execution_xml_small_output_not_truncated():
    xml = _execution_xml({"output": "small output", "duration_ms": 1})
    assert 'truncated="true"' not in xml
    assert "small output" in xml
