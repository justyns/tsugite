"""Producer cases for the prompt-XML golden test.

Each case renders one LLM-facing block with awkward input - bodies holding `&`,
`<`, and the block's own closing tag; attribute values holding quotes; optional
fields empty. Adding a producer here pins its exact output.
"""

from __future__ import annotations

# Values chosen to break naive string building.
AMP = "A & B"
ANGLE = "<urgent>"
QUOTED = 'say "hi"'
NASTY = 'x < y & z "q"'


def _execution_results():
    from tsugite.core.executor import ExecutionResult

    yield "exec/plain", ExecutionResult(output="hello", error=None, stdout="", stderr="").to_xml()
    yield "exec/special-chars", ExecutionResult(output=NASTY, error=None, stdout="", stderr="").to_xml()
    yield (
        "exec/duration",
        ExecutionResult(output="ok", error=None, stdout="", stderr="").to_xml(duration_ms=142),
    )
    yield (
        "exec/error-with-traceback",
        ExecutionResult(
            output="", error=f"boom {ANGLE} {AMP}", stdout="", stderr="Traceback\n  line one\n  line two"
        ).to_xml(),
    )
    yield (
        "exec/return-value",
        ExecutionResult(output="", error=None, stdout="", stderr="", return_value={"k": NASTY}).to_xml(),
    )
    yield (
        "exec/vars-and-state",
        ExecutionResult(
            output="o",
            error=None,
            stdout="",
            stderr="",
            variables_set={"a": "1 < 2"},
            state_keys={"s": AMP},
        ).to_xml(),
    )
    yield (
        "exec/truncated",
        ExecutionResult(output="y" * 40, error=None, stdout="", stderr="", truncated_to="/tmp/full.txt").to_xml(
            max_output_kb=0
        ),
    )
    yield "exec/empty", ExecutionResult(output="", error=None, stdout="", stderr="").to_xml()


def _history_blocks():
    from tsugite.history.reconstruction import _delivery_xml, _execution_xml, _format_error_xml

    yield "history/execution", _execution_xml({"output": NASTY, "error": None, "duration_ms": 7})
    yield "history/execution-error", _execution_xml({"output": "", "error": f"bad {AMP}"})
    yield (
        "history/delivery",
        _delivery_xml({"source": QUOTED, "kind": "alert", "title": ANGLE, "message": NASTY}),
    )
    yield "history/delivery-no-title", _delivery_xml({"source": "s", "kind": "k", "message": AMP})
    yield "history/format-error", _format_error_xml({"reason": f"two blocks {ANGLE}"})


def _file_envelope():
    import tempfile
    from pathlib import Path

    from tsugite.tools.fs import _wrap_file_metadata

    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "sample.py"
        body = 'if a < b and c & d: print("<x>")'
        p.write_text(body)
        # mtime-derived attributes are not stable across runs; assert the shape
        # of the envelope by stripping them.
        out = _wrap_file_metadata(body, p, "sample.py", None, None, None)
        head, _, rest = out.partition(">")
        stable = head.split(" modified=")[0] + ">" + rest
        yield "fs/file-envelope", stable


def _skill_resources():
    from pathlib import Path

    from tsugite.tools.skills import _append_resource_block

    yield (
        "skills/resources",
        _append_resource_block("body text", Path("/skills/my skill"), ["scripts/a.py", "references/b & c.md"]),
    )
    yield "skills/no-resources", _append_resource_block("body text", Path("/skills/x"), [])


def _hook_blocks():
    from tsugite.hooks import Block, render_blocks

    yield "hooks/plain", render_blocks([Block(tag="memory", body="User prefers tea.")])
    yield (
        "hooks/attrs-and-body",
        render_blocks([Block(tag="memory", body=NASTY, attributes={"source": QUOTED})]),
    )
    yield (
        "hooks/multiple",
        render_blocks([Block(tag="a", body="1"), Block(tag="b", body="2", attributes={"k": AMP})]),
    )


def _daemon_blocks():
    from tsugite_daemon.adapters.base import _build_client_context_block, _render_session_topic_lines
    from tsugite_daemon.session_runner import build_completion_message
    from tsugite_daemon.session_store import Session, render_pending_deliveries_xml

    yield (
        "daemon/client-context",
        _build_client_context_block(
            [
                {"key": "page", "label": QUOTED, "value": NASTY, "untrusted": True},
                {"key": "note", "label": "Note", "value": AMP},
            ]
        ),
    )
    yield "daemon/client-context-empty", _build_client_context_block([])
    yield "daemon/session-topic", "\n".join(_render_session_topic_lines(f"ship {ANGLE} {AMP}", indent="  "))

    session = Session(id="s-1")
    session.pending_deliveries = [
        {"id": "d1", "source": QUOTED, "title": ANGLE, "timestamp": "2026-01-01T00:00:00+00:00", "message": NASTY},
        {"id": "d2", "source": "cron", "message": AMP},
    ]
    yield "daemon/pending-deliveries", render_pending_deliveries_xml(session)

    finished = Session(id="s-2", title=f"My {QUOTED} {ANGLE}")
    yield "daemon/session-finished", build_completion_message(finished, "completed", f"done {AMP}")
    yield "daemon/session-finished-failed", build_completion_message(finished, "failed", ANGLE)
    yield "daemon/session-finished-no-summary", build_completion_message(finished, "completed", "")


def _agent_notices():
    from tsugite.core.agent import (
        _build_bare_python_notice_xml,
        _build_multi_block_warning_xml,
        _build_spoofed_runtime_tag_warning,
        _build_unexecuted_tool_call_notice_xml,
    )

    yield "agent/multi-block", _build_multi_block_warning_xml(3)
    yield "agent/spoofed-tag", _build_spoofed_runtime_tag_warning()
    yield "agent/unexecuted-tool-call", _build_unexecuted_tool_call_notice_xml()
    yield "agent/bare-python", _build_bare_python_notice_xml()


def all_cases() -> list[tuple[str, str]]:
    cases: list[tuple[str, str]] = []
    for producer in (
        _execution_results,
        _history_blocks,
        _file_envelope,
        _skill_resources,
        _hook_blocks,
        _daemon_blocks,
        _agent_notices,
    ):
        cases.extend(producer())
    return cases
