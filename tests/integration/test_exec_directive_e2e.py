"""End-to-end tests for `<!-- tsu:exec -->` directive.

These hit a real LLM API. Skipped automatically when no API key is set.
"""

import json

from tests.integration.conftest import run_integration_agent


def test_exec_directive_dispatcher_pattern(agent_file, work_dir):
    """Agent uses tsu:exec to inventory inbox files and the LLM sees the result."""
    inbox = work_dir / "inbox"
    inbox.mkdir()
    (inbox / "alpha.json").write_text(json.dumps({"source": "github"}))
    (inbox / "beta.json").write_text(json.dumps({"source": "slack"}))

    body = """\
<!-- tsu:exec name="dispatch" assign="targets" -->
import os, json
out = []
for p in sorted(os.listdir("inbox")):
    if p.endswith(".json"):
        out.append({"path": p, "source": json.loads(open(f"inbox/{p}").read())["source"]})
out
<!-- /tsu:exec -->

The dispatch step found these targets: {{ targets }}.

Echo the source field of each target back via return_value as a comma-separated list.
"""
    agent = agent_file(name="dispatcher", body=body)
    result = run_integration_agent(agent_path=agent, prompt="enumerate the inbox")

    lower = result.lower()
    assert "github" in lower
    assert "slack" in lower
