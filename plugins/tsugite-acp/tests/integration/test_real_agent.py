"""Live smoke test against a real claude-agent-acp subprocess.

Skipped unless TSUGITE_ACP_INTEGRATION=1 is set in the environment, since it
requires Node + an Anthropic credential. Mirrors the convention used in
tsugite/tests/integration/.
"""

from __future__ import annotations

import os
import shutil

import pytest

if os.environ.get("TSUGITE_ACP_INTEGRATION") != "1":
    pytest.skip("set TSUGITE_ACP_INTEGRATION=1 to run", allow_module_level=True)

if shutil.which("npx") is None:
    pytest.skip("npx not on PATH", allow_module_level=True)


@pytest.mark.asyncio
async def test_one_turn_against_real_agent():
    from tsugite_acp.provider import ACPProvider

    p = ACPProvider()
    try:
        resp = await p.acompletion(
            messages=[{"role": "user", "content": "Reply with just the word 'pong' and nothing else."}],
            model="haiku",
            stream=False,
        )
        assert resp.content
        assert "pong" in resp.content.lower()
    finally:
        await p.stop()
