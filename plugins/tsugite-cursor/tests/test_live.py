"""Live check against a real Cursor account. Skipped unless TSUGITE_LIVE_CURSOR=1.

Needs CURSOR_API_KEY or the cursor-api-key secret.
It uses the real HOME and config, and each run spends a few Cursor requests.

  TSUGITE_LIVE_CURSOR=1 uv run pytest plugins/tsugite-cursor/tests/test_live.py -n 0 --no-cov -s

TSUGITE_LIVE_CURSOR_MODEL picks the model (default composer-2.5). The model must exist on the account.
"""

from __future__ import annotations

import contextlib
import io
import os
import re

import pytest
from tsugite_cursor import create_provider

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.skipif(
        os.environ.get("TSUGITE_LIVE_CURSOR") != "1",
        reason="set TSUGITE_LIVE_CURSOR=1 to run against a real Cursor account",
    ),
]

MODEL = os.environ.get("TSUGITE_LIVE_CURSOR_MODEL", "composer-2.5")
TOKEN = "TSUGITE-LIVE-7731"
SYSTEM = f"""You are a test agent inside a loop that runs your Python code.
Rules, no exceptions:
1. Start every reply with the exact line {TOKEN}
2. To compute anything, write one ```python fenced block that prints the result, then stop. Do not state the result yourself.
3. After you receive an <observation> message, reply with the line FINAL: followed by the printed value, and nothing else after the {TOKEN} line.
"""
TASK = "Compute 6 * 7 with python."
CODE_BLOCK = re.compile(r"```python\n(.*?)```", re.DOTALL)


def execute(code: str) -> str:
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        exec(code, {})
    return out.getvalue().strip()


async def two_turns(provider):
    history = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": TASK}]
    first = await provider.acompletion(history, MODEL)
    assert first.content.lstrip().startswith(TOKEN), f"system prompt did not take effect: {first.content!r}"
    block = CODE_BLOCK.search(first.content)
    assert block, f"no python block in {first.content!r}"
    observation = f"<observation>{execute(block.group(1))}</observation>"
    history += [{"role": "assistant", "content": first.content}, {"role": "user", "content": observation}]
    second = await provider.acompletion(history, MODEL)
    return first, second


async def test_two_turn_session_with_a_python_exec_step():
    provider = create_provider()
    try:
        first, second = await two_turns(provider)
        state = provider.get_state()
    finally:
        await provider.stop()

    assert "FINAL: 42" in second.content, second.content
    for reply in (first, second):
        assert reply.usage and reply.usage.prompt_tokens > 0 and reply.usage.completion_tokens > 0
        assert reply.usage.total_tokens == reply.usage.prompt_tokens + reply.usage.completion_tokens
    assert state["session_id"]


async def test_resume_remembers_the_session():
    provider = create_provider()
    try:
        await two_turns(provider)
        session = provider.get_state()["session_id"]
    finally:
        await provider.stop()

    resumed = create_provider()
    resumed.set_context(resume_session=session)
    try:
        reply = await resumed.acompletion(
            [
                {"role": "system", "content": SYSTEM},
                {
                    "role": "user",
                    "content": "Which number did your code print? Answer with the number only after the first line.",
                },
            ],
            MODEL,
        )
    finally:
        await resumed.stop()
    assert "42" in reply.content, reply.content


async def test_catalog_lists_the_requested_model():
    assert MODEL in await create_provider().list_models()
