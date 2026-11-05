"""Environment, skills and guidelines blocks go in the system prompt, not the CLI transcript."""

from unittest.mock import patch

import pytest
from tsugite_claude_code.provider import ClaudeCodeProvider

BLOCKS = "<environment>\nenv\n</environment>\n\n<available_skills>\n- a\n</available_skills>\n\n<guidelines>\nbe brief\n</guidelines>\n"


class FakeProcess:
    instances: list = []

    def __init__(self):
        self.sent = []
        FakeProcess.instances.append(self)

    async def start(self, **kwargs):
        self.start_kwargs = kwargs

    def send_message(self, content):
        self.sent.append(content)
        return self._events()

    async def _events(self):
        yield {"type": "result", "text": "ok", "is_error": False, "session_id": "s"}


@pytest.mark.asyncio
async def test_resumed_turns_keep_stable_blocks_out_of_the_message():
    FakeProcess.instances = []
    with patch("tsugite_claude_code.process.ClaudeCodeProcess", FakeProcess):
        for task in ("first", "second"):
            provider = ClaudeCodeProvider()
            provider.set_context(resume_session="cc-session")
            await provider.acompletion(
                messages=[{"role": "system", "content": "sys"}, {"role": "user", "content": f"{BLOCKS}\n{task}"}],
                model="opus",
            )

    for proc, task in zip(FakeProcess.instances, ("first", "second")):
        assert proc.sent == [task]
        assert proc.start_kwargs["system_prompt"].count("<available_skills>") == 1
        assert proc.start_kwargs["system_prompt"].startswith("sys")
