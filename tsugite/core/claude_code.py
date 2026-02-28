"""Claude Code CLI subprocess provider.

Routes LLM calls through `claude --print` instead of litellm.acompletion(),
enabling Claude Max subscription auth. Text-only, no multimodal support.
"""

import asyncio
import json
import os
import shutil
import tempfile
import uuid
from collections.abc import AsyncIterator

# Env vars that must be unset to avoid "nested session" detection
_CLAUDE_ENV_VARS = {"CLAUDECODE", "CLAUDE_CODE_ENTRYPOINT", "ANTHROPIC_API_KEY"}


class ClaudeCodeProcess:
    """Manages a persistent claude CLI subprocess for LLM completions.

    Uses stream-json I/O format to send user turns via stdin and parse
    streaming responses from stdout. The subprocess holds conversation
    state in memory between turns.
    """

    def __init__(self):
        self._process: asyncio.subprocess.Process | None = None
        self._session_id: str | None = None
        self._system_prompt_file: str | None = None
        self._stderr_lines: list[str] = []
        self._stderr_task: asyncio.Task | None = None

    @property
    def session_id(self) -> str | None:
        return self._session_id

    async def _drain_stderr(self) -> None:
        """Background task: read stderr lines so the pipe never fills up."""
        try:
            while True:
                line = await self._process.stderr.readline()
                if not line:
                    break
                self._stderr_lines.append(line.decode().rstrip())
        except (asyncio.CancelledError, Exception):
            pass

    def _get_stderr(self) -> str:
        return "\n".join(self._stderr_lines[-20:])  # last 20 lines

    async def start(self, model: str, system_prompt: str, resume_session: str | None = None) -> None:
        """Launch persistent claude subprocess.

        Args:
            model: Model name (sonnet, opus, haiku, or full model ID)
            system_prompt: System prompt text
            resume_session: Optional session ID to resume

        Raises:
            RuntimeError: If claude CLI is not found or fails to start
        """
        if not shutil.which("claude"):
            raise RuntimeError(
                "Claude Code CLI not found. Install it with: npm install -g @anthropic-ai/claude-code"
            )

        # Write system prompt to temp file
        fd, self._system_prompt_file = tempfile.mkstemp(suffix=".txt", prefix="tsugite_sysprompt_")
        with os.fdopen(fd, "w") as f:
            f.write(system_prompt)

        cmd = [
            "claude", "--print",
            "--input-format", "stream-json",
            "--output-format", "stream-json",
            "--verbose",
            "--max-turns", "1",
            "--model", model,
            "--tools", "",
            "--strict-mcp-config",
            "--system-prompt-file", self._system_prompt_file,
        ]

        if resume_session:
            cmd.extend(["--resume", resume_session])
        else:
            cmd.extend(["--session-id", str(uuid.uuid4())])

        # Copy env but unset keys that trigger nested-session guard or API key usage
        env = {k: v for k, v in os.environ.items() if k not in _CLAUDE_ENV_VARS}

        self._process = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )

        # Start draining stderr in background to prevent pipe buffer deadlock
        self._stderr_task = asyncio.create_task(self._drain_stderr())

    async def send_message(self, content: str) -> AsyncIterator[dict]:
        """Write user message to stdin and yield streaming events from stdout.

        Args:
            content: User message text

        Yields:
            Dicts with type "text_delta" (streaming chunk) or "result" (final)

        Raises:
            RuntimeError: If subprocess has crashed
        """
        msg = {
            "type": "user",
            "message": {"role": "user", "content": content},
            "session_id": self._session_id or "default",
        }
        self._process.stdin.write((json.dumps(msg) + "\n").encode())
        await self._process.stdin.drain()

        while True:
            line = await self._process.stdout.readline()
            if not line:
                stderr = self._get_stderr()
                raise RuntimeError(f"Claude Code process ended unexpectedly. stderr: {stderr}")

            raw = line.decode().strip()
            if not raw:
                continue

            try:
                event = json.loads(raw)
            except json.JSONDecodeError:
                continue

            event_type = event.get("type", "")

            # Init event — capture session_id (arrives after first user message)
            if event_type == "system" and event.get("subtype") == "init":
                self._session_id = event.get("session_id")
                continue

            # Content block delta (streaming with --include-partial-messages)
            if event_type == "content_block_delta":
                delta = event.get("delta", {})
                if delta.get("type") == "text_delta":
                    yield {"type": "text_delta", "text": delta["text"]}

            # Full assistant message — extract text from content blocks
            elif event_type == "assistant":
                message = event.get("message", {})
                content_blocks = message.get("content", [])
                text = "".join(
                    block.get("text", "") for block in content_blocks if block.get("type") == "text"
                )
                if text:
                    yield {"type": "text_delta", "text": text}

            # Final result
            elif event_type == "result":
                yield {
                    "type": "result",
                    "text": event.get("result", ""),
                    "cost_usd": event.get("total_cost_usd"),
                    "duration_ms": event.get("duration_ms"),
                    "session_id": event.get("session_id", self._session_id),
                }
                return

            # Skip: rate_limit_event, system, etc.

    async def stop(self) -> None:
        """Terminate subprocess and clean up temp files."""
        if self._stderr_task:
            self._stderr_task.cancel()
            self._stderr_task = None

        if self._process:
            try:
                self._process.terminate()
                await self._process.wait()
            except ProcessLookupError:
                pass
            self._process = None

        if self._system_prompt_file and os.path.exists(self._system_prompt_file):
            os.unlink(self._system_prompt_file)
            self._system_prompt_file = None
