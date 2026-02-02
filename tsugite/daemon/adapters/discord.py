"""Discord bot adapter."""

import asyncio
import re
from types import SimpleNamespace
from typing import NamedTuple, Optional

import discord
from discord.ext import commands

from tsugite.daemon.adapters.base import BaseAdapter, ChannelContext
from tsugite.daemon.config import AgentConfig, DiscordBotConfig
from tsugite.daemon.session import SessionManager
from tsugite.events import FinalAnswerEvent, ReasoningContentEvent, ToolCallEvent
from tsugite.events.base import BaseEvent


class ProgressStep(NamedTuple):
    """Progress update step with tool info."""

    tool_name: str
    completed: bool
    emoji: str


def _handle_task_exception(task: asyncio.Task, context: str = "") -> None:
    """Handle exceptions from background tasks to prevent crashes."""
    try:
        task.result()
    except asyncio.CancelledError:
        pass
    except Exception as e:
        import traceback

        prefix = f"[{context}] " if context else ""
        print(f"{prefix}Background task error: {e}")
        traceback.print_exc()


def _handle_future_exception(future: asyncio.Future, context: str = "") -> None:
    """Handle exceptions from futures (used with run_coroutine_threadsafe)."""
    try:
        future.result()
    except asyncio.CancelledError:
        pass
    except Exception as e:
        import traceback

        prefix = f"[{context}] " if context else ""
        print(f"{prefix}Future error: {e}")
        traceback.print_exc()


class DiscordProgressHandler:
    """Lightweight handler for Discord progress updates."""

    MAX_DISPLAY_STEPS = 10  # Show last N steps if too many

    def __init__(self, channel: discord.abc.Messageable, loop: asyncio.AbstractEventLoop):
        """Initialize progress handler.

        Args:
            channel: Discord channel to send progress updates to
            loop: Discord bot's event loop (for thread-safe scheduling)
        """
        self.channel = channel
        self.loop = loop
        self.progress_msg: Optional[discord.Message] = None
        self.updates: list[ProgressStep] = []
        self.update_lock = asyncio.Lock()
        self.typing_task: Optional[asyncio.Task] = None
        self.done = False
        self._thinking_shown = False

    def handle_event(self, event: BaseEvent) -> None:
        """Handle EventBus events and update Discord message.

        This is called from a different thread (executor), so we use
        run_coroutine_threadsafe to schedule work on the Discord event loop.
        """
        future = asyncio.run_coroutine_threadsafe(self._handle_event_async(event), self.loop)
        future.add_done_callback(lambda f: _handle_future_exception(f, "progress"))

    async def _handle_event_async(self, event: BaseEvent) -> None:
        """Async implementation of event handling."""
        async with self.update_lock:
            if isinstance(event, ToolCallEvent):
                if self.updates:
                    prev_step = self.updates[-1]
                    self.updates[-1] = ProgressStep(prev_step.tool_name, True, prev_step.emoji)

                tool_emoji = self._get_tool_emoji(event.tool)
                self.updates.append(ProgressStep(event.tool, False, tool_emoji))
                await self._update_progress()

            elif isinstance(event, ReasoningContentEvent):
                if not self._thinking_shown:
                    self._thinking_shown = True
                    self.updates.append(ProgressStep("Thinking", False, "💭"))
                    await self._update_progress()

            elif isinstance(event, FinalAnswerEvent):
                if self.updates and not self.updates[-1].completed:
                    last_step = self.updates[-1]
                    self.updates[-1] = ProgressStep(last_step.tool_name, True, last_step.emoji)
                await self._collapse_to_summary()

    def _get_tool_emoji(self, tool_name: str) -> str:
        """Get emoji for tool name."""
        emoji_map = {
            "search": "🔍",
            "web": "🌐",
            "read": "📖",
            "write": "✍️",
            "edit": "📝",
            "execute": "⚙️",
            "code": "💻",
            "file": "📄",
        }
        for key, emoji in emoji_map.items():
            if key in tool_name.lower():
                return emoji
        return "🔧"

    async def _update_progress(self):
        """Update or create progress message."""
        if self.done:
            return

        lines = ["🤔 Working..."]

        display_updates = self.updates
        if len(self.updates) > self.MAX_DISPLAY_STEPS:
            skipped = len(self.updates) - self.MAX_DISPLAY_STEPS
            lines.append(f"├─ ... ({skipped} earlier steps)")
            display_updates = self.updates[-self.MAX_DISPLAY_STEPS :]

        for i, step in enumerate(display_updates):
            is_last = i == len(display_updates) - 1
            prefix = "└─" if is_last else "├─"
            suffix = " ✓" if step.completed else ""
            lines.append(f"{prefix} {step.emoji} `{step.tool_name}`{suffix}")

        text = "\n".join(lines)

        if len(text) > 2000:
            text = text[:1950] + "\n... (truncated)"

        try:
            if self.progress_msg is None:
                self.progress_msg = await self.channel.send(text)
            else:
                await self.progress_msg.edit(content=text)
        except discord.errors.HTTPException:
            pass  # Ignore errors (rate limit, deleted message, etc.)

    async def _collapse_to_summary(self):
        """Collapse progress to a summary on completion."""
        self.done = True

        if not self.progress_msg or not self.updates:
            return

        tool_steps = [u for u in self.updates if u.tool_name != "Thinking"]
        tool_counts = {}
        for step in tool_steps:
            tool_counts[step.tool_name] = tool_counts.get(step.tool_name, 0) + 1

        if tool_counts:
            top_tools = sorted(tool_counts.items(), key=lambda x: x[1], reverse=True)[:3]
            tool_summary = ", ".join(f"{name} ×{count}" for name, count in top_tools)
            if len(tool_counts) > 3:
                tool_summary += ", ..."
        else:
            tool_summary = "none"

        summary = f"✅ Done ({len(tool_steps)} steps: {tool_summary})"

        try:
            await self.progress_msg.edit(content=summary)
        except discord.errors.HTTPException:
            pass

    async def start_typing_loop(self):
        """Keep typing indicator active for long operations."""

        async def typing_loop():
            try:
                while not self.done:
                    try:
                        await self.channel.typing()
                    except (AttributeError, discord.errors.HTTPException):
                        pass  # Some channel types may not support typing
                    await asyncio.sleep(8)  # Re-trigger before 10s expiry
            except asyncio.CancelledError:
                pass
            except Exception as e:
                print(f"Typing loop error: {e}")

        self.typing_task = asyncio.create_task(typing_loop())

    async def cleanup(self, success: bool = True):
        """Clean up and finalize status message."""
        self.done = True

        if self.typing_task:
            self.typing_task.cancel()
            try:
                await self.typing_task
            except asyncio.CancelledError:
                pass

        if self.progress_msg and not success:
            try:
                failed_step = self.updates[-1].tool_name if self.updates else "unknown"
                await self.progress_msg.edit(
                    content=f"❌ Error after {len(self.updates)} steps\n└─ `{failed_step}` failed"
                )
            except discord.errors.HTTPException:
                pass
        elif success and self.updates and self.progress_msg:
            await self._collapse_to_summary()


class DiscordAdapter(BaseAdapter):
    """Discord bot adapter tied to a specific agent."""

    def __init__(
        self, bot_config: DiscordBotConfig, agent_name: str, agent_config: AgentConfig, session_manager: SessionManager
    ):
        """Initialize Discord adapter.

        Args:
            bot_config: Discord bot configuration
            agent_name: Name of the agent
            agent_config: Agent configuration
            session_manager: Session manager for this agent
        """
        super().__init__(agent_name, agent_config, session_manager)
        self.bot_config = bot_config
        self.active_progress_handlers: list[DiscordProgressHandler] = []

        intents = discord.Intents.default()
        intents.message_content = True

        self.bot = commands.Bot(command_prefix=bot_config.command_prefix, intents=intents)

        @self.bot.event
        async def on_ready():
            print(f"Discord bot '{bot_config.name}' logged in as {self.bot.user} (agent: {agent_name})")

        @self.bot.event
        async def on_error(event_method, *args, **kwargs):
            import traceback

            print(f"[{bot_config.name}] Discord event error in {event_method}:")
            traceback.print_exc()

        @self.bot.event
        async def on_message(message):
            if message.author == self.bot.user:
                return

            is_dm = isinstance(message.channel, discord.DMChannel)

            # Check allowlist for both DMs and server channels
            if bot_config.dm_policy == "allowlist":
                if str(message.author.id) not in bot_config.allow_from:
                    if is_dm:
                        await message.channel.send("You are not authorized.")
                    return

            if is_dm:
                # DMs don't require prefix
                user_msg = message.content.strip()
            else:
                # Server channels require @mention
                if not self.bot.user.mentioned_in(message):
                    return
                # Remove the mention from the message
                user_msg = message.content.replace(f"<@{self.bot.user.id}>", "").strip()
                user_msg = user_msg.replace(f"<@!{self.bot.user.id}>", "").strip()

            if not user_msg:
                return

            # Log incoming message
            channel_type = "DM" if is_dm else "channel"
            print(
                f"[{bot_config.name}] <- {message.author} ({channel_type}): {user_msg[:100]}{'...' if len(user_msg) > 100 else ''}"
            )

            # Process message in isolated task to prevent crashes from affecting the bot
            task = asyncio.create_task(self._process_message(message, user_msg, bot_config.name))
            task.add_done_callback(lambda t: _handle_task_exception(t, bot_config.name))

    async def _process_message(self, message, user_msg: str, bot_name: str):
        """Process a message in an isolated task."""
        channel_context = ChannelContext(
            source="discord",
            channel_id=str(message.channel.id),
            user_id=str(message.author.id),
            reply_to=f"discord:{message.channel.id}",
            metadata={
                "author_name": str(message.author),
                "guild_id": str(message.guild.id) if message.guild else None,
            },
        )

        progress = DiscordProgressHandler(message.channel, asyncio.get_running_loop())
        custom_logger = SimpleNamespace(ui_handler=progress)

        await progress.start_typing_loop()
        self.active_progress_handlers.append(progress)

        try:
            response = await self.handle_message(
                user_id=str(message.author.id),
                message=user_msg,
                channel_context=channel_context,
                custom_logger=custom_logger,
            )
            await progress.cleanup(success=True)

        except Exception as e:
            import traceback

            await progress.cleanup(success=False)
            response = f"Error processing message: {e}"
            print(f"[{bot_name}] ERROR: {e}")
            traceback.print_exc()
        finally:
            # Remove from active handlers after cleanup
            if progress in self.active_progress_handlers:
                self.active_progress_handlers.remove(progress)

        # Log outgoing response
        print(f"[{bot_name}] -> {message.author}: {response[:100]}{'...' if len(response) > 100 else ''}")
        await self._send_chunked(message.channel, response)

    def _split_respecting_code_blocks(self, text: str, limit: int) -> list[str]:
        """Split text into chunks respecting code block boundaries.

        Args:
            text: Text to split
            limit: Maximum chunk size

        Returns:
            List of chunks
        """
        code_block_pattern = re.compile(r"```(\w*)\n([\s\S]*?)```")
        closing_fence_len = 4  # "\n```"

        chunks: list[str] = []
        current = ""

        def flush_current() -> None:
            nonlocal current
            if current.strip():
                chunks.append(current.rstrip("\n"))
            current = ""

        def add_text_lines(text_content: str) -> None:
            nonlocal current
            for line in text_content.split("\n"):
                if len(current) + len(line) + 1 <= limit:
                    current += line + "\n"
                else:
                    flush_current()
                    current = line + "\n"

        def add_code_block(full_block: str, lang: str, inner: str) -> None:
            nonlocal current

            if len(current) + len(full_block) <= limit:
                current += full_block
                return

            if len(full_block) <= limit:
                flush_current()
                current = full_block
                return

            flush_current()
            header = f"```{lang}\n"
            code_chunk = header

            for line in inner.split("\n"):
                line_with_newline = line + "\n"
                if len(code_chunk) + len(line_with_newline) + closing_fence_len <= limit:
                    code_chunk += line_with_newline
                else:
                    chunks.append(code_chunk.rstrip("\n") + "\n```")
                    code_chunk = header + line_with_newline

            if code_chunk != header:
                current = code_chunk.rstrip("\n") + "\n```"

        last_end = 0
        for match in code_block_pattern.finditer(text):
            if match.start() > last_end:
                add_text_lines(text[last_end : match.start()])

            lang = match.group(1)
            inner = match.group(2).strip("\n")
            full_block = f"```{lang}\n{inner}\n```"
            add_code_block(full_block, lang, inner)
            last_end = match.end()

        if last_end < len(text):
            add_text_lines(text[last_end:])

        flush_current()
        return chunks

    async def _send_chunked(self, channel, text: str):
        """Send message, splitting if longer than 2000 chars while respecting code blocks.

        Args:
            channel: Discord channel to send to
            text: Message text to send
        """
        limit = 2000

        if len(text) <= limit:
            await channel.send(text)
            return

        chunks = self._split_respecting_code_blocks(text, limit)
        for chunk in chunks:
            if chunk.strip():
                await channel.send(chunk)

    async def start(self):
        """Start Discord bot."""
        await self.bot.start(self.bot_config.token)

    async def stop(self):
        """Stop Discord bot and clean up resources."""
        # Clean up any active progress handlers (orphaned typing tasks)
        for progress in self.active_progress_handlers[:]:  # Copy list to avoid modification during iteration
            await progress.cleanup(success=False)
        self.active_progress_handlers.clear()

        await self.bot.close()
