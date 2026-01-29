"""Discord bot adapter."""

import asyncio
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


class DiscordProgressHandler:
    """Lightweight handler for Discord progress updates."""

    MAX_DISPLAY_STEPS = 10  # Show last N steps if too many

    def __init__(self, channel: discord.abc.Messageable):
        """Initialize progress handler.

        Args:
            channel: Discord channel to send progress updates to
        """
        self.channel = channel
        self.progress_msg: Optional[discord.Message] = None
        self.updates: list[ProgressStep] = []
        self.update_lock = asyncio.Lock()
        self.typing_task: Optional[asyncio.Task] = None
        self.done = False
        self._thinking_shown = False

    async def handle_event(self, event: BaseEvent) -> None:
        """Handle EventBus events and update Discord message."""
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
                    await self.channel.trigger_typing()
                    await asyncio.sleep(8)  # Re-trigger before 10s expiry
            except asyncio.CancelledError:
                pass

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
        async def on_message(message):
            if message.author == self.bot.user:
                return

            if isinstance(message.channel, discord.DMChannel):
                if bot_config.dm_policy == "allowlist":
                    if str(message.author.id) not in bot_config.allow_from:
                        await message.channel.send("You are not authorized.")
                        return

            if not message.content.startswith(bot_config.command_prefix):
                return

            user_msg = message.content[len(bot_config.command_prefix) :].strip()
            if not user_msg:
                return

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

            progress = DiscordProgressHandler(message.channel)
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
                await progress.cleanup(success=False)
                response = f"Error processing message: {e}"
            finally:
                # Remove from active handlers after cleanup
                if progress in self.active_progress_handlers:
                    self.active_progress_handlers.remove(progress)

            await self._send_chunked(message.channel, response)

    async def _send_chunked(self, channel, text: str):
        """Send message, splitting if longer than 2000 chars.

        Args:
            channel: Discord channel to send to
            text: Message text to send
        """
        if len(text) <= 2000:
            await channel.send(text)
            return

        # Split on newlines to avoid breaking mid-sentence
        chunks = []
        current = ""
        for line in text.split("\n"):
            if len(current) + len(line) + 1 <= 2000:
                current += line + "\n"
            else:
                if current:
                    chunks.append(current)
                current = line + "\n"
        if current:
            chunks.append(current)

        for chunk in chunks:
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
