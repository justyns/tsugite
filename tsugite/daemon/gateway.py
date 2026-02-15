"""Main daemon gateway coordinating all adapters."""

import asyncio
import logging
import signal
from pathlib import Path
from typing import Optional

from tsugite.daemon.adapters.base import BaseAdapter, resolve_agent_path
from tsugite.daemon.config import DaemonConfig, load_daemon_config
from tsugite.daemon.session import SessionManager

logger = logging.getLogger(__name__)


def _build_notifier(discord_adapters: dict):
    """Build an async callback(message, channel_configs) -> dict for sending notifications."""

    async def _notify(message: str, channel_configs: list) -> dict:
        results = {}
        for name, config in channel_configs:
            try:
                if config.type == "discord":
                    results[name] = await _send_discord_dm(discord_adapters, config, message)
                elif config.type == "webhook":
                    results[name] = await _send_webhook(config, message)
            except Exception as e:
                logger.error("Notification to '%s' failed: %s", name, e)
                results[name] = {"error": str(e)}
        return results

    return _notify


async def _send_discord_dm(discord_adapters: dict, config, message: str) -> dict:
    """Send a Discord DM via the configured bot."""
    adapter = discord_adapters.get(config.bot)
    if not adapter:
        return {"error": f"Discord bot '{config.bot}' not found"}

    user = await adapter.bot.fetch_user(int(config.user_id))
    if not user:
        return {"error": f"Discord user '{config.user_id}' not found"}

    dm_channel = await user.create_dm()
    await adapter._send_chunked(dm_channel, message)
    return {"status": "sent"}


async def _send_webhook(config, message: str) -> dict:
    """Send a notification via webhook."""
    import httpx

    body = config.body_template.replace("{message}", message) if config.body_template else message
    headers = dict(config.headers)

    if not config.body_template:
        headers.setdefault("Content-Type", "text/plain")

    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.request(config.method, config.url, content=body, headers=headers)
        resp.raise_for_status()

    return {"status": "sent", "status_code": resp.status_code}


class Gateway:
    """Main daemon gateway routing messages between platform adapters and agents."""

    def __init__(self, config: DaemonConfig):
        self.config = config
        self.adapters: list[BaseAdapter] = []
        self._http_server = None
        self._scheduler_adapter = None

    async def start(self):
        """Start all enabled adapters."""
        from tsugite.tools import set_daemon_mode

        set_daemon_mode(True)

        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, lambda: asyncio.create_task(self._shutdown()))

        # Build reverse identity map: "discord:123456789" -> "justyn"
        identity_map: dict[str, str] = {}
        for canonical, platform_ids in self.config.identity_links.items():
            for pid in platform_ids:
                identity_map[pid] = canonical

        # One SessionManager per agent, shared across adapters
        agent_session_managers: dict[str, SessionManager] = {}
        for agent_name, agent_config in self.config.agents.items():
            agent_session_managers[agent_name] = SessionManager(
                agent_name, agent_config.workspace_dir, context_limit=agent_config.context_limit
            )

        tasks = []
        http_adapters = {}

        if self.config.discord_bots:
            try:
                from tsugite.daemon.adapters.discord import DiscordAdapter
            except ImportError as e:
                raise ImportError(
                    "Discord support requires discord.py. Install with: pip install tsugite-cli[daemon]"
                ) from e

            for bot_config in self.config.discord_bots:
                agent_name = bot_config.agent
                if agent_name not in self.config.agents:
                    raise ValueError(f"Discord bot '{bot_config.name}' references unknown agent '{agent_name}'")

                agent_config = self.config.agents[agent_name]
                agent_path = resolve_agent_path(agent_config.agent_file, agent_config.workspace_dir)
                if not agent_path:
                    raise ValueError(
                        f"Agent file '{agent_config.agent_file}' not found for bot '{bot_config.name}'. "
                        f"Searched in workspace '{agent_config.workspace_dir}' and standard paths."
                    )
                logger.info("Bot '%s' using agent: %s", bot_config.name, agent_path)

                self.adapters.append(
                    DiscordAdapter(
                        bot_config=bot_config,
                        agent_name=agent_name,
                        agent_config=agent_config,
                        session_manager=agent_session_managers[agent_name],
                        identity_map=identity_map,
                    )
                )

        if self.config.http and self.config.http.enabled:
            try:
                from tsugite.daemon.adapters.http import HTTPAgentAdapter, HTTPServer
            except ImportError as e:
                raise ImportError(
                    "HTTP support requires starlette and uvicorn. Install with: pip install tsugite-cli[daemon]"
                ) from e

            for agent_name, agent_config in self.config.agents.items():
                agent_path = resolve_agent_path(agent_config.agent_file, agent_config.workspace_dir)
                if not agent_path:
                    logger.warning("Skipping HTTP agent '%s': agent file not found", agent_name)
                    continue
                logger.info("HTTP agent '%s' using agent: %s", agent_name, agent_path)

                http_adapters[agent_name] = HTTPAgentAdapter(
                    agent_name, agent_config, agent_session_managers[agent_name], identity_map=identity_map
                )

            if http_adapters:
                from tsugite.daemon.webhook_store import WebhookStore

                webhook_store = WebhookStore(self.config.state_dir / "webhooks.json")

                self._http_server = HTTPServer(self.config.http, http_adapters, webhook_store, self.config.agents)
                tasks.append(self._http_server.start())

        # Collect adapter start tasks
        tasks.extend(adapter.start() for adapter in self.adapters)

        # Start scheduler (requires HTTP adapters to execute agents)
        if http_adapters:
            from tsugite.daemon.adapters.scheduler_adapter import SchedulerAdapter

            schedules_path = self.config.state_dir / "schedules.json"
            self._scheduler_adapter = SchedulerAdapter(http_adapters, schedules_path, self.config.notification_channels)
            tasks.append(self._scheduler_adapter.start())
            if self._http_server:
                self._http_server.scheduler = self._scheduler_adapter.scheduler

            # Give schedule tools direct access to the scheduler
            from tsugite.tools.schedule import set_scheduler

            channel_names = set(self.config.notification_channels.keys())
            set_scheduler(self._scheduler_adapter.scheduler, asyncio.get_running_loop(), channel_names)

            logger.info("Scheduler enabled (schedules: %s)", schedules_path)

        # Set up notification callback if channels are configured
        if self.config.notification_channels:
            discord_adapters = {a.bot_config.name: a for a in self.adapters if hasattr(a, "bot_config")}
            notifier = _build_notifier(discord_adapters)

            from tsugite.tools.notify import set_notifier

            set_notifier(notifier, loop)

        if not tasks:
            raise ValueError("No adapters enabled in config")

        adapter_count = len(self.adapters) + (1 if self._http_server else 0)
        logger.info("Starting %d adapter(s)...", adapter_count)
        try:
            await asyncio.gather(*tasks)
        except KeyboardInterrupt:
            logger.info("Shutting down...")
        finally:
            await self._shutdown()

    async def _shutdown(self):
        """Graceful shutdown of all adapters."""
        from tsugite.tools import set_daemon_mode
        from tsugite.tools.notify import set_notifier
        from tsugite.tools.schedule import set_scheduler

        set_notifier(None)
        set_scheduler(None)
        set_daemon_mode(False)

        components = [(a, "adapter") for a in self.adapters]
        if self._scheduler_adapter:
            components.append((self._scheduler_adapter, "scheduler"))
        if self._http_server:
            components.append((self._http_server, "HTTP server"))

        for component, label in components:
            try:
                await component.stop()
            except Exception as e:
                logger.error("Error stopping %s: %s", label, e)


async def run_daemon(config_path: Optional[Path] = None):
    """Main daemon entry point."""
    config = load_daemon_config(config_path)

    logging.basicConfig(
        level=getattr(logging, config.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )

    gateway = Gateway(config)
    await gateway.start()
