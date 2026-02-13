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

                session_manager = SessionManager(
                    agent_name, agent_config.workspace_dir, context_limit=agent_config.context_limit
                )
                self.adapters.append(
                    DiscordAdapter(
                        bot_config=bot_config,
                        agent_name=agent_name,
                        agent_config=agent_config,
                        session_manager=session_manager,
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

                session_manager = SessionManager(
                    agent_name, agent_config.workspace_dir, context_limit=agent_config.context_limit
                )
                http_adapters[agent_name] = HTTPAgentAdapter(agent_name, agent_config, session_manager)

            if http_adapters:
                from tsugite.daemon.webhook_store import WebhookStore

                webhook_store = WebhookStore(self.config.state_dir / "webhooks.json")

                self._http_server = HTTPServer(
                    self.config.http, http_adapters, webhook_store, self.config.agents
                )
                tasks.append(self._http_server.start())

        # Collect adapter start tasks
        tasks.extend(adapter.start() for adapter in self.adapters)

        # Start scheduler (requires HTTP adapters to execute agents)
        if http_adapters:
            from tsugite.daemon.adapters.scheduler_adapter import SchedulerAdapter

            schedules_path = self.config.state_dir / "schedules.json"
            self._scheduler_adapter = SchedulerAdapter(http_adapters, schedules_path)
            tasks.append(self._scheduler_adapter.start())
            if self._http_server:
                self._http_server.scheduler = self._scheduler_adapter.scheduler

            # Give schedule tools direct access to the scheduler
            from tsugite.tools.schedule import set_scheduler

            set_scheduler(self._scheduler_adapter.scheduler, asyncio.get_running_loop())

            logger.info("Scheduler enabled (schedules: %s)", schedules_path)

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
        from tsugite.tools.schedule import set_scheduler

        set_scheduler(None)
        set_daemon_mode(False)

        components = [
            (a, "adapter") for a in self.adapters
        ]
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
