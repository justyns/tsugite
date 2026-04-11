"""Main daemon gateway coordinating all adapters."""

import asyncio
import logging
import logging.handlers
import signal
import sys
from pathlib import Path
from typing import Optional

from tsugite.daemon.adapters.base import BaseAdapter, resolve_agent_path
from tsugite.daemon.config import DaemonConfig, load_daemon_config
from tsugite.daemon.session_store import SessionStore

logger = logging.getLogger(__name__)


def _build_notifier(discord_adapters: dict, push_store=None, vapid_private_key=None, vapid_claims=None):
    """Build an async callback(message, channel_configs) -> dict for sending notifications."""

    async def _notify(message: str, channel_configs: list) -> dict:
        results = {}
        for name, config in channel_configs:
            try:
                if config.type == "discord":
                    results[name] = await _send_discord_dm(discord_adapters, config, message)
                elif config.type == "webhook":
                    results[name] = await _send_webhook(config, message)
                elif config.type == "web-push":
                    results[name] = await _send_web_push_all(push_store, message, vapid_private_key, vapid_claims)
            except Exception as e:
                logger.error("Notification to '%s' failed: %s", name, e)
                results[name] = {"error": str(e)}
        return results

    return _notify


async def _send_web_push_all(push_store, message: str, vapid_private_key: str, vapid_claims: dict) -> dict:
    """Send web push to all stored subscriptions, pruning expired ones."""
    if not push_store:
        return {"error": "push store not initialized"}

    from tsugite.daemon.push import send_web_push

    subs = push_store.all()
    if not subs:
        return {"status": "no_subscribers"}

    payload = {"title": "Tsugite", "body": message[:200], "url": "/"}
    sent = 0
    expired = []
    for sub in subs:
        result = await send_web_push(sub, payload, vapid_private_key, vapid_claims)
        if result.get("status") == "expired":
            expired.append(result["endpoint"])
        elif result.get("status") == "sent":
            sent += 1

    for endpoint in expired:
        push_store.unsubscribe(endpoint)

    return {"status": "sent", "sent": sent, "expired": len(expired)}


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

    from tsugite.user_agent import set_user_agent_header

    body = config.body_template.replace("{message}", message) if config.body_template else message
    headers = dict(config.headers)
    set_user_agent_header(headers)

    if not config.body_template:
        headers.setdefault("Content-Type", "text/plain")

    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.request(config.method, config.url, content=body, headers=headers)
        resp.raise_for_status()

    return {"status": "sent", "status_code": resp.status_code}


class Gateway:
    """Main daemon gateway routing messages between platform adapters and agents."""

    def __init__(self, config: DaemonConfig, config_path: Optional[Path] = None):
        self.config = config
        self.config_path = config_path
        self.adapters: list[BaseAdapter] = []
        self._http_server = None
        self._scheduler_adapter = None
        self._session_runner = None
        self._session_store = None
        self._push_store = None
        self._vapid_private_key = None
        self._vapid_claims = None
        self._compaction_scheduler = None
        self._shutting_down = False

    async def start(self):
        """Start all enabled adapters."""
        from tsugite.tools import set_daemon_mode

        set_daemon_mode(True)

        loop = asyncio.get_running_loop()

        def _on_signal():
            if self._shutting_down:
                logger.info("Forced shutdown")
                if self._http_server and self._http_server._server:
                    self._http_server._server.force_exit = True
                loop.stop()
                return
            asyncio.create_task(self._shutdown())

        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, _on_signal)

        # Build reverse identity map: "discord:123456789" -> "justyn"
        identity_map: dict[str, str] = {}
        for canonical, platform_ids in self.config.identity_links.items():
            for pid in platform_ids:
                identity_map[pid] = canonical

        # Build per-agent context limits
        default_context_limit = 128000
        context_limits: dict[str, int] = {}
        for agent_name, agent_config in self.config.agents.items():
            if agent_config.context_limit:
                context_limit = agent_config.context_limit
            elif agent_config.model:
                from tsugite.daemon.memory import get_context_limit

                context_limit = get_context_limit(agent_config.model, fallback=default_context_limit)
                logger.info("[%s] Auto-detected context limit: %d tokens", agent_name, context_limit)
            else:
                context_limit = default_context_limit

            agent_config.context_limit = context_limit
            context_limits[agent_name] = context_limit

        # Single global session store
        session_store = SessionStore(self.config.state_dir / "session_store.json", context_limits=context_limits)
        self._session_store = session_store

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
                        session_store=session_store,
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
                    agent_name, agent_config, session_store, identity_map=identity_map
                )

            if http_adapters:
                from tsugite.daemon.auth import TOKENS_FILENAME, TokenStore
                from tsugite.daemon.webhook_store import WebhookStore

                webhook_store = WebhookStore(self.config.state_dir / "webhooks.json")
                self._token_store = TokenStore(self.config.state_dir / TOKENS_FILENAME)

                if not self._token_store.has_admin_tokens():
                    logger.warning("No API tokens configured. Run: tsugite daemon token create")
                else:
                    count = len(self._token_store.list_admin_tokens())
                    logger.info("HTTP auth enabled (%d admin token(s))", count)

                self._tsugite_api_url = f"http://127.0.0.1:{self.config.http.port}"

                self._http_server = HTTPServer(
                    self.config.http,
                    http_adapters,
                    webhook_store,
                    self.config.agents,
                    gateway=self,
                    token_store=self._token_store,
                )

                # Wire up event_bus on adapters so they can broadcast compaction state
                for adapter in http_adapters.values():
                    adapter.event_bus = self._http_server.event_bus

                # Always init push store when HTTP is enabled so subscribe/unsubscribe API works
                try:
                    from tsugite.daemon.push import PushSubscriptionStore, get_or_create_vapid_keys

                    self._push_store = PushSubscriptionStore(self.config.state_dir / "push_subscriptions.json")
                    self._vapid_private_key, vapid_public = get_or_create_vapid_keys(self.config.state_dir)
                    self._vapid_claims = {"sub": "mailto:tsugite@localhost"}
                    self._http_server.push_store = self._push_store
                    self._http_server.vapid_public_key = vapid_public
                except ImportError:
                    logger.debug("pywebpush/py-vapid not installed — web push disabled")
                    self._push_store = None
                    self._vapid_private_key = None
                    self._vapid_claims = None

                tasks.append(self._http_server.start())

        # Collect adapter start tasks
        tasks.extend(adapter.start() for adapter in self.adapters)

        # Start scheduler (requires HTTP adapters to execute agents)
        if http_adapters:
            from tsugite.daemon.adapters.scheduler_adapter import SchedulerAdapter

            schedules_path = self.config.state_dir / "schedules.json"
            self._scheduler_adapter = SchedulerAdapter(
                http_adapters,
                schedules_path,
                self.config.notification_channels,
                identity_map,
                token_store=self._token_store,
                tsugite_api_url=self._tsugite_api_url,
            )
            tasks.append(self._scheduler_adapter.start())
            if self._http_server:
                self._http_server.scheduler = self._scheduler_adapter.scheduler

            # Give schedule tools direct access to the scheduler
            from tsugite.tools.schedule import set_scheduler

            channel_names = set(self.config.notification_channels.keys())
            agent_names = set(http_adapters.keys())
            set_scheduler(self._scheduler_adapter.scheduler, asyncio.get_running_loop(), channel_names, agent_names)

            logger.info("Scheduler enabled (schedules: %s)", schedules_path)

            # Start session runner (uses the unified session store)
            from tsugite.daemon.session_runner import SessionRunner
            from tsugite.tools.sessions import set_session_runner

            event_bus = self._http_server.event_bus if self._http_server else None
            self._session_runner = SessionRunner(
                session_store,
                http_adapters,
                event_bus=event_bus,
            )
            if self._http_server:
                self._http_server.session_runner = self._session_runner
            set_session_runner(self._session_runner, asyncio.get_running_loop())
            if self._scheduler_adapter:
                self._scheduler_adapter.set_session_runner(self._session_runner)
            logger.info("Session runner enabled")

        # Start compaction scheduler for agents with auto_compact config
        agents_with_auto_compact = {
            name: cfg for name, cfg in self.config.agents.items() if cfg.auto_compact and cfg.auto_compact.schedule
        }
        if agents_with_auto_compact and http_adapters:
            from tsugite.daemon.compaction_scheduler import CompactionScheduler

            self._compaction_scheduler = CompactionScheduler(agents_with_auto_compact, session_store, http_adapters)
            tasks.append(self._compaction_scheduler.start())
            logger.info(
                "Compaction scheduler enabled for %d agent(s)",
                len(agents_with_auto_compact),
            )

        # Load adapter plugins
        from tsugite.plugins import load_adapter_plugins

        plugin_results = load_adapter_plugins(
            plugin_config=self.config.plugins,
            session_store=session_store,
            identity_map=identity_map,
            agents_config=self.config.agents,
        )
        for info, adapter in plugin_results:
            if adapter:
                self.adapters.append(adapter)
                tasks.append(adapter.start())

        # Set up notification callback if channels are configured
        if self.config.notification_channels:
            discord_adapters = {a.bot_config.name: a for a in self.adapters if hasattr(a, "bot_config")}
            push_store = self._push_store
            vapid_private_key = self._vapid_private_key
            vapid_claims = self._vapid_claims
            notifier = _build_notifier(discord_adapters, push_store, vapid_private_key, vapid_claims)

            from tsugite.tools.notify import set_notifier

            set_notifier(notifier, loop)

        if not tasks:
            raise ValueError("No adapters enabled in config")

        adapter_count = len(self.adapters) + (1 if self._http_server else 0)
        logger.info("Starting %d adapter(s)...", adapter_count)
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, Exception):
                    logger.error("Adapter failed: %s", result)
        except KeyboardInterrupt:
            logger.info("Shutting down...")
        finally:
            await self._shutdown()

    async def _shutdown(self):
        """Graceful shutdown of all adapters."""
        if self._shutting_down:
            return
        self._shutting_down = True

        from tsugite.tools import set_daemon_mode
        from tsugite.tools.notify import set_notifier
        from tsugite.tools.schedule import set_scheduler
        from tsugite.tools.sessions import set_session_runner

        set_notifier(None)
        set_scheduler(None)
        set_session_runner(None)
        set_daemon_mode(False)

        # Stop HTTP server first since SSE connections block uvicorn shutdown
        if self._http_server:
            try:
                await self._http_server.stop()
            except Exception as e:
                logger.error("Error stopping HTTP server: %s", e)

        components = [(a, "adapter") for a in self.adapters]
        if self._scheduler_adapter:
            components.append((self._scheduler_adapter, "scheduler"))
        if self._compaction_scheduler:
            components.append((self._compaction_scheduler, "compaction scheduler"))

        for component, label in components:
            try:
                await component.stop()
            except Exception as e:
                logger.error("Error stopping %s: %s", label, e)

        if self._session_store:
            try:
                self._session_store.flush()
            except Exception as e:
                logger.error("Error flushing session store: %s", e)


_LOG_FORMAT = "%(asctime)s [%(name)s] %(levelname)s: %(message)s"
_LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def _configure_logging(config: DaemonConfig) -> None:
    """Set up root logger handlers based on daemon config."""
    level = getattr(logging, config.log_level.upper(), logging.INFO)
    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_LOG_DATE_FORMAT)

    handlers: list[logging.Handler] = []
    if config.log_to_console:
        handlers.append(logging.StreamHandler(sys.stderr))
    if config.log_file:
        config.log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.handlers.RotatingFileHandler(config.log_file, maxBytes=10 * 1024 * 1024, backupCount=3))
    if not handlers:
        handlers.append(logging.NullHandler())

    root = logging.getLogger()
    root.setLevel(level)
    for h in root.handlers[:]:
        root.removeHandler(h)
    for h in handlers:
        h.setFormatter(formatter)
        root.addHandler(h)


async def run_daemon(
    config_path: Optional[Path] = None,
    config_overrides: Optional[dict] = None,
):
    """Main daemon entry point."""
    config = load_daemon_config(config_path)
    if config_overrides:
        for key, value in config_overrides.items():
            setattr(config, key, value)

    _configure_logging(config)

    from tsugite.secrets import configure_from_daemon as configure_secrets

    configure_secrets(config)

    gateway = Gateway(config, config_path=config_path)
    await gateway.start()
