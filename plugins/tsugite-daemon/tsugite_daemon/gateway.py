"""Main daemon gateway coordinating all adapters."""

import asyncio
import logging
import logging.handlers
import signal
import sys
import threading
from pathlib import Path
from typing import Optional

from tsugite_daemon.adapters.base import BaseAdapter, resolve_agent_path
from tsugite_daemon.config import DaemonConfig, load_daemon_config
from tsugite_daemon.session_store import SessionStore

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

    from tsugite_daemon.push import send_web_push

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


def _render_webhook_body(body_template: str, message: str) -> str:
    """Substitute {message} into the template.

    JSON templates (e.g. `{"text": "{message}"}`) get a JSON-escaped message -
    a quote/newline in agent output would otherwise corrupt the payload or
    inject sibling keys. Non-JSON templates get the raw text.
    """
    import json as _json

    if not body_template:
        return message
    try:
        _json.loads(body_template.replace("{message}", ""))
        escaped = _json.dumps(message)[1:-1]
        return body_template.replace("{message}", escaped)
    except _json.JSONDecodeError:
        return body_template.replace("{message}", message)


def attach_plugin_http(http_server, plugin_name: str, adapter) -> None:
    """Wire a loaded adapter plugin's HTTP surface into the daemon's Starlette app.

    Sets the shared SSE bus on the adapter (so it can broadcast events), then
    mounts its `get_http_routes()` (auth-wrapped) and `get_public_http_routes()`
    (no auth) under `/api/plugins/<plugin_name>`. A plugin that lacks the methods
    is skipped, and one that raises while producing its routes is logged and skipped.
    """
    if http_server is not None:
        try:
            adapter.event_bus = http_server.event_bus
        except Exception:  # noqa: BLE001 -- a read-only/exotic adapter shouldn't abort startup
            logger.debug("Could not set event_bus on plugin '%s'", plugin_name)

    def _collect(method_name: str) -> list:
        method = getattr(adapter, method_name, None)
        if method is None:
            return []
        try:
            return list(method() or [])
        except Exception:
            logger.warning("Plugin '%s' %s() raised; skipping those routes", plugin_name, method_name, exc_info=True)
            return []

    authed = _collect("get_http_routes")
    public = _collect("get_public_http_routes")
    if not authed and not public:
        return
    if http_server is None:
        logger.warning("Plugin '%s' registers HTTP routes but HTTP is disabled; skipping", plugin_name)
        return
    http_server.mount_plugin_routes(plugin_name, authed, public)


def attach_plugin_executors(jobs_orchestrator, plugin_name: str, adapter) -> None:
    """Register a loaded adapter plugin's job executors on the orchestrator.

    Reads `get_job_executors() -> dict[str, executor]` so a plugin (e.g.
    cc-driver) can supply a non-agent executor. No-op when the plugin exposes no
    executors or the orchestrator is disabled; a plugin that raises while
    producing its executors is logged and skipped.
    """
    method = getattr(adapter, "get_job_executors", None)
    if method is None:
        return
    try:
        executors = dict(method() or {})
    except Exception:
        logger.warning("Plugin '%s' get_job_executors() raised; skipping", plugin_name, exc_info=True)
        return
    if not executors:
        return
    if jobs_orchestrator is None:
        logger.warning(
            "Plugin '%s' registers job executors but the jobs orchestrator is disabled; skipping", plugin_name
        )
        return
    # Hand the orchestrator back to the adapter so its executors can report
    # completion/failure via complete_worker/fail_worker.
    setter = getattr(adapter, "set_jobs_orchestrator", None)
    if setter is not None:
        try:
            setter(jobs_orchestrator)
        except Exception:
            logger.warning("Plugin '%s' set_jobs_orchestrator() raised", plugin_name, exc_info=True)
    for name, executor in executors.items():
        jobs_orchestrator.register_executor(name, executor)


async def _send_webhook(config, message: str) -> dict:
    """Send a notification via webhook."""
    import httpx

    from tsugite.user_agent import set_user_agent_header

    body = _render_webhook_body(config.body_template, message) if config.body_template else message
    headers = dict(config.headers)
    set_user_agent_header(headers)

    if not config.body_template:
        headers.setdefault("Content-Type", "text/plain")

    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.request(config.method, config.url, content=body, headers=headers)
        resp.raise_for_status()

    return {"status": "sent", "status_code": resp.status_code}


def check_sandbox_prerequisites(config: DaemonConfig) -> None:
    """Fail closed if any agent enabled sandboxing but bwrap is unavailable.

    Run once at daemon startup so a misconfigured host surfaces immediately
    instead of every sandboxed turn failing (or, worse, running unsandboxed).
    """
    from tsugite.core.sandbox import sandbox_available

    enabled = sorted(name for name, agent in config.agents.items() if agent.sandbox and agent.sandbox.enabled)
    if enabled and not sandbox_available():
        raise RuntimeError(
            f"Sandbox enabled for agents {enabled} but 'bwrap' was not found on PATH. "
            "Install bubblewrap, or set sandbox.enabled: false for these agents. "
            "(Sandboxing is Linux-only and needs user-namespace support.)"
        )


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
        self._terminal_store = None
        self._pty_manager = None
        self._jobs_orchestrator = None
        self._job_store = None
        self._shutting_down = False

    async def start(self):
        """Start all enabled adapters."""
        from tsugite.tools import set_daemon_mode

        set_daemon_mode(True)

        # Fail closed: refuse to start if an agent opted into sandboxing but the
        # host can't provide it, rather than silently running its code unsandboxed.
        check_sandbox_prerequisites(self.config)

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
                from tsugite_daemon.memory import get_context_limit

                context_limit = get_context_limit(agent_config.model, fallback=default_context_limit)
                logger.info("[%s] Auto-detected context limit: %d tokens", agent_name, context_limit)
            else:
                context_limit = default_context_limit

            agent_config.context_limit = context_limit
            context_limits[agent_name] = context_limit

        # Single global session store. UI events live in the same per-session
        # JSONL as conversation history (XDG data dir, not the daemon state dir).
        from tsugite.history import get_history_dir

        session_store = SessionStore(
            self.config.state_dir / "session_store.json",
            context_limits=context_limits,
            history_dir=get_history_dir(),
        )
        self._session_store = session_store

        tasks = []
        http_adapters = {}

        if self.config.discord_bots:
            try:
                from tsugite_discord import DiscordAdapter
            except ImportError as e:
                raise ImportError(
                    "Discord support requires the tsugite-discord package. "
                    "Install with: pip install tsugite-cli[daemon] (or pip install tsugite-discord)."
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
                from tsugite_daemon.adapters.http import HTTPAgentAdapter, HTTPServer
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
                from tsugite_daemon.auth import TOKENS_FILENAME, TokenStore
                from tsugite_daemon.webhook_store import WebhookStore

                webhook_store = WebhookStore(self.config.state_dir / "webhooks.json")
                self._token_store = TokenStore(self.config.state_dir / TOKENS_FILENAME)

                admin_token_count = len(self._token_store.list_admin_tokens())
                if admin_token_count == 0:
                    logger.warning("No API tokens configured. Run: tsugite daemon token create")
                else:
                    logger.info("HTTP auth enabled (%d admin token(s))", admin_token_count)

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
                    from tsugite_daemon.push import PushSubscriptionStore, get_or_create_vapid_keys

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
            from tsugite_daemon.adapters.scheduler_adapter import SchedulerAdapter

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
            from tsugite.tools.jobs import set_jobs_orchestrator
            from tsugite.tools.sessions import set_session_runner
            from tsugite_daemon.job_store import JobStore
            from tsugite_daemon.jobs_orchestrator import JobsOrchestrator
            from tsugite_daemon.session_runner import SessionRunner

            event_bus = self._http_server.event_bus if self._http_server else None
            self._session_runner = SessionRunner(
                session_store,
                http_adapters,
                event_bus=event_bus,
            )
            if self._http_server:
                self._http_server.session_runner = self._session_runner
            set_session_runner(self._session_runner, asyncio.get_running_loop())

            # Terminal viewer: PTY runtime + persistent session store. Owned by
            # the gateway so it survives across HTTP restarts and shuts down
            # cleanly via _shutdown() below. Built before the orchestrator so it
            # can be passed in (job_status payloads include worker_terminal_id).
            from tsugite_pty.pty_manager import PtyManager
            from tsugite_pty.terminal_store import TerminalSessionStore

            self._terminal_store = TerminalSessionStore(self.config.state_dir / "terminal_sessions.json")
            self._pty_manager = PtyManager()
            terminal_state_change_cb = lambda tid, state: (  # noqa: E731
                event_bus.emit("terminal_state", {"terminal_id": tid, "state": state}) if event_bus else None
            )
            if self._http_server:
                self._http_server.terminal_store = self._terminal_store
                self._http_server.pty_manager = self._pty_manager
            # Expose to adapters so the /run slash command can reach them.
            for adapter in http_adapters.values():
                adapter.terminal_store = self._terminal_store
                adapter.pty_manager = self._pty_manager
                adapter.terminal_state_change_callback = terminal_state_change_cb

            # Expose the same runtime to the agent-facing @terminal tools.
            from tsugite_pty.tools import set_terminal_runtime

            set_terminal_runtime(self._pty_manager, self._terminal_store, terminal_state_change_cb)

            # Let terminals opened outside an agent turn (/run, the HTTP API)
            # inherit their parent session's agent sandbox config.
            from tsugite_pty.terminal_runtime import set_session_sandbox_resolver

            set_session_sandbox_resolver(self._resolve_session_sandbox)

            self._job_store = JobStore(self.config.state_dir / "jobs.json")
            self._jobs_orchestrator = JobsOrchestrator(
                self._job_store, self._session_runner, event_bus=event_bus, terminal_store=self._terminal_store
            )
            self._jobs_orchestrator.attach()
            self._jobs_orchestrator.recover_orphaned_jobs()
            set_jobs_orchestrator(self._jobs_orchestrator, asyncio.get_running_loop())
            if self._http_server:
                self._http_server.jobs_orchestrator = self._jobs_orchestrator
                self._http_server.job_store = self._job_store

            if self._scheduler_adapter:
                self._scheduler_adapter.set_session_runner(self._session_runner)
            logger.info("Session runner + Jobs orchestrator enabled")

        # Start compaction scheduler for agents with auto_compact config
        agents_with_auto_compact = {
            name: cfg for name, cfg in self.config.agents.items() if cfg.auto_compact and cfg.auto_compact.schedule
        }
        if agents_with_auto_compact and http_adapters:
            from tsugite_daemon.compaction_scheduler import CompactionScheduler

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
                attach_plugin_http(self._http_server, info.name, adapter)
                attach_plugin_executors(self._jobs_orchestrator, info.name, adapter)
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

    def _resolve_session_sandbox(self, session_id: str):
        """Resolve a session's agent sandbox config into a SandboxContext, or None.

        Used by the terminal runtime so a PTY opened for a session (via /run or the
        HTTP API, outside an agent turn) is sandboxed whenever that session's agent
        is configured for it.
        """
        from tsugite.agent_runner.helpers import SandboxContext
        from tsugite_daemon.config import SandboxSettings

        store = self._session_store
        if store is None:
            return None
        try:
            session = store.get_session(session_id)
        except Exception:
            return None
        agent_cfg = self.config.agents.get(getattr(session, "agent", None))
        # Prefer an inherited override stamped on the session (a sandboxed parent's
        # policy) over the target agent's own config, mirroring the chokepoint - else
        # a terminal opened for a sandboxed child session whose agent has sandbox
        # disabled would run on the host.
        override = (getattr(session, "metadata", None) or {}).get("sandbox_override")
        if isinstance(override, dict):
            sb = SandboxSettings.model_validate(override)
        else:
            sb = getattr(agent_cfg, "sandbox", None)
        if sb is None or not sb.enabled:
            return None
        workspace = getattr(session, "workspace_override", None) or (agent_cfg.workspace_dir if agent_cfg else None)
        return SandboxContext(
            allow_domains=list(sb.allow_domains),
            no_network=sb.no_network,
            extra_ro_binds=list(sb.extra_ro_binds),
            extra_rw_binds=list(sb.extra_rw_binds),
            pass_env=list(getattr(sb, "pass_env", [])),
            workspace_dir=Path(workspace) if workspace else None,
        )

    async def _shutdown(self):
        """Graceful shutdown of all adapters."""
        if self._shutting_down:
            return
        self._shutting_down = True

        from tsugite_pty.tools import set_terminal_runtime

        from tsugite.tools import set_daemon_mode
        from tsugite.tools.jobs import set_jobs_orchestrator
        from tsugite.tools.notify import set_notifier
        from tsugite.tools.schedule import set_scheduler
        from tsugite.tools.sessions import set_session_runner

        set_notifier(None)
        set_scheduler(None)
        set_session_runner(None)
        set_jobs_orchestrator(None, None)
        set_terminal_runtime(None, None, None)
        set_daemon_mode(False)

        if self._jobs_orchestrator:
            try:
                self._jobs_orchestrator.shutdown()
            except Exception as e:
                logger.error("Error shutting down jobs orchestrator: %s", e)

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

        if self._pty_manager:
            try:
                self._pty_manager.shutdown()
            except Exception as e:
                logger.error("Error shutting down PTY manager: %s", e)


_LOG_FORMAT = "%(asctime)s [%(name)s] %(levelname)s: %(message)s"
_LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def _configure_logging(config: DaemonConfig) -> None:
    """Set up root logger handlers based on daemon config."""
    level = getattr(logging, config.log_level.upper(), logging.INFO)
    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_LOG_DATE_FORMAT)

    handlers: list[logging.Handler] = []
    if config.log_to_console:
        handlers.append(logging.StreamHandler(sys.stderr))
    # Persistent log by default: without a durable file, a daemon crash leaves
    # no retrievable traceback (stderr dies with the terminal). An explicit
    # config.log_file wins; otherwise the log lands next to the daemon state.
    log_file = config.log_file or (config.state_dir / "daemon.log")
    log_file.parent.mkdir(parents=True, exist_ok=True)
    handlers.append(logging.handlers.RotatingFileHandler(log_file, maxBytes=10 * 1024 * 1024, backupCount=3))

    root = logging.getLogger()
    root.setLevel(level)
    for h in root.handlers[:]:
        root.removeHandler(h)
    for h in handlers:
        h.setFormatter(formatter)
        root.addHandler(h)


def _install_crash_hooks() -> None:
    """Route unhandled main-thread and worker-thread exceptions through logging
    so a crash traceback survives in the daemon log.

    The agent loop runs in worker threads (asyncio.to_thread), so
    threading.excepthook matters as much as sys.excepthook. asyncio's own loop
    exception handler already logs via the 'asyncio' logger and needs no hook.
    """
    crash_logger = logging.getLogger("tsugite_daemon.crash")

    def _hook(exc_type, exc, tb):
        crash_logger.critical("Unhandled exception (daemon crash)", exc_info=(exc_type, exc, tb))
        sys.__excepthook__(exc_type, exc, tb)

    sys.excepthook = _hook

    def _thread_hook(args):
        crash_logger.critical(
            "Unhandled exception in thread %r",
            args.thread.name if args.thread else "?",
            exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
        )

    threading.excepthook = _thread_hook


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
    _install_crash_hooks()

    from tsugite.secrets import configure_from_daemon as configure_secrets

    configure_secrets(config)

    gateway = Gateway(config, config_path=config_path)
    await gateway.start()
