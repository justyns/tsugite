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
from tsugite_daemon.plugin_wiring import attach_plugin_executors, attach_plugin_http
from tsugite_daemon.session_store import SessionStore

logger = logging.getLogger(__name__)


def _build_notifier(discord_adapters: dict, push_store=None, vapid_private_key=None, vapid_claims=None):
    async def _notify(message: str, channel_configs: list, url: str | None = None) -> dict:
        results = {}
        linked = f"{message}\n\n{url}" if url else message
        for name, config in channel_configs:
            try:
                if config.type == "discord":
                    results[name] = await _send_discord_dm(discord_adapters, config, linked)
                elif config.type == "webhook":
                    results[name] = await _send_webhook(config, linked)
                elif config.type == "web-push":
                    results[name] = await _send_web_push_all(
                        push_store, message, vapid_private_key, vapid_claims, url=url
                    )
            except Exception as e:
                logger.error("Notification to '%s' failed: %s", name, e)
                results[name] = {"error": str(e)}
        return results

    return _notify


async def _send_web_push_all(
    push_store, message: str, vapid_private_key: str, vapid_claims: dict, url: str | None = None
) -> dict:
    """Send web push to all stored subscriptions, pruning expired ones."""
    if not push_store:
        return {"error": "push store not initialized"}

    from tsugite_daemon.push import send_web_push

    subs = push_store.all()
    if not subs:
        return {"status": "no_subscribers"}

    payload = {"title": "Tsugite", "body": message[:200], "url": url or "/"}
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


async def _send_webhook(config, message: str) -> dict:
    """Send a notification via webhook."""
    import httpx

    from tsugite.user_agent import set_user_agent_header

    body = _render_webhook_body(config.body_template, message)
    headers = dict(config.headers)
    set_user_agent_header(headers)

    if not config.body_template:
        headers.setdefault("Content-Type", "text/plain")

    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.request(config.method, config.url, content=body, headers=headers)
        resp.raise_for_status()

    return {"status": "sent", "status_code": resp.status_code}


def check_sandbox_prerequisites(config: DaemonConfig) -> None:
    """Fail closed if sandboxing is enabled but bwrap is unavailable.

    Run once at daemon startup so a misconfigured host surfaces immediately
    instead of every sandboxed turn failing (or, worse, running unsandboxed).
    """
    from tsugite.core.sandbox import sandbox_available

    if config.sandbox and config.sandbox.enabled and not sandbox_available():
        raise RuntimeError(
            "Sandbox enabled but 'bwrap' was not found on PATH. "
            "Install bubblewrap, or set sandbox.enabled: false. "
            "(Sandboxing is Linux-only and needs user-namespace support.)"
        )


# DaemonConfig sections that only take effect at boot: a change to any of them is
# reported as restart_required. The rest (runtime defaults, notification_channels,
# identity_links) are hot-reconciled. Exported so the coverage test checks the same
# set reload_config uses instead of a hand-maintained copy.
RUNTIME_DEFAULT_FIELDS = (
    "default_workspace_dir",
    "default_agent_file",
    "default_model",
    "default_compaction_model",
    "default_context_limit",
    "default_max_turns",
    "timezone",
    "auto_compact",
)
"""DaemonConfig fields that feed RuntimeDefaults; hot-reconciled on reload."""

BOOT_ONLY_SECTIONS = (
    "http",
    "state_dir",
    "discord_bots",
    "plugins",
    "sandbox",
    "log_level",
    "log_file",
    "log_to_console",
)


class Gateway:
    """Main daemon gateway routing messages between platform adapters and agents."""

    # The drain must outlast one more model round-trip: the turn that asked for the
    # restart is still in flight when the tool call returns.
    _drain_deadline = 120.0
    _drain_poll = 0.5

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
        self._identity_map: dict[str, str] = {}
        self._shutting_down = False
        self.restart_requested = False

    async def start(self):
        """Start all enabled adapters."""
        from tsugite.tools import set_daemon_mode

        set_daemon_mode(True)

        # Fail closed: refuse to start if an agent opted into sandboxing but the
        # host can't provide it, rather than silently running its code unsandboxed.
        check_sandbox_prerequisites(self.config)

        loop = asyncio.get_running_loop()

        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, self._on_signal)

        # Build reverse identity map: "discord:123456789" -> "DiscordUsername". Kept on
        # self (and passed to adapters BY REFERENCE) so a config reload can
        # rebuild it in place for every holder at once.
        identity_map: dict[str, str] = {
            pid: canonical for canonical, platform_ids in self.config.identity_links.items() for pid in platform_ids
        }
        self._identity_map = identity_map

        runtime = self.config.runtime
        runtime.context_limit = self._resolve_context_limit(runtime)

        # Single global session store. UI events live in the same per-session
        # JSONL as conversation history (XDG data dir, not the daemon state dir).
        session_store = SessionStore(
            self.config.state_dir / "session_store.json",
            default_context_limit=runtime.context_limit,
        )
        self._session_store = session_store

        tasks = []
        http_adapter = None

        agent_path = resolve_agent_path(runtime.agent_file, runtime.workspace_dir)
        if not agent_path:
            raise ValueError(
                f"Agent file '{runtime.agent_file}' not found. "
                f"Searched in workspace '{runtime.workspace_dir}' and standard paths."
            )
        logger.info("Daemon runtime using agent: %s", agent_path)

        if self.config.discord_bots:
            try:
                from tsugite_discord import DiscordAdapter
            except ImportError as e:
                raise ImportError(
                    "Discord support requires the tsugite-discord package. "
                    "Install with: pip install tsugite-cli[daemon] (or pip install tsugite-discord)."
                ) from e

            for bot_config in self.config.discord_bots:
                self.adapters.append(
                    DiscordAdapter(
                        bot_config=bot_config,
                        runtime=runtime,
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

            http_adapter = HTTPAgentAdapter(runtime, session_store, identity_map=identity_map)

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
                http_adapter,
                webhook_store,
                gateway=self,
                token_store=self._token_store,
            )

            # Wire up event_bus on the adapter so it can broadcast compaction state
            http_adapter.event_bus = self._http_server.event_bus

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

        # Start scheduler (requires the HTTP adapter to execute runs)
        if http_adapter:
            from tsugite_daemon.adapters.scheduler_adapter import SchedulerAdapter

            schedules_path = self.config.state_dir / "schedules.json"
            self._scheduler_adapter = SchedulerAdapter(
                http_adapter,
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
            set_scheduler(self._scheduler_adapter.scheduler, asyncio.get_running_loop(), channel_names)

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
                http_adapter,
                event_bus=event_bus,
                notification_channels=self.config.notification_channels,
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
            # Expose to the adapter so the /run slash command can reach them.
            http_adapter.terminal_store = self._terminal_store
            http_adapter.pty_manager = self._pty_manager
            http_adapter.terminal_state_change_callback = terminal_state_change_cb

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
            self._jobs_orchestrator.reconcile_orphaned_host_sessions()
            set_jobs_orchestrator(self._jobs_orchestrator, asyncio.get_running_loop())
            if self._http_server:
                self._http_server.jobs_orchestrator = self._jobs_orchestrator
                self._http_server.job_store = self._job_store

            if self._scheduler_adapter:
                self._scheduler_adapter.set_session_runner(self._session_runner)
            logger.info("Session runner + Jobs orchestrator enabled")

        # Start compaction scheduler when auto_compact is configured
        if runtime.auto_compact and runtime.auto_compact.schedule and http_adapter:
            from tsugite_daemon.compaction_scheduler import CompactionScheduler

            self._compaction_scheduler = CompactionScheduler(runtime, session_store, http_adapter)
            tasks.append(self._compaction_scheduler.start())
            logger.info("Compaction scheduler enabled (%s)", runtime.auto_compact.schedule)

        # Load adapter plugins
        from tsugite.plugins import load_adapter_plugins
        from tsugite.ui_surfaces import registered_ui_surfaces

        # Pages registered when `set_daemon_mode(True)` above imported the plugin modules.
        pages = registered_ui_surfaces()
        for descriptor in pages.pop("", []):
            logger.warning("UI surface %r was registered outside a plugin load; skipping it", descriptor.get("kind"))

        plugin_results = load_adapter_plugins(
            plugin_config=self.config.plugins,
            session_store=session_store,
            identity_map=identity_map,
            runtime=runtime,
        )
        for info, adapter in plugin_results:
            # A plugin whose adapter is disabled or failed loses its page too: it
            # would open onto routes that never mounted.
            declared = pages.pop(info.name, [])
            if adapter:
                self.adapters.append(adapter)
                attach_plugin_http(self._http_server, info.name, adapter, declared)
                attach_plugin_executors(self._jobs_orchestrator, info.name, adapter)
                tasks.append(adapter.start())
        for name, declared in pages.items():
            attach_plugin_http(self._http_server, name, None, declared)

        # Set up notification callback if channels are configured
        if self.config.notification_channels:
            discord_adapters = {a.bot_config.name: a for a in self.adapters if hasattr(a, "bot_config")}
            notifier = _build_notifier(discord_adapters, self._push_store, self._vapid_private_key, self._vapid_claims)

            from tsugite.tools.notify import set_notifier

            set_notifier(notifier, loop)

        from tsugite.tools.daemon_control import set_restart_controller

        set_restart_controller(self, loop)

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

    def _force_exit_http(self):
        """Make uvicorn exit without waiting for open connections."""
        if self._http_server and self._http_server._server:
            self._http_server._server.force_exit = True

    def _on_signal(self):
        # uvicorn re-raises a captured signal once serve() returns, so a Ctrl-C
        # during the drain must cancel the restart rather than re-exec a daemon
        # the user just asked to stop.
        self.restart_requested = False
        if self._shutting_down:
            logger.info("Forced shutdown")
            self._force_exit_http()
            asyncio.get_running_loop().stop()
            return
        asyncio.create_task(self._shutdown())

    def preflight_restart(self) -> list[str]:
        """Problems that would stop the daemon coming back from a restart.

        A restarted daemon dies on an unloadable daemon.yaml, and silently loads no
        plugins on an unparseable config.json, so both are checked before prompting.
        """
        from tsugite.plugins import check_plugin_config

        problems = []
        try:
            load_daemon_config(self.config_path)
        except Exception as e:
            problems.append(f"Daemon config would not load: {e}")
        return problems + check_plugin_config()

    def request_restart(self) -> None:
        """Flag a restart and start draining the in-flight turns.

        Returns immediately: the turn that asked for the restart is itself in
        `_active_chats` and only pops once its tool call returns, so awaiting the
        drain here would deadlock.
        """
        self.restart_requested = True
        logger.info("Restart requested; draining in-flight turns")
        asyncio.create_task(self._drain_then_shutdown())

    async def _drain_then_shutdown(self):
        """Wait for the in-flight turns to finish, then shut down so the CLI re-execs."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._drain_deadline
        # A signal during the drain clears the flag, and then this task should stop
        # rather than hold the gateway alive until the deadline.
        while self.restart_requested and self._http_server and self._http_server._active_chats:
            if loop.time() >= deadline:
                logger.warning(
                    "Restart drain deadline reached with %d turn(s) in flight; forcing shutdown",
                    len(self._http_server._active_chats),
                )
                self._force_exit_http()
                break
            await asyncio.sleep(self._drain_poll)
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
        # Prefer an inherited override stamped on the session (a sandboxed parent's
        # policy) over the daemon default, mirroring the chokepoint - else a terminal
        # opened for a sandboxed child session would run on the host when the daemon
        # default has sandbox disabled.
        override = (getattr(session, "metadata", None) or {}).get("sandbox_override")
        if isinstance(override, dict):
            sb = SandboxSettings.model_validate(override)
        else:
            sb = self.config.sandbox
        if sb is None or not sb.enabled:
            return None
        workspace = getattr(session, "workspace_override", None) or self.config.default_workspace_dir
        return SandboxContext(
            allow_domains=list(sb.allow_domains),
            no_network=sb.no_network,
            extra_ro_binds=list(sb.extra_ro_binds),
            extra_rw_binds=list(sb.extra_rw_binds),
            pass_env=list(getattr(sb, "pass_env", [])),
            workspace_dir=Path(workspace) if workspace else None,
        )

    @staticmethod
    def _resolve_context_limit(runtime) -> int:
        """The effective default context limit: explicit config, else
        auto-detected from the model, else the built-in default."""
        default_context_limit = 128000
        if runtime.context_limit:
            return runtime.context_limit
        if runtime.model:
            from tsugite_daemon.memory import get_context_limit

            limit = get_context_limit(runtime.model, fallback=default_context_limit)
            logger.info("Auto-detected context limit: %d tokens", limit)
            return limit
        return default_context_limit

    async def reload_config(self) -> dict:
        """Re-read the daemon YAML and hot-apply what can be applied at runtime.

        Reconciled live: the runtime defaults (hot-swapped on the adapter, applying
        on its NEXT run), notification channels, and identity links (the map is
        mutated in place - every adapter holds a reference). Boot-only sections
        (http, state_dir, discord_bots, plugins, sandbox, logging) are compared and
        reported under restart_required instead of silently ignored.
        """
        new = load_daemon_config(self.config_path)
        result: dict = {"updated": [], "restart_required": []}

        # Keep the session store's default limit in step with the reloaded config.
        runtime = new.runtime
        runtime.context_limit = self._resolve_context_limit(runtime)
        # Sandbox is boot-only (BOOT_ONLY_SECTIONS): keep the policy the daemon
        # started with, so agent turns and terminals resolve the same one and a
        # changed `sandbox:` really does need the restart it reports.
        runtime.sandbox = self.config.sandbox
        if self._session_store is not None:
            self._session_store.update_context_limit(runtime.context_limit)

        # Every DaemonConfig section is hot-reconciled (runtime defaults,
        # notification_channels, identity_links) or boot-only (BOOT_ONLY_SECTIONS); a
        # coverage test asserts none is silently omitted (see test_config_reload).
        result["restart_required"] = [
            name for name in BOOT_ONLY_SECTIONS if getattr(self.config, name) != getattr(new, name)
        ]

        if self._http_server and self._http_server.adapter is not None:
            # BaseAdapter reads its runtime per run, so this applies on the next turn.
            if self._http_server.adapter.runtime != runtime:
                self._http_server.adapter.runtime = runtime
                result["updated"].append("runtime")

        for field in RUNTIME_DEFAULT_FIELDS:
            setattr(self.config, field, getattr(new, field))
        self.config.notification_channels = new.notification_channels
        self.config.identity_links = new.identity_links
        self._identity_map.clear()
        self._identity_map.update(
            {pid: canonical for canonical, platform_ids in new.identity_links.items() for pid in platform_ids}
        )

        logger.info(
            "Config reloaded: ~%d%s",
            len(result["updated"]),
            f" (restart required for: {', '.join(result['restart_required'])})" if result["restart_required"] else "",
        )
        return result

    async def _shutdown(self):
        """Graceful shutdown of all adapters."""
        if self._shutting_down:
            return
        self._shutting_down = True

        from tsugite_pty.tools import set_terminal_runtime

        from tsugite.tools import set_daemon_mode
        from tsugite.tools.daemon_control import set_restart_controller
        from tsugite.tools.jobs import set_jobs_orchestrator
        from tsugite.tools.notify import set_notifier
        from tsugite.tools.schedule import set_scheduler
        from tsugite.tools.sessions import set_session_runner

        set_notifier(None)
        set_scheduler(None)
        set_session_runner(None)
        set_jobs_orchestrator(None, None)
        set_restart_controller(None, None)
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
) -> bool:
    """Main daemon entry point. Returns True when a restart was requested."""
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
    return gateway.restart_requested
