"""Fixtures for Playwright E2E tests."""

import asyncio
import socket
import threading
import time
import uuid
from unittest.mock import AsyncMock, patch

import pytest
import uvicorn
from tsugite_daemon.adapters.http import HTTPAgentAdapter, HTTPServer
from tsugite_daemon.auth import TokenStore
from tsugite_daemon.config import HTTPConfig, RuntimeDefaults
from tsugite_daemon.session_store import SessionStore
from tsugite_daemon.webhook_store import WebhookStore

from .helpers import E2E_USER_ID, wait_for_authed


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="session")
def e2e_tmp(tmp_path_factory):
    return tmp_path_factory.mktemp("e2e")


@pytest.fixture(scope="session")
def e2e_workspace(e2e_tmp):
    ws = e2e_tmp / "workspace"
    ws.mkdir()
    return ws


@pytest.fixture(scope="session")
def e2e_session_store(e2e_tmp):
    return SessionStore(e2e_tmp / "sessions.json", default_context_limit=128000)


@pytest.fixture(scope="session")
def e2e_token_store(e2e_tmp):
    return TokenStore(e2e_tmp / "tokens.json")


@pytest.fixture(scope="session")
def e2e_auth_token(e2e_token_store):
    _meta, raw = e2e_token_store.create_admin_token(name="e2e-test")
    return raw


@pytest.fixture(scope="session")
def e2e_adapter(e2e_workspace, e2e_session_store):
    runtime = RuntimeDefaults(workspace_dir=e2e_workspace, agent_file="default")

    with patch("tsugite.workspace.Workspace") as mock_ws:
        from tsugite.workspace import WorkspaceNotFoundError

        mock_ws.load.side_effect = WorkspaceNotFoundError("not found")
        adapter = HTTPAgentAdapter(
            runtime=runtime,
            session_store=e2e_session_store,
        )

    # Tripwire: every e2e test that triggers chat must use mock_chat() first.
    # No real provider calls are allowed from this suite. Replace the default
    # handle_message with a raiser; mock_chat() swaps it for a configured fake.
    async def _require_mock_chat(*args, **kwargs):
        raise AssertionError(
            "e2e tests must call mock_chat(...) before sending a message. "
            "Real handle_message was invoked without the fixture; this would "
            "hit a real LLM provider."
        )

    adapter._original_handle_message = _require_mock_chat
    adapter.handle_message = _require_mock_chat
    return adapter


@pytest.fixture(scope="session")
def e2e_server(e2e_tmp, e2e_workspace, e2e_adapter, e2e_token_store):
    port = _free_port()
    config = HTTPConfig(enabled=True, host="127.0.0.1", port=port)

    server = HTTPServer(
        config=config,
        adapter=e2e_adapter,
        webhook_store=WebhookStore(e2e_tmp / "webhooks.json"),
        token_store=e2e_token_store,
    )

    uvi_config = uvicorn.Config(server.app, host="127.0.0.1", port=port, log_level="warning")
    uvi_server = uvicorn.Server(uvi_config)
    uvi_server.install_signal_handlers = lambda: None

    thread = threading.Thread(target=asyncio.run, args=(uvi_server.serve(),), daemon=True)
    thread.start()

    for _ in range(50):
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.1):
                break
        except OSError:
            time.sleep(0.1)

    yield f"http://127.0.0.1:{port}", server

    uvi_server.should_exit = True
    thread.join(timeout=5)


@pytest.fixture(scope="session")
def base_url(e2e_server):
    url, _ = e2e_server
    return url


@pytest.fixture(autouse=True)
def _reset_daemon_state(e2e_session_store, e2e_server):
    """Drop every session and any live server state between tests.

    The daemon (uvicorn) and its session store are `scope="session"` because
    standing up a fresh server per test would be too slow. Clearing the
    in-memory dicts at fixture setup is cheap and gives each test a clean
    sidebar plus a clean live-progress map.
    """
    _url, server = e2e_server
    with e2e_session_store._lock:
        e2e_session_store._sessions.clear()
        e2e_session_store._thread_index.clear()
        e2e_session_store._channel_index.clear()
    server._active_chats.clear()
    yield


@pytest.fixture(autouse=True)
def _e2e_jsonl_history(reset_history_backend_fixture):
    """Pin the JSONL history backend for the e2e suite.

    These tests were written for the JSONL era: they seed `<id>.jsonl` files and
    patch `tsugite.history.sqlite_backend.get_history_dir`. The daemon now defaults to the
    SQLite backend, so without this the in-process daemon would never read the
    seeded files. Depends on the global (autouse) reset_history_backend_fixture so
    that reset runs first on setup and the default backend is restored on teardown.
    """
    from tsugite.history import set_history_backend
    from tsugite.history.sqlite_backend import SqliteHistoryBackend

    set_history_backend(SqliteHistoryBackend())


@pytest.fixture
def session_runner_backend(e2e_server, e2e_session_store, e2e_adapter):
    """Wire a real SessionRunner onto the server.

    Almost every session-mutation endpoint (rename, pin, mark-viewed, cancel,
    restart, metadata, events - see `adapters/http/sessions.py`) 503s with
    "session runner not available" without this; it's not optional plumbing
    for a handful of tests, it's load-bearing for basic UI interaction since
    even selecting an unread seeded session fires mark-viewed. Wraps the same
    `e2e_session_store` tests already seed through, so there's one source of
    truth. Restart/cancel-style flows that start a *new* run still route
    through `adapter.handle_message` - a test that exercises one of those must
    still call `mock_chat(...)` first.
    """
    from tsugite_daemon.session_runner import SessionRunner

    _url, server = e2e_server
    runner = SessionRunner(e2e_session_store, e2e_adapter, event_bus=server.event_bus)
    server.session_runner = runner
    yield runner
    server.session_runner = None


@pytest.fixture
def authenticated_page(page, base_url, e2e_auth_token, session_runner_backend):
    """Page with auth token + a fixed synthetic user id pre-injected into localStorage."""
    page.goto(base_url + "/api/health")
    page.evaluate(f"localStorage.setItem('tsugite_token', '{e2e_auth_token}')")
    page.evaluate(f"localStorage.setItem('tsugite_user_id', {E2E_USER_ID!r})")
    page.goto(base_url)
    wait_for_authed(page)
    return page


@pytest.fixture
def chat_page(authenticated_page, e2e_session_store):
    """Authenticated page with the (default) Chats view open on a seeded session.

    Seeding happens before the reload so the view's own on-mount session load
    picks it up - the Chats surface is the default-docked tab, so a fresh page
    load already lands here without any nav click.
    """
    page = authenticated_page
    e2e_session_store.get_or_create_interactive(E2E_USER_ID)

    page.reload()
    wait_for_authed(page)
    # The session-menu trigger only renders once a session is actually
    # selected (Conversation.svelte gates it on `row`), so waiting on it
    # confirms the seeded session - not just the empty chat shell - is live.
    page.wait_for_selector('[data-testid="chat-session-menu-trigger"]', timeout=5000)
    return page


@pytest.fixture
def mock_chat(e2e_adapter):
    """Factory to configure what the mock agent returns during chat.

    Usage:
        mock_chat("Hello!", events=[("reaction", {"emoji": "👍"})])

    `delay` holds the turn in flight, so a test can observe mid-turn UI state.
    """
    _original = getattr(e2e_adapter, "_original_handle_message", None)
    if _original is None:
        e2e_adapter._original_handle_message = e2e_adapter.handle_message

    def _configure(response="Test response", events=None, delay=0):
        async def fake_handle(user_id, message, channel_context, custom_logger=None):
            if custom_logger and events:
                handler = custom_logger.ui_handler
                for ev_type, ev_data in events:
                    handler._emit(ev_type, ev_data)
            if delay:
                await asyncio.sleep(delay)
            return response

        e2e_adapter.handle_message = AsyncMock(side_effect=fake_handle)

    yield _configure

    # Restore original (or previous mock) after test
    if _original:
        e2e_adapter.handle_message = _original


# ---------------------------------------------------------------------------
# View-specific backends. Production wires these onto HTTPServer post-construction
# (`gateway.py`: "Set by Gateway ..."), so e2e_server leaves them as their None
# defaults for tests that don't need them. Each fixture below wires the real
# (lightweight, file-backed) store straight onto the running server for the
# duration of one test and unwires it after - no daemon restart needed.
# ---------------------------------------------------------------------------


@pytest.fixture
def job_store(e2e_server, e2e_tmp):
    """A real JobStore wired onto the server, for GET/read-path job tests.

    GET /api/jobs only touches `job_store` (see jobs.py), so seeding a Job
    directly here is enough to exercise listing/rendering without needing a
    live orchestrator or touching any LLM.
    """
    from tsugite_daemon.job_store import JobStore

    _url, server = e2e_server
    store = JobStore(e2e_tmp / f"jobs-{uuid.uuid4().hex}.json")
    server.job_store = store
    yield store
    server.job_store = None


@pytest.fixture
def jobs_backend(e2e_server, session_runner_backend, e2e_tmp):
    """Full job-creation stack (JobStore + JobsOrchestrator on the shared SessionRunner).

    Needed for POST /api/jobs specifically: `_api_create_job` 503s without a
    live `jobs_orchestrator` (see `_require_auth_and_jobs`). The orchestrator's
    `create_and_start_job` schedules a background task that calls back into
    `adapter.handle_message` - the same seam `mock_chat` patches - so a test
    using this fixture MUST call `mock_chat(...)` before POSTing, or the
    tripwire in `e2e_adapter` will raise (harmlessly, in the background task,
    but the run is then not clean).
    """
    from tsugite_daemon.job_store import JobStore
    from tsugite_daemon.jobs_orchestrator import JobsOrchestrator

    _url, server = e2e_server
    store = JobStore(e2e_tmp / f"jobs-{uuid.uuid4().hex}.json")
    orchestrator = JobsOrchestrator(store, session_runner_backend, event_bus=server.event_bus)
    server.job_store = store
    server.jobs_orchestrator = orchestrator
    yield store
    server.job_store = None
    server.jobs_orchestrator = None


@pytest.fixture
def terminal_backend(e2e_server, e2e_tmp):
    """A real TerminalSessionStore + PtyManager wired onto the server.

    Terminals are real OS ptys (no fake/injectable backend - see
    `terminal_runtime.spawn_terminal`), so this is the same minimal recipe
    `tests/daemon/test_terminal_endpoints.py` uses: no LLM/provider involved
    anywhere in this path.
    """
    from tsugite_pty.pty_manager import PtyManager
    from tsugite_pty.terminal_store import TerminalSessionStore

    _url, server = e2e_server
    store = TerminalSessionStore(e2e_tmp / f"terminals-{uuid.uuid4().hex}.json")
    manager = PtyManager()
    server.terminal_store = store
    server.pty_manager = manager
    yield store, manager
    server.terminal_store = None
    server.pty_manager = None
    manager.shutdown()


@pytest.fixture
def scheduler_backend(e2e_server, e2e_tmp):
    """A real Scheduler wired onto the server, run loop never started.

    `Scheduler.add()` just persists the entry; nothing fires it without the
    scheduler's own asyncio loop running (never started here), so this is
    safe to seed directly with no notion of a real agent run.
    """
    from tsugite_daemon.scheduler import Scheduler

    _url, server = e2e_server
    scheduler = Scheduler(e2e_tmp / f"schedules-{uuid.uuid4().hex}.json", AsyncMock())
    server.scheduler = scheduler
    yield scheduler
    server.scheduler = None


@pytest.fixture
def writable_secrets_backend(e2e_tmp):
    """Swap the process-global secrets backend for a writable file backend.

    The daemon's default ("env") backend rejects writes (`EnvSecretBackend.set`
    raises NotImplementedError - env vars aren't settable at runtime), so
    POST /api/secrets/{name} 400s against it. Tests exercising the write path
    need this fixture; read-only tests don't.

    Reads/restores the module's private `_backend` slot directly rather than
    going through `get_backend()` for the "original" snapshot: `get_backend()`
    eagerly constructs a backend from the *real* local tsugite config if none
    is set yet, which can itself raise (e.g. a sqlite backend configured on
    the dev machine needs a passphrase this process doesn't have) - a purely
    defensive save/restore shouldn't be able to fail like that.
    """
    import tsugite.secrets as secrets_module
    from tsugite.secrets.file import FileSecretBackend

    original = secrets_module._backend
    backend = FileSecretBackend({"path": str(e2e_tmp / f"secrets-{uuid.uuid4().hex}")})
    secrets_module.set_backend(backend)
    yield backend
    secrets_module._backend = original
