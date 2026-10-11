"""Script schedules that opt in get a scoped, per-run daemon token in their env."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from starlette.testclient import TestClient
from tsugite_daemon.adapters.http import HTTPAgentAdapter, HTTPServer
from tsugite_daemon.adapters.scheduler_adapter import SchedulerAdapter
from tsugite_daemon.auth import SCOPE_SCRIPT_RUN, TokenStore
from tsugite_daemon.config import HTTPConfig, RuntimeDefaults
from tsugite_daemon.scheduler import ScheduleEntry
from tsugite_daemon.session_runner import SessionRunner
from tsugite_daemon.session_store import Session, SessionSource, SessionStore
from tsugite_daemon.webhook_store import WebhookStore

OWN_SCHEDULE = "nightly"
OWN_SESSION = "sess-own"
OTHER_SESSION = "sess-other"
OWN_JOB = "job-own"
OTHER_JOB = "job-other"
OTHER_JOB_UNKNOWN_SESSION = "job-orphan"

ALLOWED = [
    ("GET", f"/api/schedules/{OWN_SCHEDULE}"),
    ("GET", f"/api/schedules/{OWN_SCHEDULE}/sessions"),
    ("GET", f"/api/sessions/{OWN_SESSION}"),
    ("GET", f"/api/sessions/{OWN_SESSION}/events"),
    ("GET", f"/api/jobs/{OWN_JOB}"),
]

OTHERS_RECORDS = [
    ("GET", "/api/schedules/other"),
    ("GET", "/api/schedules/other/sessions"),
    ("GET", f"/api/sessions/{OTHER_SESSION}"),
    ("GET", f"/api/sessions/{OTHER_SESSION}/events"),
    ("GET", "/api/sessions/no-such-session"),
    ("GET", f"/api/jobs/{OTHER_JOB}"),
    ("GET", f"/api/jobs/{OTHER_JOB_UNKNOWN_SESSION}"),
    ("GET", "/api/jobs/no-such-job"),
]

FORBIDDEN = [
    ("GET", "/api/jobs"),
    ("POST", "/api/jobs"),
    ("GET", "/api/schedules"),
    ("POST", "/api/schedules"),
    ("POST", f"/api/schedules/{OWN_SCHEDULE}/run"),
    ("PATCH", f"/api/schedules/{OWN_SCHEDULE}"),
    ("DELETE", f"/api/schedules/{OWN_SCHEDULE}"),
    ("POST", f"/api/schedules/{OWN_SCHEDULE}/enable"),
    ("POST", f"/api/schedules/{OWN_SCHEDULE}/disable"),
    ("POST", "/api/schedules/cleanup"),
    ("GET", "/api/sessions"),
    ("GET", f"/api/sessions/{OWN_SESSION}/metadata"),
    ("GET", f"/api/sessions/{OWN_SESSION}/settings"),
    ("POST", f"/api/jobs/{OWN_JOB}/cancel"),
    ("POST", f"/api/jobs/{OWN_JOB}/retry"),
    ("POST", f"/api/jobs/{OWN_JOB}/mark-done"),
    ("POST", "/api/sessions"),
    ("PATCH", f"/api/sessions/{OWN_SESSION}"),
    ("POST", f"/api/sessions/{OWN_SESSION}/cancel"),
    ("POST", "/api/daemon/reload-config"),
    ("POST", "/api/secrets/anything"),
    ("GET", "/api/secrets"),
    ("POST", "/api/chat"),
]


@pytest.fixture
def token_store(tmp_path):
    return TokenStore(tmp_path / "tokens.json")


@pytest.fixture
def client(tmp_path, token_store):
    from tsugite.workspace import WorkspaceNotFoundError

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with patch("tsugite.workspace.Workspace") as ws_cls:
        ws_cls.load.side_effect = WorkspaceNotFoundError("not found")
        adapter = HTTPAgentAdapter(
            runtime=RuntimeDefaults(workspace_dir=workspace, agent_file="default"),
            session_store=SessionStore(tmp_path / "session_store.json"),
        )
    server = HTTPServer(
        config=HTTPConfig(enabled=True, host="127.0.0.1", port=8486),
        adapter=adapter,
        webhook_store=WebhookStore(tmp_path / "webhooks.json"),
        token_store=token_store,
    )
    store = adapter.session_store
    for sid, parent in ((OWN_SESSION, OWN_SCHEDULE), (OTHER_SESSION, "other")):
        store.create_session(Session(id=sid, source=SessionSource.SCHEDULE.value, user_id="u", parent_id=parent))
    jobs = {
        job_id: SimpleNamespace(parent_session_id=session_id, to_payload=lambda: {})
        for job_id, session_id in (
            (OWN_JOB, OWN_SESSION),
            (OTHER_JOB, OTHER_SESSION),
            (OTHER_JOB_UNKNOWN_SESSION, "gone"),
        )
    }
    server.session_runner = SessionRunner(store=store, adapter=None)
    server.jobs_orchestrator = SimpleNamespace(get_job=jobs.get)
    return TestClient(server.app)


def _call(client, method, path, token):
    return client.request(method, path, headers={"Authorization": f"Bearer {token}"}, json={})


@pytest.mark.parametrize("method,path", ALLOWED)
def test_run_token_reaches_its_own_records(client, token_store, method, path):
    token = token_store.issue(schedule_id=OWN_SCHEDULE, scope=SCOPE_SCRIPT_RUN)
    assert _call(client, method, path, token).status_code not in (401, 403)


@pytest.mark.parametrize("method,path", OTHERS_RECORDS)
def test_run_token_is_refused_other_schedules_records(client, token_store, method, path):
    token = token_store.issue(schedule_id=OWN_SCHEDULE, scope=SCOPE_SCRIPT_RUN)
    assert _call(client, method, path, token).status_code == 403


@pytest.mark.parametrize("method,path", FORBIDDEN)
def test_run_token_is_refused_everywhere_else(client, token_store, method, path):
    token = token_store.issue(schedule_id=OWN_SCHEDULE, scope=SCOPE_SCRIPT_RUN)
    assert _call(client, method, path, token).status_code == 403


def test_unscoped_tokens_keep_full_access(client, token_store):
    _t, admin = token_store.create_admin_token(name="a")
    agent = token_store.issue(schedule_id="s")
    for token in (admin, agent):
        assert _call(client, "POST", "/api/daemon/reload-config", token).status_code != 403


def _scheduler(tmp_path, token_store):
    store = SessionStore(tmp_path / "session_store.json")
    adapter = AsyncMock()
    adapter.agent_name = "bot"
    adapter.session_store = store
    adapter.event_bus = None
    sa = SchedulerAdapter(
        adapter=adapter,
        schedules_path=tmp_path / "schedules.json",
        token_store=token_store,
        tsugite_api_url="http://127.0.0.1:8486",
    )
    sa.set_session_runner(SessionRunner(store=store, adapter=None))
    return sa


def _script(command, **kw):
    return ScheduleEntry(
        id="job1",
        prompt="",
        schedule_type="cron",
        cron_expr="*/5 * * * *",
        execution_type="script",
        command=command,
        **kw,
    )


class _SpyStore(TokenStore):
    issued: list

    def issue(self, *a, **kw):
        raw = super().issue(*a, **kw)
        self.issued = [*getattr(self, "issued", []), (raw, kw)]
        return raw


@pytest.mark.asyncio
async def test_opted_in_script_gets_url_and_scoped_token_then_loses_it(tmp_path, history_dir):
    store = _SpyStore(tmp_path / "tokens.json")
    sa = _scheduler(tmp_path, store)
    out = tmp_path / "env.txt"

    await sa._run_script(_script(f'echo "$TSUGITE_DAEMON_URL $TSUGITE_DAEMON_TOKEN" > {out}', daemon_token=True))

    url, token = out.read_text().split()
    assert url == "http://127.0.0.1:8486"
    assert [r for r, _ in store.issued] == [token]
    assert store.issued[0][1]["scope"] == SCOPE_SCRIPT_RUN
    assert store.resolve(token) is None


@pytest.mark.asyncio
async def test_token_is_revoked_when_the_script_times_out(tmp_path, history_dir):
    store = _SpyStore(tmp_path / "tokens.json")
    sa = _scheduler(tmp_path, store)

    with pytest.raises(RuntimeError, match="timed out"):
        await sa._run_script(_script("sleep 5", daemon_token=True, script_timeout=1))

    assert store.resolve(store.issued[0][0]) is None


@pytest.mark.asyncio
async def test_token_is_off_by_default(tmp_path, history_dir, monkeypatch):
    monkeypatch.delenv("TSUGITE_DAEMON_TOKEN", raising=False)
    monkeypatch.delenv("TSUGITE_DAEMON_URL", raising=False)
    store = _SpyStore(tmp_path / "tokens.json")
    sa = _scheduler(tmp_path, store)
    out = tmp_path / "env.txt"

    await sa._run_script(_script(f'echo "[$TSUGITE_DAEMON_URL][$TSUGITE_DAEMON_TOKEN]" > {out}'))

    assert out.read_text().strip() == "[][]"
    assert not hasattr(store, "issued")


@pytest.mark.asyncio
async def test_opt_in_without_a_token_store_fails(tmp_path, history_dir):
    sa = _scheduler(tmp_path, None)

    with pytest.raises(RuntimeError, match="no token store"):
        await sa._run_script(_script("true", daemon_token=True))
