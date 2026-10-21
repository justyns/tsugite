"""Agents: identities with their own workspace that sessions run under."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from starlette.testclient import TestClient
from tsugite_daemon.adapters.base import BaseAdapter, ChannelContext
from tsugite_daemon.adapters.http import HTTPAgentAdapter, HTTPServer
from tsugite_daemon.agent_store import Agent, AgentStore
from tsugite_daemon.commands import CommandError, cmd_bg
from tsugite_daemon.config import HTTPConfig, RuntimeDefaults
from tsugite_daemon.session_runner import SessionRunner, _current_session_id
from tsugite_daemon.session_store import Session, SessionSource, SessionStore
from tsugite_daemon.webhook_store import WebhookStore

from tsugite.agent_runner.runner import _resolve_workspace_dir
from tsugite.cli.helpers import PathContext
from tsugite.tools import sessions as sessions_tools


class _StubAdapter(BaseAdapter):
    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass


def _workspace(path, *agent_names):
    path.mkdir()
    for name in agent_names:
        (path / f"{name}.md").write_text(f"---\nname: {name}\n---\n\nHi.\n")
    return path


class TestAgentStore:
    def test_agents_persist_across_reload(self, tmp_path):
        store = AgentStore(tmp_path / "store.json")
        store.create(Agent(id="homelab", name="Homelab", workspace="/ws/homelab", template="default", model="m"))
        store.update("homelab", color="teal")

        reloaded = AgentStore(tmp_path / "store.json").get("homelab")

        assert reloaded.name == "Homelab"
        assert reloaded.model == "m"
        assert reloaded.color == "teal"

    def test_update_swaps_in_a_new_agent(self, tmp_path):
        store = AgentStore(tmp_path / "store.json")
        before = store.create(Agent(id="a", name="A", workspace="/w", template="t"))

        after = store.update("a", name="B")

        assert before.name == "A"
        assert store.get("a") is after
        assert after.name == "B"

    def test_rejects_a_color_outside_the_palette(self, tmp_path):
        store = AgentStore(tmp_path / "store.json")
        with pytest.raises(ValueError, match="color"):
            store.create(Agent(id="a", name="A", workspace="/w", template="t", color="#123456"))
        store.create(Agent(id="a", name="A", workspace="/w", template="t", color="green"))
        with pytest.raises(ValueError, match="color"):
            store.update("a", color="magenta")
        assert store.get("a").color == "green"

    @pytest.mark.parametrize("agent_id", ["Homelab", "", "has space", "-lead"])
    def test_create_rejects_a_non_slug_id(self, tmp_path, agent_id):
        with pytest.raises(ValueError, match="Invalid agent id"):
            AgentStore(tmp_path / "store.json").create(Agent(id=agent_id, name="x", workspace="/w", template="t"))

    def test_create_rejects_a_duplicate_id(self, tmp_path):
        store = AgentStore(tmp_path / "store.json")
        store.create(Agent(id="a", name="A", workspace="/w", template="t"))
        with pytest.raises(ValueError, match="already exists"):
            store.create(Agent(id="a", name="A2", workspace="/w", template="t"))

    def test_update_rejects_id_and_default(self, tmp_path):
        store = AgentStore(tmp_path / "store.json")
        store.create(Agent(id="a", name="A", workspace="/w", template="t"))
        with pytest.raises(ValueError, match="Unknown agent field"):
            store.update("a", is_default=True)

    def test_ensure_default_creates_one_agent_from_the_runtime(self, tmp_path):
        store = AgentStore(tmp_path / "store.json")
        ws = tmp_path / "My Workspace"

        store.ensure_default(ws, "default")
        first = store.default()

        assert (first.id, first.name, first.workspace, first.template, first.color) == (
            "my-workspace",
            "My Workspace",
            str(ws),
            "default",
            "blue",
        )
        assert first.is_default
        assert store.for_session(None) is first

    def test_ensure_default_syncs_the_runtime_workspace_and_template(self, tmp_path):
        AgentStore(tmp_path / "store.json").ensure_default(tmp_path / "atlas", "default")

        store = AgentStore(tmp_path / "store.json")
        store.ensure_default(tmp_path / "moved", "other")

        assert [(a.id, a.workspace, a.template) for a in store.list()] == [("atlas", str(tmp_path / "moved"), "other")]
        assert AgentStore(tmp_path / "store.json").default().workspace == str(tmp_path / "moved")

    @pytest.mark.parametrize("dirname", ["_scratch", "w" * 70])
    def test_ensure_default_uses_default_id_for_a_non_slug_workspace_name(self, tmp_path, dirname):
        store = AgentStore(tmp_path / "store.json")

        store.ensure_default(tmp_path / dirname, "default")

        assert store.default().id == "default"


@pytest.fixture
def default_ws(tmp_path):
    return _workspace(tmp_path / "main", "default", "pinned")


@pytest.fixture
def homelab_ws(tmp_path):
    return _workspace(tmp_path / "homelab", "ops", "triage")


@pytest.fixture
def store(tmp_path, default_ws, homelab_ws):
    store = SessionStore(tmp_path / "state" / "store.json")
    store.agents.ensure_default(default_ws, "default")
    store.agents.create(
        Agent(id="homelab", name="Homelab", workspace=str(homelab_ws), template="ops", model="anthropic:claude-x")
    )
    return store


@pytest.fixture
def adapter(store, default_ws, monkeypatch):
    adapter = _StubAdapter(
        RuntimeDefaults(workspace_dir=default_ws, agent_file="default", model="openai:gpt-4o"), store
    )
    monkeypatch.setattr(adapter, "_build_message_context", lambda message, *a, **kw: message)
    monkeypatch.setattr(adapter, "_build_agent_context", lambda *a, **kw: {})
    monkeypatch.setattr(adapter, "_update_skill_ttl", lambda *a, **kw: None)
    return adapter


@pytest.fixture
def run_calls(monkeypatch):
    calls = []

    def fake_run_agent(**kwargs):
        calls.append({**kwargs, "executor_dir": _resolve_workspace_dir(None, kwargs["path_context"])})
        result = MagicMock()
        result.__str__ = lambda self: "ok"
        result.token_count = None
        return result

    monkeypatch.setattr("tsugite_daemon.adapters.base.run_agent", fake_run_agent)
    return calls


async def _turn(adapter, session_id, **metadata):
    await adapter.handle_message(
        "alice",
        "hi",
        ChannelContext(
            source="http",
            channel_id=None,
            user_id="alice",
            reply_to="http:alice",
            metadata={"conv_id_override": session_id, **metadata},
        ),
    )


class TestTurnResolution:
    @pytest.mark.asyncio
    async def test_turn_runs_the_agent_template_in_its_workspace(self, adapter, run_calls, homelab_ws):
        session = adapter.session_store.create_session(Session(id="", user_id="alice", agent_id="homelab"))

        await _turn(adapter, session.id)

        call = run_calls[-1]
        assert call["agent_path"] == homelab_ws / "ops.md"
        assert call["path_context"].workspace_dir == homelab_ws
        assert call["path_context"].effective_cwd == homelab_ws
        assert call["exec_options"].model_override == "anthropic:claude-x"

    @pytest.mark.asyncio
    async def test_session_without_agent_runs_the_default_agent(self, adapter, run_calls, store, default_ws):
        store.agents.update(store.agents.default().id, model="openai:default-agent-model", reasoning_effort="high")
        session = store.create_session(Session(id="", user_id="alice"))

        await _turn(adapter, session.id)

        call = run_calls[-1]
        assert call["agent_path"] == default_ws / "default.md"
        assert call["path_context"].workspace_dir == default_ws
        assert call["exec_options"].model_override == "openai:default-agent-model"
        assert call["exec_options"].reasoning_effort_override == "high"

    @pytest.mark.asyncio
    async def test_session_settings_beat_the_agent(self, adapter, run_calls, store, homelab_ws):
        session = store.create_session(
            Session(id="", user_id="alice", agent_id="homelab", agent_file="triage", model_override="openai:mine")
        )

        await _turn(adapter, session.id)

        call = run_calls[-1]
        assert call["exec_options"].model_override == "openai:mine"
        assert call["agent_path"] == homelab_ws / "triage.md"

    @pytest.mark.asyncio
    async def test_working_directory_override_keeps_the_identity_workspace(
        self, adapter, run_calls, store, homelab_ws, tmp_path
    ):
        worktree = tmp_path / "worktree"
        worktree.mkdir()
        session = store.create_session(
            Session(id="", user_id="alice", agent_id="homelab", workspace_override=str(worktree))
        )

        await _turn(adapter, session.id)

        ctx = run_calls[-1]["path_context"]
        assert ctx.workspace_dir == homelab_ws
        assert ctx.effective_cwd == worktree
        assert ctx.invoked_from == worktree
        assert run_calls[-1]["executor_dir"] == worktree

    @pytest.mark.asyncio
    async def test_working_directory_context_files_are_attached(self, adapter, run_calls, store, homelab_ws, tmp_path):
        (homelab_ws / "AGENTS.md").write_text("identity rules")
        worktree = tmp_path / "worktree"
        worktree.mkdir()
        (worktree / "AGENTS.md").write_text("repo rules")
        (worktree / "CLAUDE.md").write_text("claude rules")
        (worktree / "README.md").write_text("readme")
        session = store.create_session(
            Session(id="", user_id="alice", agent_id="homelab", workspace_override=str(worktree))
        )

        await _turn(adapter, session.id)

        assert [(a.name, a.content) for a in run_calls[-1]["attachments"]] == [
            (str(worktree / "AGENTS.md"), "repo rules"),
            (str(worktree / "CLAUDE.md"), "claude rules"),
        ]

    @pytest.mark.asyncio
    async def test_unreadable_working_directory_context_file_is_skipped(
        self, adapter, run_calls, store, tmp_path, caplog
    ):
        worktree = tmp_path / "worktree"
        worktree.mkdir()
        (worktree / "AGENTS.md").write_bytes(b"\xff\xfe not utf-8 \x80")
        session = store.create_session(
            Session(id="", user_id="alice", agent_id="homelab", workspace_override=str(worktree))
        )

        await _turn(adapter, session.id)

        assert run_calls[-1]["attachments"] == []
        assert "AGENTS.md" in caplog.text

    @pytest.mark.asyncio
    async def test_identity_workspace_agents_md_is_not_attached_twice(self, adapter, run_calls, store, homelab_ws):
        (homelab_ws / "AGENTS.md").write_text("identity rules")
        session = store.create_session(Session(id="", user_id="alice", agent_id="homelab"))

        await _turn(adapter, session.id)

        assert run_calls[-1]["attachments"] == []


class TestUnknownAgent:
    @pytest.mark.asyncio
    async def test_turn_fails_for_a_session_whose_agent_is_gone(self, adapter, run_calls, store):
        session = store.create_session(Session(id="", user_id="alice", agent_id="ghost"))

        with pytest.raises(ValueError, match=f"Unknown agent 'ghost' for session {session.id}"):
            await _turn(adapter, session.id)
        assert run_calls == []


def _pinned_session(store, **fields):
    return store.create_session(
        Session(id="", source=SessionSource.INTERACTIVE.value, user_id="alice", agent_file="pinned", **fields)
    )


class TestSessionAgentFile:
    @pytest.mark.asyncio
    async def test_chat_turn_runs_the_session_agent_file(self, adapter, run_calls, store, default_ws):
        session = _pinned_session(store)

        await _turn(adapter, session.id)

        assert run_calls[-1]["agent_path"] == default_ws / "pinned.md"

    @pytest.mark.asyncio
    async def test_per_turn_override_beats_the_session_agent_file(self, adapter, run_calls, store, default_ws):
        session = _pinned_session(store)

        await _turn(adapter, session.id, agent_file_override=str(default_ws / "default.md"))

        assert run_calls[-1]["agent_path"] == default_ws / "default.md"

    @pytest.mark.asyncio
    async def test_reply_to_session_runs_the_session_agent_file(self, adapter, run_calls, store, default_ws):
        session = _pinned_session(store)

        await SessionRunner(store, adapter).reply_to_session(session.id, "follow up")

        assert run_calls[-1]["agent_path"] == default_ws / "pinned.md"

    @pytest.mark.parametrize("rotate", ["compact", "branch"])
    def test_compaction_and_branch_keep_the_agent_and_agent_file(self, store, rotate, monkeypatch):
        session = _pinned_session(store, agent_id="homelab")
        if rotate == "compact":
            successor = store.compact_session(session.id)
        else:
            monkeypatch.setattr(
                "tsugite_daemon.session_store.get_history_backend",
                lambda: MagicMock(create_branch=lambda *a, **kw: "branch-1"),
            )
            successor = store.branch_session(session.id, at_event_id=1)

        assert (successor.agent_id, successor.agent_file) == ("homelab", "pinned")


@pytest.fixture
def runner(store, adapter):
    runner = SessionRunner(store, adapter)
    runner._run_session = AsyncMock()
    return runner


class TestChildSessionsInheritTheAgent:
    @pytest.mark.asyncio
    async def test_start_session_copies_the_parent_agent(self, store, runner):
        parent = store.create_session(Session(id="", user_id="alice", agent_id="homelab"))

        child = runner.start_session(Session(id="", prompt="go"), inherit_from=parent.id)
        orphan = runner.start_session(Session(id="", prompt="go"))

        assert (child.agent_id, orphan.agent_id) == ("homelab", None)
        with pytest.raises(ValueError, match="ghost"):
            runner.start_session(Session(id="", prompt="go"), inherit_from="ghost")

    def test_start_session_tool_inherits_from_the_current_session(self, store, runner):
        parent = store.create_session(Session(id="", user_id="alice", agent_id="homelab"))
        calls = []
        runner.start_session = lambda session, **kw: calls.append(kw) or session
        token = _current_session_id.set(parent.id)
        try:
            with (
                patch.object(sessions_tools, "_session_runner", runner),
                patch.object(sessions_tools, "_call", lambda fn, *a, **kw: fn(*a, **kw)),
            ):
                sessions_tools.start_session(prompt="go")
        finally:
            _current_session_id.reset(token)

        assert calls == [{"inherit_from": parent.id}]

    @pytest.mark.asyncio
    async def test_bg_command(self, store, adapter, runner, monkeypatch):
        parent = store.create_session(Session(id="", user_id="alice", agent_id="homelab"))
        monkeypatch.setattr(sessions_tools, "_session_runner", runner)

        result = await cmd_bg(adapter, prompt="go", session_id=parent.id)

        child_id = result.rsplit(" ", 1)[-1].rstrip(")")
        assert store.get_session(child_id).agent_id == "homelab"

    @pytest.mark.asyncio
    async def test_bg_command_rejects_an_unknown_session(self, adapter, runner, monkeypatch):
        monkeypatch.setattr(sessions_tools, "_session_runner", runner)

        with pytest.raises(CommandError, match="ghost"):
            await cmd_bg(adapter, prompt="go", session_id="ghost")

    @pytest.mark.asyncio
    async def test_bg_command_reports_other_start_errors(self, adapter, runner, monkeypatch):
        monkeypatch.setattr(sessions_tools, "_session_runner", runner)
        monkeypatch.setattr(runner, "start_session", MagicMock(side_effect=ValueError("disk full")))

        result = await cmd_bg(adapter, prompt="go")

        assert result == "Failed to start background session: disk full"


def test_cli_executor_runs_in_the_workspace_not_the_root(tmp_path):
    ctx = PathContext(invoked_from=tmp_path, workspace_dir=tmp_path / "workspace", effective_cwd=tmp_path / "root")

    assert _resolve_workspace_dir(None, ctx) == tmp_path / "workspace"


@pytest.fixture
def server(store, default_ws, tmp_path):
    from tsugite_daemon.auth import TokenStore

    with patch("tsugite.workspace.Workspace") as mock_ws_cls:
        from tsugite.workspace import WorkspaceNotFoundError

        mock_ws_cls.load.side_effect = WorkspaceNotFoundError("not found")
        mock_ws_cls.try_load.return_value = None
        adapter = HTTPAgentAdapter(
            runtime=RuntimeDefaults(workspace_dir=default_ws, agent_file="default"), session_store=store
        )
    srv = HTTPServer(
        config=HTTPConfig(enabled=True, host="127.0.0.1", port=8378),
        adapter=adapter,
        webhook_store=WebhookStore(tmp_path / "webhooks.json"),
        token_store=TokenStore(tmp_path / "tokens.json"),
    )
    srv.session_runner = SessionRunner(store=store, adapter=None)
    return srv


@pytest.fixture
def client(server):
    return TestClient(server.app)


@pytest.fixture
def auth(server):
    _st, raw = server._token_store.create_admin_token(name="test-token")
    return {"Authorization": f"Bearer {raw}"}


class TestAgentsApi:
    @pytest.mark.parametrize(
        "method,path",
        [
            ("GET", "/api/agents"),
            ("POST", "/api/agents"),
            ("GET", "/api/agents/homelab"),
            ("PATCH", "/api/agents/homelab"),
        ],
    )
    def test_requires_auth(self, client, method, path):
        assert client.request(method, path, json={}).status_code == 401

    def test_list_shows_the_default_agent(self, client, auth, default_ws):
        body = client.get("/api/agents", headers=auth).json()

        assert {a["id"]: a["is_default"] for a in body["agents"]} == {"main": True, "homelab": False}

    def test_get_unknown_agent_is_404(self, client, auth):
        assert client.get("/api/agents/nope", headers=auth).status_code == 404

    def test_create_then_patch(self, client, auth, homelab_ws):
        created = client.post(
            "/api/agents",
            headers=auth,
            json={"id": "research", "name": "Research", "workspace": str(homelab_ws), "template": "ops"},
        )
        assert created.status_code == 201
        assert created.json()["is_default"] is False

        patched = client.patch("/api/agents/research", headers=auth, json={"model": "openai:gpt-5", "color": "green"})
        assert patched.status_code == 200
        assert client.get("/api/agents/research", headers=auth).json()["model"] == "openai:gpt-5"

        cleared = client.patch("/api/agents/research", headers=auth, json={"model": None, "color": None})
        assert (cleared.status_code, cleared.json()["model"], cleared.json()["color"]) == (200, None, None)

    def test_create_expands_and_stores_the_resolved_workspace(self, client, auth, homelab_ws, monkeypatch):
        monkeypatch.setenv("HOME", str(homelab_ws.parent))

        resp = client.post(
            "/api/agents", headers=auth, json={"id": "research", "workspace": "~/homelab/.", "template": "ops"}
        )

        assert resp.status_code == 201
        assert resp.json()["workspace"] == str(homelab_ws.resolve())

    @pytest.mark.parametrize(
        "body,error",
        [
            ({"id": "x", "name": "X", "workspace": "/does/not/exist", "template": "ops"}, "workspace"),
            ({"id": "x", "name": "X", "workspace": "HOMELAB", "template": "missing"}, "template"),
            ({"id": "homelab", "name": "X", "workspace": "HOMELAB", "template": "ops"}, "already exists"),
            ({"name": "X", "workspace": "HOMELAB", "template": "ops"}, "id"),
            ({"id": "x", "workspace": "homelab", "template": "ops"}, "absolute"),
            ({"id": 5, "workspace": "HOMELAB", "template": "ops"}, "id"),
            ({"id": "x", "name": ["X"], "workspace": "HOMELAB", "template": "ops"}, "name"),
            ({"id": "x", "workspace": "HOMELAB", "template": "ops", "color": "#ff0000"}, "color"),
        ],
    )
    def test_create_rejects_bad_input(self, client, auth, homelab_ws, body, error, monkeypatch):
        monkeypatch.chdir(homelab_ws.parent)
        body = {k: (str(homelab_ws) if v == "HOMELAB" else v) for k, v in body.items()}

        resp = client.post("/api/agents", headers=auth, json=body)

        assert resp.status_code == 400
        assert error in resp.json()["error"]

    @pytest.mark.parametrize("method,path", [("POST", "/api/agents"), ("PATCH", "/api/agents/homelab")])
    def test_rejects_unknown_fields(self, client, auth, homelab_ws, method, path):
        body = {"id": "x", "workspace": str(homelab_ws), "template": "ops", "is_default": True}

        resp = client.request(method, path, headers=auth, json=body if method == "POST" else {"is_default": True})

        assert resp.status_code == 400
        assert "Unknown agent field" in resp.json()["error"]

    @pytest.mark.parametrize("workspace", ["/", "STATE", "STATE/sub"])
    def test_rejects_the_root_and_state_dir_as_workspace(self, client, auth, tmp_path, workspace):
        (tmp_path / "state" / "sub").mkdir(parents=True, exist_ok=True)
        workspace = workspace.replace("STATE", str(tmp_path / "state"))

        resp = client.post("/api/agents", headers=auth, json={"id": "x", "workspace": workspace, "template": "ops"})

        assert resp.status_code == 400
        assert "workspace" in resp.json()["error"]

    @pytest.mark.parametrize("field", ["workspace", "template"])
    def test_default_agent_workspace_and_template_come_from_the_config(self, client, auth, homelab_ws, field):
        value = str(homelab_ws) if field == "workspace" else "pinned"

        resp = client.patch("/api/agents/main", headers=auth, json={field: value})

        assert resp.status_code == 400
        assert "daemon config" in resp.json()["error"]
        assert client.patch("/api/agents/main", headers=auth, json={"name": "Main"}).status_code == 200

    @pytest.mark.parametrize(
        "body,error",
        [
            ({"workspace": None}, "workspace"),
            ({"workspace": "/does/not/exist"}, "workspace"),
            ({"workspace": "MAIN"}, "template"),
            ({"color": "magenta"}, "color"),
            ({"name": None}, "name"),
        ],
    )
    def test_patch_rejects_bad_input(self, client, auth, default_ws, body, error):
        body = {k: (str(default_ws) if v == "MAIN" else v) for k, v in body.items()}

        resp = client.patch("/api/agents/homelab", headers=auth, json=body)

        assert resp.status_code == 400
        assert error in resp.json()["error"]

    def test_patch_unknown_agent_is_404(self, client, auth):
        assert client.patch("/api/agents/nope", headers=auth, json={"name": "X"}).status_code == 404

    def test_new_chat_takes_an_agent_and_lists_it(self, client, auth, store):
        resp = client.post("/api/chat/sessions/new", headers=auth, json={"agent_id": "homelab"})
        assert resp.status_code == 201
        session_id = resp.json()["id"]

        assert store.get_session(session_id).agent_id == "homelab"
        rows = client.get("/api/chat/sessions", headers=auth).json()["sessions"]
        assert next(r for r in rows if r["id"] == session_id)["agent_id"] == "homelab"
        assert client.get(f"/api/sessions/{session_id}", headers=auth).json()["agent_id"] == "homelab"

    @pytest.mark.parametrize("path", ["/api/chat/sessions/new", "/api/sessions"])
    @pytest.mark.parametrize("agent_id", ["nope", ["homelab"], {"id": "homelab"}, 3])
    def test_new_session_rejects_a_bad_agent_id(self, client, auth, path, agent_id):
        resp = client.post(path, headers=auth, json={"prompt": "go", "agent_id": agent_id})
        assert resp.status_code == 400

    def test_background_session_takes_an_agent(self, client, auth, server, store):
        server.session_runner.start_session = lambda session: store.create_session(session)

        resp = client.post("/api/sessions", headers=auth, json={"prompt": "go", "agent_id": "homelab"})

        assert resp.status_code == 201
        assert store.get_session(resp.json()["id"]).agent_id == "homelab"
        rows = client.get("/api/sessions", headers=auth).json()["sessions"]
        assert next(r for r in rows if r["id"] == resp.json()["id"])["agent_id"] == "homelab"

    def test_session_payloads_resolve_a_missing_agent_to_the_default(self, client, auth, store):
        session = store.create_session(Session(id="", user_id="web-anonymous"))

        rows = client.get("/api/chat/sessions", headers=auth).json()["sessions"]
        assert next(r for r in rows if r["id"] == session.id)["agent_id"] == "main"
        assert client.get(f"/api/sessions/{session.id}", headers=auth).json()["agent_id"] == "main"


class TestOrphanSessionDisplay:
    """A session whose agent is gone still lists; only its turns fail."""

    @pytest.fixture
    def orphan(self, store):
        return store.create_session(Session(id="", user_id="web-anonymous", agent_id="ghost"))

    def test_session_lists_still_load(self, client, auth, orphan):
        for path in ("/api/chat/sessions", "/api/sessions"):
            resp = client.get(path, headers=auth)
            assert resp.status_code == 200
            assert next(r for r in resp.json()["sessions"] if r["id"] == orphan.id)["agent_id"] == "ghost"
        assert client.get(f"/api/sessions/{orphan.id}", headers=auth).json()["agent_id"] == "ghost"

    def test_model_display_is_unknown(self, client, auth, orphan, server):
        resp = client.get("/api/chat/effort-levels", headers=auth, params={"session_id": orphan.id})

        assert resp.status_code == 200
        assert resp.json()["model"] == "unknown"
        assert server.adapter.resolve_session_model(orphan.id) == "unknown"

    def test_workspace_files_reject_the_session(self, client, auth, orphan):
        resp = client.get("/api/workspace/content", headers=auth, params={"path": "x.md", "session_id": orphan.id})

        assert resp.status_code == 400
        assert "ghost" in resp.json()["error"]

    @pytest.mark.parametrize(
        "method,path",
        [
            ("GET", "/api/context-providers/x/choices"),
            ("GET", "/api/context-providers/x/search"),
            ("POST", "/api/context-providers/x/capture"),
        ],
    )
    def test_context_providers_reject_the_session(self, client, auth, orphan, method, path):
        if method == "GET":
            resp = client.get(path, headers=auth, params={"session_id": orphan.id})
        else:
            resp = client.post(path, headers=auth, json={"session_id": orphan.id})

        assert resp.status_code == 400
        assert "ghost" in resp.json()["error"]
