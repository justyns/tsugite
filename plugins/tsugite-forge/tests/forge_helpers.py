"""Fakes for the forge plugin's tests: a secrets backend, a Forgejo server and a `gh` executable."""

import json
import os
import sys
from pathlib import Path

import httpx

FIXTURES = Path(__file__).parent / "fixtures"
TOKEN = "tok-SECRETVALUE-123"
SECRET_NAME = "forgejo-token"
FORGEJO_URL = "https://forge.example.test"


class Clock:
    """A clock tests move by hand."""

    def __init__(self, now=1000.0):
        self.now = now

    def __call__(self):
        return self.now


class StubSecretBackend:
    def get(self, name: str) -> str | None:
        return TOKEN if name == SECRET_NAME else None


def forgejo_config(**overrides):
    from tsugite_forge.config import ForgejoForge

    fields = {"name": "home", "kind": "forgejo", "url": FORGEJO_URL, "token_secret": SECRET_NAME}
    return ForgejoForge(**{**fields, "repos": ["octo-org/demo"], **overrides})


def github_config(**overrides):
    from tsugite_forge.config import GithubForge

    return GithubForge(**{"name": "gh", "kind": "github", "repos": ["octo-org/demo"], **overrides})


class FakeForgejo:
    """Answers from fixtures/forgejo.json, keyed by path and query, and records each request.

    A route value is a JSON body, a string body, `{"__status": n, ...}` for an error,
    or a callable taking the request. A page past the first with no route is empty.
    """

    def __init__(self, overrides: dict | None = None):
        self.routes = {**json.loads((FIXTURES / "forgejo.json").read_text()), **(overrides or {})}
        self.requests: list[httpx.Request] = []
        self.transport = httpx.MockTransport(self._handle)

    def _handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        route = self.routes.get(request.url.raw_path.decode())
        if route is None and request.url.params.get("page", "1") != "1":
            return httpx.Response(200, json=[])
        if route is None:
            return httpx.Response(404, json={"message": "not found"})
        if callable(route):
            return route(request)
        if isinstance(route, str):
            return httpx.Response(200, text=route)
        if isinstance(route, dict) and "__status" in route:
            return httpx.Response(route["__status"], json={k: v for k, v in route.items() if k != "__status"})
        return httpx.Response(200, json=route)


FAKE_GH = """#!{python}
import json, os, sys

argv = sys.argv[1:]
with open(os.environ["FAKE_GH_LOG"], "a") as log:
    log.write(json.dumps(argv) + "\\n")
joined = " ".join(argv)
with open(os.environ["FAKE_GH_RESPONSES"]) as responses:
    for entry in json.load(responses):
        if all(part in joined for part in entry["match"]):
            out = entry.get("stdout", "")
            sys.stdout.write(out if isinstance(out, str) else json.dumps(out))
            sys.stderr.write(entry.get("stderr", ""))
            sys.exit(entry.get("exit", 0))
sys.stderr.write("fake gh: no fixture for " + joined + "\\n")
sys.exit(1)
"""


class FakeGh:
    """A `gh` on PATH that replays fixtures/github.json, or `responses`, matched by argv substrings."""

    def __init__(self, tmp_path: Path, monkeypatch, responses: list | None = None):
        bin_dir = tmp_path / "fake-gh"
        bin_dir.mkdir()
        gh = bin_dir / "gh"
        gh.write_text(FAKE_GH.format(python=sys.executable))
        gh.chmod(0o755)
        responses_file = tmp_path / "gh-responses.json"
        if responses is None:
            responses = json.loads((FIXTURES / "github.json").read_text())
        responses_file.write_text(json.dumps(responses))
        self.log = tmp_path / "gh-calls.jsonl"
        monkeypatch.setenv("FAKE_GH_RESPONSES", str(responses_file))
        monkeypatch.setenv("FAKE_GH_LOG", str(self.log))
        monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")

    @property
    def calls(self) -> list[list[str]]:
        if not self.log.exists():
            return []
        return [json.loads(line) for line in self.log.read_text().splitlines()]


def make_pr(**overrides):
    from tsugite_forge.providers import PullRequest

    fields = {
        "forge": "home",
        "repo": "octo-org/demo",
        "number": 7,
        "title": "Add retry to the sync job",
        "author": "octocat",
        "state": "open",
        "draft": False,
        "head_ref": "retry-sync",
        "head_sha": "7" * 40,
        "base_ref": "main",
        "base_sha": "1" * 40,
        "labels": ["bug", "sync"],
        "requested_reviewers": ["reviewer1"],
        "ci_state": "failure",
        "unresolved_threads": 1,
        "review_state": "changes_requested",
        "created_at": "2026-10-01T12:00:00Z",
        "updated_at": "2026-10-09T12:00:00Z",
        "url": "https://forge.example.test/octo-org/demo/pulls/7",
    }
    return PullRequest(**{**fields, **overrides})


FORGES = {
    "home": {
        "name": "home",
        "kind": "forgejo",
        "url": FORGEJO_URL,
        "token_secret": SECRET_NAME,
        "repos": ["octo-org/*"],
    },
    "gh": {"name": "gh", "kind": "github", "repos": ["octo-org/demo"]},
}


def make_adapter(clock, session_store=None, forges=("home", "gh"), poll_minutes=5):
    """An adapter over `forges`: forgejo `home` and github `gh`, which both list octo-org/demo."""
    from tsugite_forge.adapter import ForgeAdapter
    from tsugite_forge.config import ForgePluginConfig

    config = ForgePluginConfig(poll_minutes=poll_minutes, forges=[FORGES[name] for name in forges])
    return ForgeAdapter(config, clock=clock, session_store=session_store)


def forge_http(adapter, tmp_path, surfaces=()):
    """A test client for a daemon HTTP server with the forge routes mounted, its token store and the server."""
    from starlette.testclient import TestClient
    from tsugite_daemon.adapters.http import HTTPServer
    from tsugite_daemon.auth import TokenStore
    from tsugite_daemon.config import HTTPConfig
    from tsugite_daemon.plugin_wiring import attach_plugin_http
    from tsugite_daemon.webhook_store import WebhookStore

    token_store = TokenStore(tmp_path / "tokens.json")
    server = HTTPServer(
        config=HTTPConfig(enabled=True, host="127.0.0.1", port=8374),
        adapter=None,
        webhook_store=WebhookStore(tmp_path / "webhooks.json"),
        token_store=token_store,
    )
    # attach_plugin_http replaces event_bus with the server's broadcaster.
    bus = adapter.event_bus
    attach_plugin_http(server, "forge", adapter, list(surfaces))
    if bus is not None:
        adapter.event_bus = bus
    return TestClient(server.app), token_store, server


def admin_headers(token_store):
    _record, raw = token_store.create_admin_token(name="test-request-token")
    return {"Authorization": f"Bearer {raw}"}


def seed(adapter, now):
    """Fill both forges' caches as a successful refresh at `now` would."""
    home, gh = adapter.forges["home"], adapter.forges["gh"]
    home.repos, home.viewer = ["octo-org/demo", "octo-org/tools"], "reviewer1"
    home.prs = [
        make_pr(),
        make_pr(
            number=8,
            title="Rework the settings page",
            author="reviewer1",
            draft=True,
            labels=[],
            requested_reviewers=[],
            ci_state="none",
            unresolved_threads=0,
            review_state="none",
            created_at="2026-10-10T09:00:00Z",
            updated_at="2026-10-10T09:00:00Z",
            url="https://forge.example.test/octo-org/demo/pulls/8",
        ),
    ]
    gh.repos, gh.viewer = ["octo-org/demo"], "octocat"
    gh.prs = [
        make_pr(
            forge="gh",
            number=12,
            title="Fix flaky upload test",
            labels=["tests"],
            requested_reviewers=["octocat"],
            ci_state="success",
            unresolved_threads=None,
            review_state="none",
            created_at="2026-10-10T11:30:00Z",
            updated_at="2026-10-10T11:45:00Z",
            url="https://github.com/octo-org/demo/pull/12",
        )
    ]
    for cache in (home, gh):
        cache.last_refresh = now - 120
        cache.next_attempt = now + 180


def git_checkout(path: Path, origin: str | None) -> Path:
    import subprocess

    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    if origin:
        subprocess.run(["git", "-C", str(path), "remote", "add", "origin", origin], check=True)
    return path


def seed_history(session_id: str, workspace, events=()):
    """Record a history session that started in `workspace`. `events` is (type, data) pairs."""
    from tsugite.history import get_history_backend

    storage = get_history_backend().create(
        agent_name="test", model="m", session_id=session_id, workspace=str(workspace) if workspace else None
    )
    for event_type, data in events:
        storage.record(event_type, **data)
    return storage


def tool_calls(*calls):
    """A code_execution event holding (tool, arguments) calls."""
    return ("code_execution", {"code": "...", "tool_calls": [{"tool": t, "arguments": a} for t, a in calls]})
