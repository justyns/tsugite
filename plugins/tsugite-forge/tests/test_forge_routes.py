"""The forge plugin's authenticated daemon routes."""

from datetime import UTC, datetime

import pytest
from forge_helpers import TOKEN, FakeForgejo, admin_headers, forge_http, make_adapter, make_pr, seed

NOW = datetime(2026, 10, 10, 12, 0, tzinfo=UTC).timestamp()
BASE = "/api/plugins/forge"
FORGEJO = "home"
PR_7 = f"/prs/{FORGEJO}/octo-org/demo/7"


class StubGithub:
    def list_repos(self):
        return ["octo-org/demo"]

    def list_prs(self, repos):
        return [make_pr(forge="gh", number=12, title="Fix flaky upload test")]

    def viewer(self):
        return "octocat"


@pytest.fixture
def adapter(secrets_backend):
    from tsugite_forge.providers.forgejo import ForgejoProvider

    adapter = make_adapter(lambda: NOW)
    seed(adapter, NOW)
    home = adapter.forges["home"]
    adapter.server = FakeForgejo()
    home.provider = ForgejoProvider(home.config, transport=adapter.server.transport)
    adapter.forges["gh"].provider = StubGithub()
    return adapter


@pytest.fixture
def http(adapter, tmp_path):
    return forge_http(adapter, tmp_path)


@pytest.fixture
def client(http):
    return http[0]


@pytest.fixture
def token_store(http):
    return http[1]


@pytest.fixture
def headers(token_store):
    return admin_headers(token_store)


@pytest.mark.parametrize(
    "method, path",
    [("get", "/prs"), ("get", PR_7), ("get", "/status"), ("post", "/refresh")],
)
def test_routes_require_a_daemon_token(client, method, path):
    assert getattr(client, method)(BASE + path).status_code == 401


def test_a_scoped_surface_token_reaches_forge_routes_only(client, token_store):
    raw = token_store.issue(ttl=60, scope="plugin:forge", identity="plugin-surface:forge")
    scoped = {"Authorization": f"Bearer {raw}"}
    assert client.get(f"{BASE}/status", headers=scoped).status_code == 200
    assert client.get("/api/sessions", headers=scoped).status_code == 403


def test_prs_lists_the_normalized_cache(client, headers):
    resp = client.get(f"{BASE}/prs", headers=headers)
    assert resp.status_code == 200, resp.text
    prs = resp.json()["prs"]
    assert [(pr["forge"], pr["number"]) for pr in prs] == [("gh", 12), ("home", 8), ("home", 7)]
    assert prs[2]["labels"] == ["bug", "sync"]
    assert prs[2]["head_sha"] == "7" * 40


@pytest.mark.parametrize(
    "query, numbers",
    [
        ("forge=home", [8, 7]),
        ("repo=gh:octo-org/demo", [12]),
        ("author=reviewer1", [8]),
        ("label=bug", [7]),
        ("needs_review=true", [12, 7]),
        ("needs_review=false", [8]),
        ("state=open&forge=gh", [12]),
    ],
)
def test_prs_filters(client, headers, query, numbers):
    resp = client.get(f"{BASE}/prs?{query}", headers=headers)
    assert [pr["number"] for pr in resp.json()["prs"]] == numbers


@pytest.mark.parametrize("query", ["state=closed", "needs_review=maybe", "forge=nope", "repo=nope:octo-org/demo"])
def test_prs_rejects_bad_filters(client, headers, query):
    resp = client.get(f"{BASE}/prs?{query}", headers=headers)
    assert resp.status_code == 400
    assert resp.json()["error"]


def test_pr_detail_reads_live_parts(client, headers):
    resp = client.get(f"{BASE}{PR_7}", headers=headers)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["pr"]["title"] == "Add retry to the sync job"
    assert body["checks"] == "failure"
    assert body["threads"] == {"total": 2, "unresolved": 1}
    assert body["diff"].startswith("diff --git a/sync.py")


def test_pr_detail_include_picks_parts(client, headers):
    body = client.get(f"{BASE}{PR_7}?include=checks,threads", headers=headers).json()
    assert set(body) == {"pr", "checks", "threads"}
    assert set(client.get(f"{BASE}{PR_7}?include=", headers=headers).json()) == {"pr"}


@pytest.mark.parametrize(
    "path, status",
    [("/prs/nowhere/octo-org/demo/7", 404), (PR_7 + "?include=comments", 400)],
)
def test_pr_detail_refuses_bad_requests(client, headers, path, status):
    resp = client.get(BASE + path, headers=headers)
    assert resp.status_code == status
    assert resp.json()["error"]


@pytest.mark.parametrize("repo", ["%2E%2E/%2E%2E", "octo-org/demo%3Fx%3D1", "octo-org/demo%23x", "octo-org/legacy"])
def test_pr_detail_only_reaches_a_listed_repo(client, headers, adapter, repo):
    resp = client.get(f"{BASE}/prs/{FORGEJO}/{repo}/7", headers=headers)
    assert resp.status_code == 404
    assert "is not in forge 'home'" in resp.json()["error"]
    assert adapter.server.requests == []


def test_pr_detail_reports_a_forge_failure_as_502(client, headers):
    resp = client.get(f"{BASE}/prs/{FORGEJO}/octo-org/demo/404", headers=headers)
    assert resp.status_code == 502
    assert "404" in resp.json()["error"]


def test_status_reports_each_forge(client, headers):
    forges = client.get(f"{BASE}/status", headers=headers).json()["forges"]
    assert [(f["name"], f["kind"], f["ok"], f["cache_age_seconds"], f["viewer"]) for f in forges] == [
        ("home", "forgejo", True, 120, "reviewer1"),
        ("gh", "github", True, 120, "octocat"),
    ]


def test_refresh_refreshes_every_forge_now(client, headers, adapter):
    resp = client.post(f"{BASE}/refresh", headers=headers)
    assert resp.status_code == 200, resp.text
    assert all(f["cache_age_seconds"] == 0 for f in resp.json()["forges"])
    assert [pr.number for pr in adapter.forges["home"].prs] == [7, 8]


def test_responses_never_contain_the_token(client, headers, adapter, monkeypatch):
    from tsugite.secrets.registry import get_registry

    get_registry().register("forgejo-token", TOKEN)
    adapter.forges["gh"].prs = [make_pr(forge="gh", number=12, title=f"leak {TOKEN}")]
    adapter.forges["gh"].error = f"gh: echoed {TOKEN}"
    monkeypatch.setattr(adapter.forges["home"].provider, "diff", lambda repo, number: f"+ token = {TOKEN}\n")
    bodies = [
        client.get(f"{BASE}/prs", headers=headers).text,
        client.get(f"{BASE}{PR_7}", headers=headers).text,
        client.get(f"{BASE}/status", headers=headers).text,
    ]
    assert all(TOKEN not in body for body in bodies)
