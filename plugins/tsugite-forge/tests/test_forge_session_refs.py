"""Session refs end to end: stored pins and hides, refs derived from history, the routes and the tools."""

import asyncio
import subprocess
from datetime import UTC, datetime

import pytest
from forge_helpers import (
    TOKEN,
    Clock,
    admin_headers,
    forge_http,
    git_checkout,
    make_adapter,
    make_pr,
    seed,
    seed_history,
    tool_calls,
)

NOW = datetime(2026, 10, 10, 12, 0, tzinfo=UTC).timestamp()
BASE = "/api/plugins/forge"


@pytest.fixture
def workspace(tmp_path):
    ws = git_checkout(tmp_path / "ws", "git@forge.example.test:octo-org/tools.git")
    git_checkout(ws / "demo", "ssh://git@forge.example.test:2222/octo-org/demo.git")
    git_checkout(ws / "site", "https://github.com/octo-org/site.git")
    git_checkout(ws / "other", "git@gitlab.example.test:octo-org/other.git")
    (tmp_path / "plain").mkdir()
    return ws


@pytest.fixture
def store(tmp_path):
    from tsugite_daemon.session_store import Session, SessionSource, SessionStore

    store = SessionStore(tmp_path / "session_store.json")
    store.create_session(Session(id="s-new", source=SessionSource.INTERACTIVE.value, user_id="u1"))
    return store


class Bus:
    def __init__(self):
        self.events = []

    def emit(self, event_type, data):
        self.events.append((event_type, data))


@pytest.fixture
def clock():
    return Clock(NOW)


@pytest.fixture
def adapter(store, clock):
    adapter = make_adapter(clock, session_store=store)
    seed(adapter, NOW)
    adapter.event_bus = Bus()
    return adapter


@pytest.fixture
def history(workspace, tmp_path, store):
    seed_history(
        "s-old",
        workspace,
        [
            tool_calls(("read_file", {"path": "demo/src/app.py", "start_line": 1})),
            ("user_input", {"text": "Look at https://github.com/octo-org/site/pull/12 please"}),
        ],
    )
    seed_history(
        "s-new",
        workspace,
        [
            ("compaction", {"summary": "earlier work", "source_session_id": "s-old"}),
            tool_calls(
                ("read_file", {"path": "site/README.md"}),
                ("list_files", {"path": "other"}),
                ("list_files", {"path": "."}),
                ("spawn_job", {"prompt": "fix it", "repo": "demo"}),
                ("forge_pr", {"repo": "home:octo-org/demo", "number": 7}),
                ("write_file", {"path": str(tmp_path / "plain" / "x.txt"), "content": "x"}),
            ),
            ("model_response", {"raw_content": "Opened https://forge.example.test/octo-org/demo/pulls/8, see also #9"}),
        ],
    )
    store.set_metadata_bulk(
        "s-new",
        {
            "refs": [
                {"forge": "home", "repo": "octo-org/demo", "pr": 8, "source": "pinned"},
                {"forge": "gh", "repo": "octo-org/site", "pr": 12, "hidden": True},
            ]
        },
    )


def _rows(refs):
    return [(r["forge"], r["repo"], r["pr"], r["source"], r["hidden"]) for r in refs]


EXPECTED = [
    ("home", "octo-org/demo", 8, "pinned", False),
    ("home", "octo-org/demo", None, "touched", False),
    ("gh", "octo-org/site", None, "touched", False),
    ("home", "octo-org/tools", None, "touched", False),
    ("home", "octo-org/demo", 7, "touched", False),
    ("gh", "octo-org/site", 12, "mentioned", True),
    ("home", "octo-org/demo", 9, "mentioned", False),
]


def test_session_refs_merge_pins_history_and_hides(adapter, history):
    refs = adapter.session_refs("s-new")
    assert _rows(refs) == EXPECTED
    by_key = {(r["forge"], r["repo"], r["pr"]): r for r in refs}
    assert by_key[("home", "octo-org/demo", 7)]["pull"]["title"] == "Add retry to the sync job"
    assert by_key[("home", "octo-org/demo", 9)]["pull"] is None
    assert by_key[("home", "octo-org/demo", None)]["pull"] is None


def test_git_is_asked_once_per_checkout(adapter, history, monkeypatch):
    calls = []
    real_run = subprocess.run

    def counting_run(argv, *args, **kwargs):
        calls.append(argv)
        return real_run(argv, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", counting_run)
    adapter.session_refs("s-new")
    roots = [argv[2] for argv in calls]
    assert len(roots) == len(set(roots)) == 4


def test_compaction_chain_is_followed_but_bounded(adapter, workspace, store, monkeypatch):
    import tsugite_forge.refs as refs

    monkeypatch.setattr(refs, "MAX_CHAIN", 2)
    seed_history("s-1", workspace, [("user_input", {"text": "https://forge.example.test/octo-org/demo/pulls/1"})])
    seed_history(
        "s-2",
        workspace,
        [
            ("compaction", {"source_session_id": "s-1"}),
            ("user_input", {"text": "https://forge.example.test/octo-org/demo/pulls/2"}),
        ],
    )
    seed_history(
        "s-new",
        workspace,
        [
            ("compaction", {"source_session_id": "s-2"}),
            ("user_input", {"text": "https://forge.example.test/octo-org/demo/pulls/3"}),
        ],
    )
    assert [r["pr"] for r in adapter.session_refs("s-new")] == [2, 3]


@pytest.mark.parametrize("value", ["https://forge.example.test/octo-org/demo/pulls/7", {"pr": 1}, [None, 3]])
def test_a_malformed_refs_value_reads_as_empty(adapter, history, store, value):
    store.set_metadata_bulk("s-new", {"refs": value})
    rows = _rows(adapter.session_refs("s-new"))
    assert ("home", "octo-org/demo", 8, "mentioned", False) in rows
    assert all(source != "pinned" for _f, _r, _p, source, _h in rows)


def test_bare_number_needs_every_pin_in_one_repo(adapter, history, store):
    pins = [
        {"forge": "home", "repo": "octo-org/demo", "pr": 8, "source": "pinned"},
        {"forge": "home", "repo": "octo-org/tools", "pr": None, "source": "pinned"},
    ]
    store.set_metadata_bulk("s-new", {"refs": pins})
    assert all(row["pr"] != 9 for row in adapter.session_refs("s-new"))


def test_a_session_without_history_has_only_its_pins(adapter, store):
    store.set_metadata_bulk(
        "s-new", {"refs": [{"forge": "gh", "repo": "octo-org/demo", "pr": None, "source": "pinned"}]}
    )
    assert _rows(adapter.session_refs("s-new")) == [("gh", "octo-org/demo", None, "pinned", False)]


def test_unknown_session_is_a_lookup_error(adapter):
    with pytest.raises(LookupError):
        adapter.session_refs("nope")


# ── routes ──


@pytest.fixture
def client(adapter, tmp_path):
    client, token_store, _server = forge_http(adapter, tmp_path)
    client.headers.update(admin_headers(token_store))
    return client


def test_refs_routes_require_a_daemon_token(client):
    del client.headers["Authorization"]
    assert client.get(f"{BASE}/sessions/s-new/refs").status_code == 401
    assert client.post(f"{BASE}/sessions/s-new/refs", json={"action": "pin"}).status_code == 401


def test_get_refs(client, history):
    resp = client.get(f"{BASE}/sessions/s-new/refs")
    assert resp.status_code == 200, resp.text
    assert _rows(resp.json()["refs"]) == EXPECTED


def test_get_refs_for_an_unknown_session_is_404(client):
    assert client.get(f"{BASE}/sessions/nope/refs").status_code == 404


def test_post_pin_hide_unhide_unpin(client, adapter, store, history):
    def post(action, forge, repo, pr):
        resp = client.post(
            f"{BASE}/sessions/s-new/refs", json={"action": action, "forge": forge, "repo": repo, "pr": pr}
        )
        assert resp.status_code == 200, resp.text
        return _rows(resp.json()["refs"])

    assert ("home", "octo-org/demo", 7, "pinned", False) in post("pin", "home", "octo-org/demo", 7)
    assert ("home", "octo-org/tools", None, "touched", True) in post("hide", "home", "octo-org/tools", None)
    assert ("home", "octo-org/tools", None, "touched", False) in post("unhide", "home", "octo-org/tools", None)
    assert ("home", "octo-org/demo", 7, "touched", False) in post("unpin", "home", "octo-org/demo", 7)
    assert ("gh", "octo-org/site", 12, "mentioned", False) in post("unhide", "gh", "octo-org/site", 12)
    assert store.get_session("s-new").metadata["refs"] == [
        {"forge": "home", "repo": "octo-org/demo", "pr": 8, "source": "pinned"}
    ]
    event_type, data = adapter.event_bus.events[-1]
    assert (event_type, data["action"], data["id"]) == ("session_update", "metadata_updated", "s-new")


@pytest.mark.parametrize(
    "body, status",
    [
        ({"action": "merge", "forge": "home", "repo": "octo-org/demo", "pr": 7}, 400),
        ({"action": "pin", "forge": "nowhere", "repo": "octo-org/demo", "pr": 7}, 400),
        ({"action": "pin", "forge": ["home"], "repo": "octo-org/demo", "pr": 7}, 400),
        ({"action": "pin", "forge": "home", "repo": "octo-org/*", "pr": 7}, 400),
        ({"action": "pin", "forge": "home", "repo": "octo-org/demo", "pr": "7"}, 400),
        ({"action": "pin", "forge": "home", "repo": "octo-org/demo"}, 400),
    ],
)
def test_post_refuses_bad_bodies(client, body, status):
    assert client.post(f"{BASE}/sessions/s-new/refs", json=body).status_code == status


def test_post_refuses_a_malformed_body(client):
    resp = client.post(
        f"{BASE}/sessions/s-new/refs", content=b"{not json", headers={"Content-Type": "application/json"}
    )
    assert (resp.status_code, resp.json()) == (400, {"error": "invalid JSON body"})


def test_refs_response_never_contains_the_token(client, adapter, history):
    from tsugite.secrets.registry import get_registry

    get_registry().register("forgejo-token", TOKEN)
    adapter.forges["home"].prs = [make_pr(title=f"leak {TOKEN}")]
    resp = client.get(f"{BASE}/sessions/s-new/refs")
    assert resp.status_code == 200
    assert "leak ***" in resp.text and TOKEN not in resp.text


def test_post_for_an_unknown_session_is_404(client):
    body = {"action": "pin", "forge": "home", "repo": "octo-org/demo", "pr": 7}
    assert client.post(f"{BASE}/sessions/nope/refs", json=body).status_code == 404


# ── tools ──


@pytest.mark.asyncio
async def test_forge_pin_and_unpin_act_on_the_current_session(adapter, store):
    from tsugite_daemon.session_runner import set_current_session_id
    from tsugite_forge.tools import forge_pin, forge_unpin, set_forge_adapter

    adapter.loop = asyncio.get_running_loop()
    set_forge_adapter(adapter)
    set_current_session_id("s-new")
    try:
        assert (
            await asyncio.to_thread(forge_pin, "home:octo-org/demo", 7)
            == "Pinned home:octo-org/demo#7 to this session."
        )
        assert await asyncio.to_thread(forge_pin, "octo-org/tools") == "Pinned home:octo-org/tools to this session."
        assert [e["pr"] for e in store.get_session("s-new").metadata["refs"]] == [7, None]
        assert await asyncio.to_thread(forge_unpin, "home:octo-org/demo", 7) == "Unpinned home:octo-org/demo#7."
        assert store.get_session("s-new").metadata["refs"] == [
            {"forge": "home", "repo": "octo-org/tools", "pr": None, "source": "pinned"}
        ]
    finally:
        set_forge_adapter(None)


@pytest.mark.asyncio
async def test_forge_unpin_removes_a_pin_on_a_repo_the_forge_no_longer_lists(adapter, store):
    from tsugite_daemon.session_runner import set_current_session_id
    from tsugite_forge.tools import forge_pin, forge_unpin, set_forge_adapter

    adapter.loop = asyncio.get_running_loop()
    set_forge_adapter(adapter)
    set_current_session_id("s-new")
    try:
        await asyncio.to_thread(forge_pin, "home:octo-org/demo", 7)
        adapter.forges["home"].repos = []
        assert await asyncio.to_thread(forge_unpin, "octo-org/demo", 7) == "Unpinned home:octo-org/demo#7."
        assert store.get_session("s-new").metadata["refs"] == []
        with pytest.raises(ValueError, match="not pinned"):
            await asyncio.to_thread(forge_unpin, "octo-org/demo", 7)
    finally:
        set_forge_adapter(None)


def test_forge_pin_needs_a_session(adapter):
    from tsugite_forge.tools import forge_pin, set_forge_adapter

    set_forge_adapter(adapter)
    try:
        with pytest.raises(RuntimeError, match="session"):
            forge_pin("home:octo-org/demo", 7)
    finally:
        set_forge_adapter(None)


# ── caching ──


@pytest.fixture
def spy(monkeypatch):
    """Counts git runs and history loads."""
    import tsugite_forge.refs as refs

    counts = {"git": 0, "loads": 0}
    real_run, real_load = subprocess.run, refs.load_histories

    def run(argv, *args, **kwargs):
        counts["git"] += argv[0] == "git"
        return real_run(argv, *args, **kwargs)

    def load(chain):
        counts["loads"] += 1
        return real_load(chain)

    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(refs, "load_histories", load)
    return counts


def test_unchanged_history_reuses_the_derived_refs(adapter, history, spy):
    first = adapter.session_refs("s-new")
    before = dict(spy)
    assert adapter.session_refs("s-new") == first
    assert spy == before


def test_a_new_history_event_derives_again(adapter, history, spy):
    from tsugite.history import get_history_backend

    adapter.session_refs("s-new")
    loads = spy["loads"]
    get_history_backend().load("s-new").record("user_input", text="https://forge.example.test/octo-org/demo/pulls/30")
    assert ("home", "octo-org/demo", 30) in {(r["forge"], r["repo"], r["pr"]) for r in adapter.session_refs("s-new")}
    assert spy["loads"] == loads + 1


def test_pin_and_hide_show_with_a_warm_cache(adapter, history, spy):
    adapter.session_refs("s-new")
    adapter.update_refs("s-new", "pin", "home", "octo-org/demo", 7)
    adapter.update_refs("s-new", "hide", "home", "octo-org/tools", None)
    rows = {(r["forge"], r["repo"], r["pr"]): r for r in adapter.session_refs("s-new")}
    assert rows[("home", "octo-org/demo", 7)]["source"] == "pinned"
    assert rows[("home", "octo-org/tools", None)]["hidden"] is True
    assert spy["loads"] == 1


def test_git_origins_expire_after_their_ttl(adapter, history, spy, clock):
    from tsugite_forge.adapter import ORIGIN_TTL

    from tsugite.history import get_history_backend

    def new_event():
        get_history_backend().load("s-new").record("user_input", text="more")

    adapter.session_refs("s-new")
    git = spy["git"]
    new_event()
    adapter.session_refs("s-new")
    assert spy["git"] == git
    clock.now += ORIGIN_TTL + 1
    new_event()
    adapter.session_refs("s-new")
    assert spy["git"] == 2 * git


class RepoProvider:
    def __init__(self, repos):
        self.repos = repos

    def list_repos(self):
        return self.repos

    def list_prs(self, repos):
        return []

    def viewer(self):
        return "octocat"


@pytest.mark.asyncio
async def test_a_refresh_derives_again(adapter, workspace, clock):
    seed_history("s-new", workspace, [tool_calls(("forge_pin", {"repo": "home:octo-org/fresh", "pr": 3}))])
    home = adapter.forges["home"]
    home.repos, home.last_refresh = [], None
    assert adapter.session_refs("s-new") == []
    home.provider = RepoProvider(["octo-org/fresh"])
    adapter.forges["gh"].provider = RepoProvider(["octo-org/demo"])
    clock.now += 1
    await adapter.refresh()
    assert _rows(adapter.session_refs("s-new")) == [("home", "octo-org/fresh", 3, "touched", False)]
