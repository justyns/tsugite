"""Static asset caching contract: browsers must revalidate JS/CSS on every
load (so a daemon upgrade shows up on a plain reload) while 304s keep
within-version loads cheap. A stale cached stylesheet produced at least one
phantom bug report."""

from pathlib import Path
from unittest.mock import patch

import pytest
import tsugite_daemon
from starlette.testclient import TestClient
from tsugite_daemon.adapters.http import HTTPAgentAdapter, HTTPServer
from tsugite_daemon.auth import TokenStore
from tsugite_daemon.config import AgentConfig, HTTPConfig
from tsugite_daemon.session_store import SessionStore

WEB_DIR = Path(tsugite_daemon.__file__).parent / "web"

# These assert against real Vite-built assets - gitignored build output absent on a
# source checkout that hasn't run the web build. Skip (not fail) when no build is
# present; a packaged install or a built CI run still exercises them.
_HAS_BUILT_UI = (WEB_DIR / "index.html").exists() and any((WEB_DIR / "assets").glob("*.js"))
pytestmark = pytest.mark.skipif(not _HAS_BUILT_UI, reason="no built web UI on this checkout")


def _built_asset(suffix: str) -> str:
    """Resolve a real Vite-emitted asset (hashed names change every build)."""
    match = next(iter(sorted((WEB_DIR / "assets").glob(f"*{suffix}"))), None)
    assert match, f"no built {suffix} asset under web/assets - run `mise run web-build`"
    return f"/static/assets/{match.name}"


@pytest.fixture
def client(tmp_path):
    from tsugite.workspace import WorkspaceNotFoundError

    workspace = tmp_path / "ws"
    workspace.mkdir()
    store = SessionStore(tmp_path / "session_store.json")
    config = AgentConfig(workspace_dir=workspace, agent_file="default")
    with patch("tsugite.workspace.Workspace") as mock_ws:
        mock_ws.load.side_effect = WorkspaceNotFoundError("nope")
        adapter = HTTPAgentAdapter(agent_name="test-agent", agent_config=config, session_store=store)
    server = HTTPServer(
        config=HTTPConfig(enabled=True, host="127.0.0.1", port=8374),
        adapters={"test-agent": adapter},
        webhook_store=None,
        agent_configs={"test-agent": config},
        token_store=TokenStore(tmp_path / "tokens.json"),
    )
    return TestClient(server.app)


def test_js_and_css_must_revalidate(client):
    for path in (_built_asset(".js"), _built_asset(".css")):
        resp = client.get(path)
        assert resp.status_code == 200, path
        assert "no-cache" in resp.headers.get("cache-control", ""), (
            f"{path} must carry no-cache so a plain reload picks up a new daemon version"
        )
        assert resp.headers.get("etag"), f"{path} needs an ETag for cheap 304 revalidation"


def test_revalidation_is_cheap_304(client):
    path = _built_asset(".js")
    first = client.get(path)
    resp = client.get(path, headers={"If-None-Match": first.headers["etag"]})
    assert resp.status_code == 304, "unchanged assets must revalidate as 304, not a full refetch"


def test_index_html_is_no_cache(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "no-cache" in resp.headers.get("cache-control", "")


def test_icons_may_cache_but_js_never_silently(client):
    """The revalidate list is suffix-scoped; .js is the load-bearing suffix even
    though Vite hashes filenames (index.html itself is no-cache, so hashed
    references stay fresh - the suffix rule is defense in depth)."""
    resp = client.get(_built_asset(".js"))
    assert "no-cache" in resp.headers.get("cache-control", "")
