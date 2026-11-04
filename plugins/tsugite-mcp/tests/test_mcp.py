"""Sign-in, token handling, tool registration and routes for the MCP plugin, against an in-process server."""

import asyncio
import json
import logging
from contextlib import asynccontextmanager

import pytest
from fake_server import URL

from tsugite.secrets.registry import get_registry
from tsugite.tools import call_tool, expand_tool_specs, get_tool

pytestmark = pytest.mark.asyncio

ORIGIN = "http://daemon"
CALLBACK = "/api/plugins/mcp/callback"


async def settled(adapter, name="fake"):
    for _ in range(200):
        if adapter.servers[name].status != "connecting":
            return adapter.servers[name]
        await asyncio.sleep(0.01)
    raise AssertionError("server never left 'connecting'")


async def begin(daemon, adapter, name="fake", **client):
    """Add the server unless it exists and start the sign-in. Returns the authorization URL."""
    if name not in adapter.servers:
        await adapter.add(name, URL, **client)
    resp = await daemon.post(
        f"/api/plugins/mcp/servers/{name}/signin", json={"redirect_base": ORIGIN}, headers=daemon.auth_headers
    )
    assert resp.status_code == 200, resp.text
    return resp.json()["authorization_url"]


async def sign_in(daemon, adapter, fake, name="fake", **client):
    """Start the sign-in and play the browser. Returns the callback response."""
    code, state = fake.authorize(await begin(daemon, adapter, name, **client))
    callback = await daemon.get(CALLBACK, params={"code": code, "state": state})
    await settled(adapter, name)
    return callback


async def test_sign_in_registers_pkce_exchange_and_tools(daemon, adapter, fake, secrets_backend, caplog):
    caplog.set_level(logging.DEBUG)
    assert (await adapter.add("fake", URL)).status == "needs_signin"
    assert fake.requests.count("register") == 0

    callback = await sign_in(daemon, adapter, fake)
    assert callback.status_code == 200

    server = adapter.servers["fake"]
    assert server.status == "connected"
    assert fake.requests.count("register") == 1
    assert fake.grants == ["authorization_code"]
    assert fake.denied == []
    assert fake.clients == {"client-1": [ORIGIN + CALLBACK]}
    assert [t["name"] for t in server.tools] == ["mcp_fake_echo", "mcp_fake_leak", "mcp_fake_fail"]
    assert set(secrets_backend.values) == {"mcp-fake-tokens", "mcp-fake-client"}

    listing = await daemon.get("/api/plugins/mcp/servers", headers=daemon.auth_headers)
    assert listing.json()["servers"][0]["tools"][0] == {"name": "mcp_fake_echo", "description": "Repeat text back."}
    secret_values = [
        json.loads(secrets_backend.values["mcp-fake-tokens"])[k] for k in ("access_token", "refresh_token")
    ]
    for token in secret_values:
        assert token not in listing.text
        assert token not in caplog.text


async def test_tools_run_as_registered_tsugite_tools(daemon, adapter, fake):
    await sign_in(daemon, adapter, fake)

    assert expand_tool_specs(["@mcp_fake", "-mcp_fake_fail"], strict=True) == ["mcp_fake_echo", "mcp_fake_leak"]
    assert expand_tool_specs(["mcp_fake_*"], strict=True) == ["mcp_fake_echo", "mcp_fake_leak", "mcp_fake_fail"]
    info = get_tool("mcp_fake_echo")
    assert list(info.parameters) == ["text", "loud"]
    assert info.parameters["text"]["required"] and not info.parameters["loud"]["required"]
    assert info.parameters["text"]["type"] is str

    assert await asyncio.to_thread(call_tool, "mcp_fake_echo", text="hi") == "echo: hi"
    assert await asyncio.to_thread(call_tool, "mcp_fake_echo", text="hi", loud=True) == "echo: HI"
    # An argument left at None is not sent.
    assert fake.requests[-1] == "call:echo:['loud', 'text']"
    assert fake.requests[-2] == "call:echo:['text']"


async def test_tool_results_never_contain_the_token(daemon, adapter, fake):
    await sign_in(daemon, adapter, fake)

    assert await asyncio.to_thread(call_tool, "mcp_fake_leak") == "token ***"


async def test_tool_error_is_raised(daemon, adapter, fake):
    await sign_in(daemon, adapter, fake)

    with pytest.raises(RuntimeError, match="it broke"):
        await asyncio.to_thread(call_tool, "mcp_fake_fail")


@pytest.mark.parametrize(
    "params",
    [{}, {"state": "wrong"}, {"code": "c", "state": "wrong"}, {"code": "c"}],
    ids=["bare", "state-only-wrong", "code-wrong-state", "code-without-state"],
)
async def test_callback_rejects_unknown_state(daemon, adapter, fake, params):
    url = await begin(daemon, adapter)

    rejected = await daemon.get(CALLBACK, params=params)

    assert rejected.status_code == 400
    assert adapter.servers["fake"].status == "connecting"
    code, state = fake.authorize(url)
    assert (await daemon.get(CALLBACK, params={"code": code, "state": state})).status_code == 200
    assert (await settled(adapter)).status == "connected"


async def test_callback_state_is_single_use(daemon, adapter, fake):
    url = await begin(daemon, adapter)
    code, state = fake.authorize(url)
    assert (await daemon.get(CALLBACK, params={"code": code, "state": state})).status_code == 200
    await settled(adapter)

    assert (await daemon.get(CALLBACK, params={"code": code, "state": state})).status_code == 400


async def test_cancelled_signin_forgets_its_state(daemon, adapter, fake):
    code, state = fake.authorize(await begin(daemon, adapter))

    await adapter.remove("fake")
    await asyncio.sleep(0)

    assert state not in adapter._flows
    assert (await daemon.get(CALLBACK, params={"code": code, "state": state})).status_code == 400


async def test_refresh_cancels_a_pending_signin(daemon, adapter, fake):
    code, state = fake.authorize(await begin(daemon, adapter))

    await daemon.post("/api/plugins/mcp/servers/fake/refresh", headers=daemon.auth_headers)
    await asyncio.sleep(0)

    assert adapter._signins == {}
    assert (await daemon.get(CALLBACK, params={"code": code, "state": state})).status_code == 400


@pytest.mark.parametrize("body", [b"not json", b"[1]"], ids=["not-json", "not-object"])
@pytest.mark.parametrize(
    ("method", "path"),
    [("post", "/servers"), ("put", "/servers/fake"), ("post", "/servers/fake/signin")],
    ids=["add", "update", "signin"],
)
async def test_malformed_body_is_rejected(daemon, adapter, method, path, body):
    await adapter.add("fake", URL)

    resp = await getattr(daemon, method)(
        f"/api/plugins/mcp{path}", content=body, headers={**daemon.auth_headers, "content-type": "application/json"}
    )

    assert resp.status_code == 400
    assert resp.json() == {"error": "expected a JSON object"}


async def test_denied_authorization_shows_the_error(daemon, adapter, fake):
    url = await begin(daemon, adapter)
    _code, state = fake.authorize(url)

    await daemon.get(CALLBACK, params={"error": "access_denied", "state": state})

    server = await settled(adapter)
    assert server.status == "error"
    assert "access_denied" in server.error


async def test_wrong_pkce_verifier_is_rejected_by_the_server(daemon, adapter, fake):
    url = await begin(daemon, adapter)
    code, state = fake.authorize(url)
    _challenge, redirect_uri, client_id = fake.codes[code]
    fake.codes[code] = ("0" * 43, redirect_uri, client_id)

    await daemon.get(CALLBACK, params={"code": code, "state": state})

    assert (await settled(adapter)).status == "error"
    assert fake.denied == ["pkce"]


async def test_expired_token_is_refreshed(daemon, adapter, fake, secrets_backend):
    await sign_in(daemon, adapter, fake)
    record = json.loads(secrets_backend.values["mcp-fake-tokens"])
    old_access = record["access_token"]
    record["expires_at"] = 1.0
    secrets_backend.values["mcp-fake-tokens"] = json.dumps(record)

    assert await asyncio.to_thread(call_tool, "mcp_fake_echo", text="again") == "echo: again"

    assert fake.grants == ["authorization_code", "refresh_token"]
    stored = json.loads(secrets_backend.values["mcp-fake-tokens"])
    assert stored["access_token"] != old_access
    assert stored["refresh_token"] == record["refresh_token"]
    assert stored["expires_at"] > 1.0
    assert adapter.servers["fake"].status == "connected"


async def test_rejected_refresh_asks_for_sign_in(daemon, adapter, fake, secrets_backend):
    await sign_in(daemon, adapter, fake)
    record = json.loads(secrets_backend.values["mcp-fake-tokens"])
    record["expires_at"] = 1.0
    secrets_backend.values["mcp-fake-tokens"] = json.dumps(record)
    fake.refresh_tokens.clear()

    with pytest.raises(RuntimeError, match="needs sign-in"):
        await asyncio.to_thread(call_tool, "mcp_fake_echo", text="again")

    assert adapter.servers["fake"].status == "needs_signin"
    assert fake.requests.count("register") == 1


async def test_server_without_oauth_connects_directly(tmp_path, secrets_backend):
    from fake_server import FakeServer
    from tsugite_mcp.adapter import McpAdapter

    open_server = FakeServer(oauth=False)
    adapter = McpAdapter(tmp_path / "servers.json", client_factory=open_server.client_factory)
    await adapter.start()
    try:
        assert (await adapter.add("fake", URL)).status == "connected"
        assert open_server.requests == []
    finally:
        await adapter.stop()


async def test_unusable_and_colliding_tools_are_skipped(caplog):
    from mcp.types import Tool
    from tsugite_mcp import tools

    listed = [
        Tool(name="a-b", inputSchema={"type": "object"}),
        Tool(name="a_b", inputSchema={"type": "object"}),
        Tool(name="range", inputSchema={"type": "object", "properties": {"from": {"type": "string"}}}),
        Tool(name="flag", inputSchema={"type": "object", "properties": {"on": True}}),
    ]
    try:
        assert [t["name"] for t in tools.set_tools("t", listed)] == ["mcp_t_a_b"]
        assert expand_tool_specs(["mcp_t_*"]) == ["mcp_t_a_b"]
    finally:
        tools.clear_tools("t")
    assert "'from'" in caplog.text and "'on'" in caplog.text


async def test_failed_registration_is_an_error(tmp_path, secrets_backend, monkeypatch):
    from fake_server import FakeServer
    from tsugite_mcp import tools
    from tsugite_mcp.adapter import McpAdapter

    adapter = McpAdapter(tmp_path / "servers.json", client_factory=FakeServer(oauth=False).client_factory)
    await adapter.add("fake", URL)
    monkeypatch.setattr(tools, "set_tools", lambda *_: 1 / 0)

    await adapter.refresh("fake")

    assert adapter.servers["fake"].status == "error"
    assert expand_tool_specs(["mcp_fake_*"]) == []


async def test_server_removed_mid_refresh_registers_no_tools(tmp_path, secrets_backend, monkeypatch):
    from fake_server import FakeServer
    from tsugite_mcp.adapter import McpAdapter

    adapter = McpAdapter(tmp_path / "servers.json", client_factory=FakeServer(oauth=False).client_factory)
    await adapter.add("fake", URL)
    session = adapter._session

    @asynccontextmanager
    async def removing(server, signin=None):
        async with session(server, signin) as s:
            await adapter.remove(server.name)
            yield s

    monkeypatch.setattr(adapter, "_session", removing)
    await adapter.refresh("fake")

    assert expand_tool_specs(["mcp_fake_*"]) == []


async def test_disconnect_removes_tools_and_secrets(daemon, adapter, fake, secrets_backend):
    await sign_in(daemon, adapter, fake)

    resp = await daemon.delete("/api/plugins/mcp/servers/fake", headers=daemon.auth_headers)

    assert resp.json() == {"servers": []}
    assert secrets_backend.values == {}
    assert expand_tool_specs(["mcp_fake_*"]) == []
    assert not adapter.path.read_text().strip("{}\n ")


async def test_servers_survive_a_restart(daemon, adapter, fake, tmp_path):
    from tsugite_mcp.adapter import McpAdapter

    await sign_in(daemon, adapter, fake)
    await adapter.stop()
    assert expand_tool_specs(["mcp_fake_*"]) == []

    restarted = McpAdapter(adapter.path, client_factory=fake.client_factory)
    await restarted.start()
    try:
        assert (await settled(restarted)).status == "connected"
        assert expand_tool_specs(["@mcp_fake"]) == ["mcp_fake_echo", "mcp_fake_fail", "mcp_fake_leak"]
    finally:
        await restarted.stop()


async def test_routes_need_the_daemon_token_except_the_callback(daemon):
    for method, path in [("get", "/servers"), ("post", "/servers"), ("delete", "/servers/fake")]:
        assert (await getattr(daemon, method)(f"/api/plugins/mcp{path}")).status_code in (401, 403)
    assert (await daemon.get(CALLBACK)).status_code == 400


async def test_unknown_server_is_404(daemon):
    resp = await daemon.post("/api/plugins/mcp/servers/nope/signin", json={}, headers=daemon.auth_headers)
    assert resp.status_code == 404


REDIRECT = ORIGIN + CALLBACK
CLIENT_SECRET = "console-secret-123"


@pytest.mark.parametrize("fake", [{"dynamic": False}], indirect=True)
async def test_preregistered_public_client(daemon, adapter, fake, secrets_backend):
    fake.preregister("public-id", REDIRECT)

    assert (await sign_in(daemon, adapter, fake, client_id="public-id")).status_code == 200

    assert adapter.servers["fake"].status == "connected"
    assert fake.token_auth == ["none"]
    assert fake.denied == []
    assert "register" not in fake.requests
    assert set(secrets_backend.values) == {"mcp-fake-tokens", "mcp-fake-client-preset"}


@pytest.mark.parametrize(
    ("fake", "method"),
    [
        ({"dynamic": False, "auth_methods": ["client_secret_basic", "client_secret_post"]}, "client_secret_basic"),
        ({"dynamic": False, "auth_methods": ["client_secret_post"]}, "client_secret_post"),
    ],
    indirect=["fake"],
)
async def test_preregistered_confidential_client(daemon, adapter, fake, method):
    fake.preregister("conf-id", REDIRECT, CLIENT_SECRET)

    await sign_in(daemon, adapter, fake, client_id="conf-id", client_secret=CLIENT_SECRET)

    assert adapter.servers["fake"].status == "connected"
    assert fake.token_auth == [method]
    assert fake.denied == []


@pytest.mark.parametrize("fake", [{"auth_methods": ["client_secret_post"]}], indirect=True)
async def test_confidential_client_is_refreshed_with_the_saved_method(daemon, adapter, fake, secrets_backend):
    fake.preregister("conf-id", REDIRECT, CLIENT_SECRET)
    await sign_in(daemon, adapter, fake, client_id="conf-id", client_secret=CLIENT_SECRET)
    record = json.loads(secrets_backend.values["mcp-fake-tokens"])
    record["expires_at"] = 1.0
    secrets_backend.values["mcp-fake-tokens"] = json.dumps(record)

    assert await asyncio.to_thread(call_tool, "mcp_fake_echo", text="again") == "echo: again"

    assert fake.token_auth == ["client_secret_post", "client_secret_post"]
    assert fake.denied == []


async def test_preregistered_client_skips_dynamic_registration(daemon, adapter, fake):
    fake.preregister("public-id", REDIRECT)

    await sign_in(daemon, adapter, fake, client_id="public-id")

    assert adapter.servers["fake"].status == "connected"
    assert "register" not in fake.requests
    assert list(fake.clients) == ["public-id"]


@pytest.mark.parametrize("fake", [{"dynamic": False}], indirect=True)
async def test_server_without_registration_and_no_client_says_so(daemon, adapter, fake):
    server = await adapter.add("fake", URL)

    assert server.status == "needs_client"
    assert "dynamic client registration" in server.error
    listing = (await daemon.get("/api/plugins/mcp/servers", headers=daemon.auth_headers)).json()
    assert listing["servers"][0]["status"] == "needs_client"
    resp = await daemon.post(
        "/api/plugins/mcp/servers/fake/signin", json={"redirect_base": ORIGIN}, headers=daemon.auth_headers
    )
    assert resp.status_code == 502
    assert "dynamic client registration" in resp.json()["error"]
    assert fake.requests.count("register") == 0


@pytest.mark.parametrize("fake", [{"dynamic": False}], indirect=True)
async def test_adding_a_client_to_a_server_that_needs_one(daemon, adapter, fake):
    fake.preregister("public-id", REDIRECT)
    await adapter.add("fake", URL)

    resp = await daemon.put(
        "/api/plugins/mcp/servers/fake", json={"client_id": "public-id"}, headers=daemon.auth_headers
    )

    assert resp.json()["servers"][0]["status"] == "needs_signin"
    assert (await sign_in(daemon, adapter, fake)).status_code == 200


async def test_responses_never_contain_the_client_secret(daemon, adapter, fake, secrets_backend, caplog):
    caplog.set_level(logging.DEBUG)
    headers = daemon.auth_headers
    fake.preregister("conf-id", REDIRECT, CLIENT_SECRET)
    fake.auth_methods = ["client_secret_basic"]

    added = await daemon.post(
        "/api/plugins/mcp/servers",
        json={"name": "fake", "url": URL, "client_id": "conf-id", "client_secret": CLIENT_SECRET},
        headers=headers,
    )
    code, state = fake.authorize(
        (
            await daemon.post("/api/plugins/mcp/servers/fake/signin", json={"redirect_base": ORIGIN}, headers=headers)
        ).json()["authorization_url"]
    )
    await daemon.get(CALLBACK, params={"code": code, "state": state})
    await settled(adapter)
    listing = await daemon.get("/api/plugins/mcp/servers", headers=headers)
    replaced = await daemon.put(
        "/api/plugins/mcp/servers/fake",
        json={"client_id": "conf-id", "client_secret": "other-secret-456"},
        headers=headers,
    )

    entry = listing.json()["servers"][0]
    assert (entry["client_id"], entry["has_secret"]) == ("conf-id", True)
    assert "client_secret" not in entry
    for response in (added, listing, replaced):
        assert CLIENT_SECRET not in response.text
    assert "other-secret-456" not in replaced.text
    assert CLIENT_SECRET not in caplog.text
    assert "conf-id" not in adapter.path.read_text()
    assert CLIENT_SECRET not in adapter.path.read_text()
    assert get_registry().mask("x other-secret-456") == "x ***"


async def test_secret_can_be_cleared_and_kept(daemon, adapter, fake):
    headers = daemon.auth_headers
    fake.preregister("conf-id", REDIRECT, CLIENT_SECRET)
    await adapter.add("fake", URL, client_id="conf-id", client_secret=CLIENT_SECRET)

    kept = await daemon.put("/api/plugins/mcp/servers/fake", json={"client_id": "conf-id"}, headers=headers)
    cleared = await daemon.put(
        "/api/plugins/mcp/servers/fake", json={"client_id": "conf-id", "clear_secret": True}, headers=headers
    )
    removed = await daemon.put("/api/plugins/mcp/servers/fake", json={}, headers=headers)

    assert kept.json()["servers"][0]["has_secret"] is True
    assert cleared.json()["servers"][0]["has_secret"] is False
    assert removed.json()["servers"][0]["client_id"] == ""


async def test_secret_without_client_id_is_rejected(daemon):
    resp = await daemon.post(
        "/api/plugins/mcp/servers",
        json={"name": "fake", "url": URL, "client_secret": "s"},
        headers=daemon.auth_headers,
    )

    assert resp.status_code == 400
