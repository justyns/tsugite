"""The MCP plugin's daemon adapter: the server list, OAuth sign-in, tool calls and HTTP routes."""

import asyncio
import contextlib
import json
import logging
import re
from dataclasses import dataclass, field
from ipaddress import ip_address
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import httpx
from mcp import ClientSession
from mcp.client.auth import OAuthClientProvider, OAuthFlowError
from mcp.client.auth.utils import (
    build_oauth_authorization_server_metadata_discovery_urls,
    build_protected_resource_metadata_discovery_urls,
    handle_auth_metadata_response,
    handle_protected_resource_response,
)
from mcp.client.streamable_http import streamable_http_client
from mcp.shared._httpx_utils import create_mcp_http_client
from mcp.shared.auth import OAuthClientMetadata
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse
from starlette.routing import Route
from tsugite_daemon.adapters.base import BaseAdapter

from tsugite.config import get_xdg_data_path
from tsugite.secrets.registry import get_registry
from tsugite_mcp import tools
from tsugite_mcp.storage import NO_REDIRECT, SecretStorage

logger = logging.getLogger(__name__)

NAME_PATTERN = re.compile(r"[a-z][a-z0-9]{0,23}")
CALLBACK_PATH = "/api/plugins/mcp/callback"
LIST_TIMEOUT = 30
SIGNIN_TIMEOUT = 300
NEEDS_CLIENT = (
    "This server does not offer dynamic client registration. Register Tsugite in the provider's developer "
    "console with the redirect URL shown above, then edit this server and enter the client ID and secret."
)
HTML = "<!doctype html><meta charset=utf-8><title>Tsugite</title><p>{}</p>"


@dataclass
class Server:
    name: str
    url: str
    status: str = "connecting"
    error: str | None = None
    tools: list = field(default_factory=list)


def _leaves(exc: BaseException):
    """The exceptions inside nested exception groups, which the SDK's task groups wrap errors in."""
    if isinstance(exc, BaseExceptionGroup):
        for inner in exc.exceptions:
            yield from _leaves(inner)
    else:
        yield exc


def _needs_signin(exc: BaseException) -> bool:
    return any(
        type(leaf) is OAuthFlowError or (isinstance(leaf, httpx.HTTPStatusError) and leaf.response.status_code == 401)
        for leaf in _leaves(exc)
    )


def _describe(exc: BaseException) -> str:
    text = "; ".join(f"{type(leaf).__name__}: {leaf}" for leaf in _leaves(exc))
    return get_registry().mask(text)


def pick_auth_method(supported: list[str] | None, has_secret: bool) -> str:
    """The token endpoint auth method for a user-entered client, from the server's metadata."""
    supported = supported or ["client_secret_basic"]
    for method in ("client_secret_basic", "client_secret_post") if has_secret else ("none",):
        if method in supported:
            return method
    kind = "with" if has_secret else "without"
    raise RuntimeError(
        f"the server's token endpoint accepts {', '.join(supported)}, which does not fit a client {kind} a secret"
    )


class _Provider(OAuthClientProvider):
    async def _exchange_token_authorization_code(self, *args, **kwargs):
        storage = self.context.storage
        if storage.using_preset and self.context.oauth_metadata:
            info = self.context.client_info
            method = pick_auth_method(
                self.context.oauth_metadata.token_endpoint_auth_methods_supported, bool(info.client_secret)
            )
            info.token_endpoint_auth_method = method
            await storage.set_preset({**(await storage.get_preset()), "auth_method": method})
        return await super()._exchange_token_authorization_code(*args, **kwargs)


def validate_server(name: str, url: str) -> None:
    if not NAME_PATTERN.fullmatch(name):
        raise ValueError("name must be 1-24 lowercase letters and digits, starting with a letter")
    parsed = urlparse(url)
    loopback = parsed.hostname == "localhost"
    if parsed.hostname and not loopback:
        with contextlib.suppress(ValueError):
            loopback = ip_address(parsed.hostname).is_loopback
    if parsed.scheme != "https" and not (parsed.scheme == "http" and loopback):
        raise ValueError("url must be https (http is allowed for localhost)")
    if not parsed.hostname:
        raise ValueError("url needs a host")


class McpAdapter(BaseAdapter):
    def __init__(self, path: Path, client_factory=create_mcp_http_client):
        # Not agent-scoped, so BaseAdapter.__init__ is skipped.
        self.event_bus = None
        self.loop: asyncio.AbstractEventLoop | None = None
        self.path = path
        self.client_factory = client_factory
        self.servers = {name: Server(name, url) for name, url in self._read().items()}
        self._signins: dict[str, asyncio.Task] = {}
        # state -> the future the OAuth callback route resolves with (code, state)
        self._flows: dict[str, asyncio.Future] = {}
        self._startup: asyncio.Future | None = None

    def _read(self) -> dict[str, str]:
        return json.loads(self.path.read_text()) if self.path.exists() else {}

    def _write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({s.name: s.url for s in self.servers.values()}, indent=2))

    async def start(self) -> None:
        self.loop = asyncio.get_running_loop()
        tools.set_adapter(self)
        self._startup = asyncio.gather(*(self.refresh(name) for name in self.servers))

    async def stop(self) -> None:
        tools.set_adapter(None)
        for task in [self._startup, *self._signins.values()]:
            if task is not None:
                task.cancel()
        for name in self.servers:
            tools.clear_tools(name)

    @contextlib.asynccontextmanager
    async def _session(self, server: Server, signin=None):
        """An initialized session. `signin` is (redirect_uri, redirect_handler, callback_handler).

        Without it, auth is attached only when tokens exist, and a server that needs a new
        sign-in fails with an error `_needs_signin` recognizes.
        """
        storage = SecretStorage(server.name)
        auth = None
        if signin or await storage.get_tokens():
            redirect_uri, redirect_handler, callback_handler = signin or (NO_REDIRECT, None, None)
            auth = _Provider(
                server.url,
                OAuthClientMetadata(
                    client_name="Tsugite",
                    redirect_uris=[redirect_uri],
                    grant_types=["authorization_code", "refresh_token"],
                    response_types=["code"],
                    token_endpoint_auth_method="none",
                ),
                storage,
                redirect_handler,
                callback_handler,
            )
            auth.context.token_expiry_time = await storage.expires_at()
        async with self.client_factory(auth=auth) as http_client:
            async with streamable_http_client(server.url, http_client=http_client) as streams:
                async with ClientSession(streams[0], streams[1]) as session:
                    await session.initialize()
                    yield session

    def _cancel_signin(self, name: str) -> None:
        if task := self._signins.pop(name, None):
            task.cancel()

    async def refresh(self, name: str, signin=None) -> None:
        """Connect to the server and replace its status and tools."""
        if not signin:
            self._cancel_signin(name)
        server = self.servers[name]
        server.status = "connecting"
        try:
            async with asyncio.timeout(SIGNIN_TIMEOUT if signin else LIST_TIMEOUT):
                async with self._session(server, signin) as session:
                    listed = (await session.list_tools()).tools
            if self.servers.get(name) is not server:
                return
            registered = tools.set_tools(name, listed)
        except Exception as exc:
            server.status, server.error, server.tools = "error", _describe(exc), []
            if _needs_signin(exc):
                server.status, server.error = "needs_signin", None
                await self._check_client(server)
            tools.clear_tools(name)
            return
        server.status, server.error, server.tools = "connected", None, registered

    async def _check_client(self, server: Server) -> None:
        """Flag a server that needs a client the user has not entered, because it cannot register one itself."""
        storage = SecretStorage(server.name)
        if await storage.get_preset() or await storage.get_registration():
            return
        try:
            dynamic = await self._registers_dynamically(server)
        except Exception as exc:
            server.status, server.error = "error", _describe(exc)
            return
        if not dynamic:
            server.status, server.error = "needs_client", NEEDS_CLIENT

    async def _registers_dynamically(self, server: Server) -> bool:
        """Whether the authorization server advertises a registration endpoint. True when no metadata is found."""
        async with self.client_factory() as http:
            auth_server = None
            for url in build_protected_resource_metadata_discovery_urls(None, server.url):
                if resource := await handle_protected_resource_response(await http.get(url)):
                    auth_server = str(resource.authorization_servers[0])
                    break
            for url in build_oauth_authorization_server_metadata_discovery_urls(auth_server, server.url):
                ok, metadata = await handle_auth_metadata_response(await http.get(url))
                if metadata:
                    return metadata.registration_endpoint is not None
                if not ok:
                    break
        return True

    async def call_tool(self, name: str, tool: str, arguments: dict) -> str:
        server = self.servers[name]
        try:
            async with self._session(server) as session:
                result = await session.call_tool(tool, arguments)
        except Exception as exc:
            if _needs_signin(exc):
                server.status, server.error = "needs_signin", None
                raise RuntimeError(f"{name} needs sign-in, open Connectors") from None
            raise RuntimeError(_describe(exc)) from None
        text = "\n".join(c.text if c.type == "text" else f"[{c.type} content omitted]" for c in result.content)
        text = get_registry().mask(text)
        if result.isError:
            raise RuntimeError(text)
        return text

    async def add(self, name: str, url: str, client_id: str = "", client_secret: str = "") -> Server:
        validate_server(name, url)
        if name in self.servers:
            raise ValueError(f"{name} is already connected")
        if client_secret and not client_id:
            raise ValueError("a client secret needs a client ID")
        self.servers[name] = Server(name, url)
        self._write()
        if client_id:
            await self._save_client(name, client_id, client_secret, False)
        await self.refresh(name)
        return self.servers[name]

    async def _save_client(self, name: str, client_id: str, client_secret: str, clear_secret: bool) -> None:
        """Store the user's client, or remove it when `client_id` is empty. An empty secret keeps the saved one for the same ID."""
        storage = SecretStorage(name)
        old = await storage.get_preset() or {}
        record = {"client_id": client_id} if client_id else None
        kept = "" if clear_secret or client_id != old.get("client_id") else old.get("client_secret", "")
        secret = client_secret or kept
        if record and secret:
            record["client_secret"] = secret
        if (record or {}) == {k: v for k, v in old.items() if k != "auth_method"}:
            return
        await storage.delete_registration()
        await storage.set_preset(record)

    async def update(self, name: str, client_id: str, client_secret: str = "", clear_secret: bool = False) -> None:
        if client_secret and not client_id:
            raise ValueError("a client secret needs a client ID")
        await self._save_client(name, client_id, client_secret, clear_secret)
        await self.refresh(name)

    async def remove(self, name: str) -> None:
        self._cancel_signin(name)
        del self.servers[name]
        self._write()
        tools.clear_tools(name)
        await SecretStorage(name).delete_all()

    async def signin(self, name: str, redirect_base: str) -> str:
        """Start the OAuth flow and return the URL the user opens. It finishes after the callback."""
        self._cancel_signin(name)
        redirect_uri = redirect_base.rstrip("/") + CALLBACK_PATH
        storage = SecretStorage(name)
        if self.servers[name].status == "needs_client":
            raise RuntimeError(NEEDS_CLIENT)
        registered = await storage.get_registration()
        if registered and [str(uri) for uri in registered.redirect_uris] != [redirect_uri]:
            await storage.delete_registration()
        url_ready: asyncio.Future = self.loop.create_future()
        state = None

        async def redirect_handler(url: str) -> None:
            nonlocal state
            state = parse_qs(urlparse(url).query)["state"][0]
            self._flows[state] = self.loop.create_future()
            url_ready.set_result(url)

        async def callback_handler() -> tuple[str, str]:
            return await self._flows[state]

        task = asyncio.create_task(self.refresh(name, (redirect_uri, redirect_handler, callback_handler)))
        task.add_done_callback(lambda _: self._flows.pop(state, None))
        self._signins[name] = task
        await asyncio.wait({url_ready, task}, return_when=asyncio.FIRST_COMPLETED)
        if not url_ready.done():
            raise RuntimeError(self.servers[name].error or "the server did not ask for sign-in")
        return url_ready.result()

    def get_http_routes(self) -> list:
        return [
            Route("/servers", self._list, methods=["GET"]),
            Route("/servers", self._add, methods=["POST"]),
            Route("/servers/{name}", self._update, methods=["PUT"]),
            Route("/servers/{name}", self._remove, methods=["DELETE"]),
            Route("/servers/{name}/refresh", self._refresh, methods=["POST"]),
            Route("/servers/{name}/signin", self._signin, methods=["POST"]),
        ]

    def get_public_http_routes(self) -> list:
        # The browser's redirect cannot send the daemon token. A request is accepted only
        # when its `state` matches a sign-in this process started.
        return [Route("/callback", self._callback, methods=["GET"])]

    async def _status(self) -> dict:
        """The server list for the page. The client secret is reported only as `has_secret`."""
        servers = []
        for s in self.servers.values():
            preset = await SecretStorage(s.name).get_preset() or {}
            servers.append(
                {
                    "name": s.name,
                    "url": s.url,
                    "status": s.status,
                    "error": s.error,
                    "client_id": preset.get("client_id", ""),
                    "has_secret": bool(preset.get("client_secret")),
                    "tools": s.tools,
                }
            )
        return {"servers": servers}

    async def _list(self, request: Request) -> JSONResponse:
        return JSONResponse(await self._status())

    async def _add(self, request: Request) -> JSONResponse:
        try:
            body = await _body(request)
            await self.add(
                str(body.get("name", "")),
                str(body.get("url", "")),
                str(body.get("client_id", "")).strip(),
                str(body.get("client_secret", "")).strip(),
            )
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        return JSONResponse(await self._status())

    async def _update(self, request: Request) -> JSONResponse:
        if (name := self._known(request)) is None:
            return JSONResponse({"error": "no such server"}, status_code=404)
        try:
            body = await _body(request)
            await self.update(
                name,
                str(body.get("client_id", "")).strip(),
                str(body.get("client_secret", "")).strip(),
                bool(body.get("clear_secret")),
            )
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        return JSONResponse(await self._status())

    def _known(self, request: Request) -> str | None:
        name = request.path_params["name"]
        return name if name in self.servers else None

    async def _remove(self, request: Request) -> JSONResponse:
        if (name := self._known(request)) is None:
            return JSONResponse({"error": "no such server"}, status_code=404)
        await self.remove(name)
        return JSONResponse(await self._status())

    async def _refresh(self, request: Request) -> JSONResponse:
        if (name := self._known(request)) is None:
            return JSONResponse({"error": "no such server"}, status_code=404)
        await self.refresh(name)
        return JSONResponse(await self._status())

    async def _signin(self, request: Request) -> JSONResponse:
        if (name := self._known(request)) is None:
            return JSONResponse({"error": "no such server"}, status_code=404)
        try:
            body = await _body(request)
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        try:
            url = await self.signin(name, str(body.get("redirect_base", "")))
        except Exception as exc:
            return JSONResponse({"error": _describe(exc)}, status_code=502)
        return JSONResponse({"authorization_url": url})

    async def _callback(self, request: Request) -> HTMLResponse:
        params = request.query_params
        flow = self._flows.get(params.get("state", ""))
        if flow is None or flow.done():
            return HTMLResponse(HTML.format("This sign-in link is unknown or expired."), status_code=400)
        if "code" in params:
            flow.set_result((params["code"], params["state"]))
            return HTMLResponse(HTML.format("Signed in. You can close this tab."))
        flow.set_exception(RuntimeError(f"authorization denied: {params.get('error', 'no code returned')}"))
        return HTMLResponse(HTML.format("Sign-in was not completed. You can close this tab."), status_code=400)


async def _body(request: Request) -> dict:
    try:
        body = await request.json()
    except json.JSONDecodeError:
        body = None
    if not isinstance(body, dict):
        raise ValueError("expected a JSON object")
    return body


def create_adapter(*, config, runtime, session_store, identity_map):
    """Adapter-plugin factory (the tsugite.adapters entry point). Returns None unless plugins.mcp.enabled is true."""
    if not (config or {}).get("enabled"):
        logger.info("mcp plugin installed but disabled (set plugins.mcp.enabled: true to activate)")
        return None
    return McpAdapter(get_xdg_data_path("mcp") / "servers.json")
