"""An in-process MCP server with OAuth (discovery, registration, PKCE, refresh) for the plugin's tests."""

import base64
import hashlib
import secrets
from urllib.parse import parse_qs, urlparse

import httpx
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

BASE = "https://mcp.test"
URL = f"{BASE}/mcp"

TOOLS = [
    {
        "name": "echo",
        "description": "Repeat text back.\nSecond line.",
        "inputSchema": {
            "type": "object",
            "properties": {"loud": {"type": "boolean"}, "text": {"type": "string"}},
            "required": ["text"],
        },
    },
    {"name": "leak", "description": "Return the caller's bearer token.", "inputSchema": {"type": "object"}},
    {"name": "fail", "description": "Always errors.", "inputSchema": {"type": "object"}},
]


class FakeServer:
    """Serves OAuth metadata, `/register`, `/token` and a JSON-only Streamable HTTP `/mcp`.

    With `oauth=False` the MCP endpoint accepts anyone. `denied` lists requests the OAuth
    endpoints rejected, e.g. a PKCE verifier that does not match. `dynamic=False` leaves
    `registration_endpoint` out of the metadata. `auth_methods` is what the token endpoint
    advertises. A client made in the provider's console is added with `preregister`.
    """

    def __init__(self, oauth=True, dynamic=True, auth_methods=("none",)):
        self.oauth = oauth
        self.dynamic = dynamic
        self.auth_methods = list(auth_methods)
        self.secrets = {}
        self.token_auth = []
        self.clients = {}
        self.codes = {}
        self.access = set()
        self.refresh_tokens = set()
        self.grants = []
        self.requests = []
        self.denied = []
        self.app = Starlette(
            routes=[
                Route("/.well-known/oauth-protected-resource/{path:path}", self.resource_metadata),
                Route("/.well-known/oauth-protected-resource", self.resource_metadata),
                Route("/.well-known/oauth-authorization-server", self.server_metadata),
                Route("/register", self.register, methods=["POST"]),
                Route("/token", self.token, methods=["POST"]),
                Route("/mcp", self.mcp, methods=["GET", "POST", "DELETE"]),
            ]
        )

    def preregister(self, client_id, redirect_uri, secret=None):
        self.clients[client_id] = [redirect_uri]
        self.secrets[client_id] = secret

    def client_factory(self, headers=None, timeout=None, auth=None):
        transport = httpx.ASGITransport(app=self.app)
        return httpx.AsyncClient(
            transport=transport, base_url=BASE, headers=headers, timeout=timeout, auth=auth, follow_redirects=True
        )

    async def resource_metadata(self, request: Request):
        self.requests.append("resource-metadata")
        return JSONResponse({"resource": URL, "authorization_servers": [BASE]})

    async def server_metadata(self, request: Request):
        self.requests.append("server-metadata")
        metadata = {
            "issuer": BASE,
            "authorization_endpoint": f"{BASE}/authorize",
            "token_endpoint": f"{BASE}/token",
            "response_types_supported": ["code"],
            "grant_types_supported": ["authorization_code", "refresh_token"],
            "code_challenge_methods_supported": ["S256"],
            "token_endpoint_auth_methods_supported": self.auth_methods,
        }
        if self.dynamic:
            metadata["registration_endpoint"] = f"{BASE}/register"
        return JSONResponse(metadata)

    async def register(self, request: Request):
        self.requests.append("register")
        body = await request.json()
        client_id = f"client-{len(self.clients) + 1}"
        self.clients[client_id] = body["redirect_uris"]
        return JSONResponse({**body, "client_id": client_id}, status_code=201)

    def authorize(self, url: str) -> tuple[str, str]:
        """What the browser and the user do at the authorization URL. Returns (code, state)."""
        query = {k: v[0] for k, v in parse_qs(urlparse(url).query).items()}
        assert urlparse(url).path == "/authorize"
        assert query["response_type"] == "code"
        assert query["code_challenge_method"] == "S256"
        assert query["redirect_uri"] in self.clients[query["client_id"]]
        code = secrets.token_urlsafe(8)
        self.codes[code] = (query["code_challenge"], query["redirect_uri"], query["client_id"])
        return code, query["state"]

    def issue(self, with_refresh=True):
        access = f"access-{secrets.token_hex(6)}"
        self.access.add(access)
        body = {"access_token": access, "token_type": "Bearer", "expires_in": 3600}
        if with_refresh:
            refresh = f"refresh-{secrets.token_hex(6)}"
            self.refresh_tokens.add(refresh)
            body["refresh_token"] = refresh
        return JSONResponse(body)

    async def token(self, request: Request):
        form = parse_qs((await request.body()).decode())
        form = {k: v[0] for k, v in form.items()}
        if not self.check_client(request, form):
            return JSONResponse({"error": "invalid_client"}, status_code=401)
        self.grants.append(form["grant_type"])
        if form["grant_type"] == "refresh_token":
            if form["refresh_token"] not in self.refresh_tokens:
                self.denied.append("refresh")
                return JSONResponse({"error": "invalid_grant"}, status_code=400)
            return self.issue(with_refresh=False)
        challenge, redirect_uri, client_id = self.codes.pop(form["code"])
        digest = hashlib.sha256(form["code_verifier"].encode()).digest()
        if base64.urlsafe_b64encode(digest).rstrip(b"=").decode() != challenge:
            self.denied.append("pkce")
            return JSONResponse({"error": "invalid_grant"}, status_code=400)
        assert (form["redirect_uri"], form["client_id"]) == (redirect_uri, client_id)
        return self.issue()

    def check_client(self, request: Request, form: dict) -> bool:
        """Verify the client's secret when it has one, and record how it authenticated."""
        header = request.headers.get("authorization", "")
        client_id = form.get("client_id", "")
        if header.startswith("Basic "):
            client_id, _, given = base64.b64decode(header[6:]).decode().partition(":")
            method = "client_secret_basic"
        else:
            given = form.get("client_secret")
            method = "client_secret_post" if given else "none"
        self.token_auth.append(method)
        if method not in self.auth_methods or given != self.secrets.get(client_id):
            self.denied.append("client-auth")
            return False
        return True

    async def mcp(self, request: Request):
        bearer = request.headers.get("authorization", "").removeprefix("Bearer ")
        if self.oauth and bearer not in self.access:
            self.requests.append("mcp-401")
            challenge = f'Bearer resource_metadata="{BASE}/.well-known/oauth-protected-resource/mcp"'
            return Response(status_code=401, headers={"WWW-Authenticate": challenge})
        if request.method != "POST":
            return Response(status_code=405)
        message = await request.json()
        if "id" not in message:
            return Response(status_code=202)
        params = message.get("params", {})
        match message["method"]:
            case "initialize":
                result = {
                    "protocolVersion": params["protocolVersion"],
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "fake", "version": "1"},
                }
            case "tools/list":
                result = {"tools": TOOLS}
            case "tools/call":
                result = self.call(params["name"], params.get("arguments", {}), bearer)
            case _:
                return JSONResponse({"jsonrpc": "2.0", "id": message["id"], "error": {"code": -32601, "message": "?"}})
        return JSONResponse({"jsonrpc": "2.0", "id": message["id"], "result": result})

    def call(self, name: str, arguments: dict, bearer: str) -> dict:
        self.requests.append(f"call:{name}:{sorted(arguments)}")
        match name:
            case "echo":
                text = arguments["text"].upper() if arguments.get("loud") else arguments["text"]
                return {"content": [{"type": "text", "text": f"echo: {text}"}]}
            case "leak":
                return {"content": [{"type": "text", "text": f"token {bearer}"}]}
            case _:
                return {"content": [{"type": "text", "text": "it broke"}], "isError": True}
