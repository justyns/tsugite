"""The OnlyOffice plugin's daemon adapter: HTTP routes, editor config, lifecycle.

Kept out of tools.py so importing the tool half does not drag tsugite_daemon and
starlette into every process that lists tools; only the daemon loads this module,
via the tsugite.adapters entry point. The turn protocol the routes and the tools
meet over lives in sessions.py.
"""

import asyncio
import logging
from pathlib import Path
from urllib.parse import quote

import httpx
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse
from starlette.routing import Route
from tsugite_daemon.adapters.base import BaseAdapter

from tsugite_onlyoffice import jwt
from tsugite_onlyoffice.command_service import CommandClient, CommandServiceError
from tsugite_onlyoffice.config import OnlyOfficeConfig
from tsugite_onlyoffice.documents import (
    MEDIA_TYPE,
    OutsideDocumentsError,
    canonical,
    list_documents,
    resolve,
    resolve_existing,
    write_atomic,
)
from tsugite_onlyoffice.sessions import DocumentSessions
from tsugite_onlyoffice.tools import DOCUMENT_EVENT, set_onlyoffice_runtime

logger = logging.getLogger(__name__)

# The editor may not fetch document.url until well after the config was issued,
# and it re-fetches on every refresh, so the file token outlives the config one.
FILE_TOKEN_TTL = 3600

# Which route a token was minted for. The two claim sets are otherwise the same
# and the callback's carries no expiry, so without this the doc_token off any
# callback URL is a file token that never runs out. The document server replays
# both URLs verbatim and inspects neither claim.
USE_FILE = "file"
USE_CALLBACK = "callback"

DOWNLOAD_TIMEOUT = 60.0

# The document server answers a request it will not attempt with this and no
# detail, which reads as a fetch failure but is not one.
ERROR_NOT_ATTEMPTED = -4

# Every tsugite theme but this one is dark, and the web UI defaults to a dark
# one, so a name that is not in here is a better bet as dark than as light.
LIGHT_THEMES = {"latte"}

# Callback statuses the document server sends to callbackUrl.
STATUS_EDITING = 1
STATUS_SAVE = 2
STATUS_SAVE_ERROR = 3
STATUS_CLOSED_UNCHANGED = 4
STATUS_FORCE_SAVE = 6
STATUS_FORCE_SAVE_ERROR = 7


def _bearer(request: Request) -> str | None:
    header = request.headers.get("authorization", "")
    return header[7:].strip() if header.lower().startswith("bearer ") else None


def _refused(exc: ValueError) -> JSONResponse:
    """Turn a documents-policy refusal into the status code it is owed."""
    return JSONResponse({"error": str(exc)}, status_code=403 if isinstance(exc, OutsideDocumentsError) else 404)


def _callback_refusal(message: str, status: int = 401) -> JSONResponse:
    """A callback answer the document server reads as a refusal it should retry."""
    return JSONResponse({"error": 1, "message": message}, status_code=status)


class OnlyOfficeAdapter(BaseAdapter):
    """Long-lived daemon-side object for one ONLYOFFICE Docs server."""

    def __init__(self, config: OnlyOfficeConfig):
        # Not agent-scoped, so BaseAdapter.__init__ is skipped; the gateway
        # overwrites event_bus with the SSE broadcaster.
        self.config = config
        self.event_bus = None
        self.sessions = DocumentSessions(config, self._command_client, self._announce)
        self._secret: str | None = None
        self.http: httpx.AsyncClient | None = None
        self.commands: CommandClient | None = None

    async def start(self) -> None:
        # Tools run on executor threads; a turn has to be driven from the loop
        # the callback and the SSE bus live on.
        self.sessions.loop = asyncio.get_running_loop()
        # One client for every outbound request. Building one costs a CA-bundle
        # load on the loop, and a per-request client throws the connection pool
        # away, so each command would pay a fresh TLS handshake.
        self.http = httpx.AsyncClient(timeout=DOWNLOAD_TIMEOUT)
        set_onlyoffice_runtime(self.sessions)
        logger.info("onlyoffice adapter started (document server %s)", self.config.server_url)

    async def stop(self) -> None:
        set_onlyoffice_runtime(None)
        # A timer that fires into a torn-down adapter announces a key nobody can
        # serve, so the pending swaps go with the loop they were armed on.
        self.sessions.cancel_pending_announces()
        self.sessions.loop = None
        if self.http is not None:
            await self.http.aclose()
            self.http = None
        self.commands = None
        logger.info("onlyoffice adapter stopped")

    def _announce(self, relative: str, key: str) -> None:
        """Tell the web UI a document moved on, and which key it moved on to.

        Logged because this is the only step between an agent's edit and the tab
        finding out about it, and a tab that never refreshes looks the same from
        the outside whether the event was never sent or never arrived.
        """
        if self.event_bus is None:
            logger.warning("onlyoffice cannot announce %s: no event bus on this adapter", relative)
            return
        logger.info("onlyoffice announcing %s on key %s", relative, key)
        self.event_bus.emit(DOCUMENT_EVENT, {"path": relative, "key": key})

    def _command_client(self) -> CommandClient:
        """The CommandService client, built on first use so the secret stays lazy."""
        if self.commands is None:
            self.commands = CommandClient(self.config.server_url, self.jwt_secret(), self.http)
        return self.commands

    def _rejection_message(self, code: int) -> str:
        """Say what a non-zero CommandService code means for this deployment."""
        public = self.config.public_base_url
        if code == ERROR_NOT_ATTEMPTED:
            return (
                f"The document server at {self.config.server_url} answered error {code}, which means it did "
                f"not attempt the connection. Check plugins.onlyoffice.public_base_url ({public}): the "
                "document server fetches that address itself, and a stock document server refuses private "
                "addresses until services.CoAuthoring.request-filtering-agent.allowPrivateIPAddress is on."
            )
        return (
            f"The document server at {self.config.server_url} rejected the command with error {code}. Check "
            f"plugins.onlyoffice.public_base_url ({public}), which the document server fetches itself, and "
            "that the shared JWT secret is the same on both sides."
        )

    # ── HTTP ──

    def get_http_routes(self) -> list:
        return [
            Route("/docs", self._list_documents, methods=["GET"]),
            Route("/config", self._editor_config, methods=["GET"]),
            Route("/health", self._health, methods=["GET"]),
        ]

    def get_public_http_routes(self) -> list:
        # The document server carries no daemon token, so these two gate
        # themselves on the shared JWT instead.
        return [
            Route("/file/{doc:path}", self._serve_file, methods=["GET"]),
            Route("/callback/{doc:path}", self._save_callback, methods=["POST"]),
        ]

    async def _health(self, request: Request) -> JSONResponse:
        """Ask the document server for its version and report what came back.

        The report carries `ok`, the configured `server_url` and
        `public_base_url`, the reported `version` when there is one, and a
        `message` saying what to fix when there is not.
        """
        report = {"server_url": self.config.server_url, "public_base_url": self.config.public_base_url}
        try:
            version = await self._command_client().version()
        except CommandServiceError as exc:
            return JSONResponse(
                {**report, "ok": False, "command_error": exc.code, "message": self._rejection_message(exc.code)}
            )
        except Exception as exc:  # noqa: BLE001 -- a diagnostic reports the failure rather than raising it
            return JSONResponse(
                {
                    **report,
                    "ok": False,
                    "message": (
                        f"Could not reach the document server at {self.config.server_url}: {exc!s} "
                        f"({type(exc).__name__}). Check plugins.onlyoffice.server_url and that this daemon can "
                        "open a connection to it."
                    ),
                }
            )
        return JSONResponse(
            {
                **report,
                "ok": True,
                "version": version,
                "message": (
                    f"The document server at {self.config.server_url} answered, version {version}. It fetches "
                    f"plugins.onlyoffice.public_base_url ({self.config.public_base_url}) itself, so that address "
                    "has to be reachable from the document server and not only from this browser."
                ),
            }
        )

    async def _list_documents(self, request: Request) -> JSONResponse:
        documents = await asyncio.to_thread(list_documents, self.config.documents_dir)
        return JSONResponse({"documents": documents})

    async def _editor_config(self, request: Request) -> JSONResponse:
        asked = request.query_params.get("path")
        if not asked:
            return JSONResponse({"error": "the 'path' query parameter is required"}, status_code=400)
        # Reduced to one spelling before anything is built from it: the two URLs
        # and both token claims leave this process, and a client or proxy that
        # normalizes `./notes.docx` on the way out then fetches a path no token
        # was minted for. The announce carries this spelling too, so it is also
        # what the page has to hold to recognise its own document.
        try:
            relative = canonical(self.config.documents_dir, asked)
            path = resolve_existing(self.config.documents_dir, relative)
        except ValueError as exc:
            return _refused(exc)
        config = self.editor_config(relative, path, request.query_params.get("user"), request.query_params.get("theme"))
        return JSONResponse({"server_url": self.config.server_url, "path": relative, "config": config})

    async def _serve_file(self, request: Request) -> FileResponse | JSONResponse:
        relative = request.path_params["doc"]
        # The document server's outbound header name is configurable, so the
        # query form is the only one guaranteed to reach us.
        token = request.query_params.get("token") or _bearer(request)
        if not token:
            return JSONResponse({"error": "missing token"}, status_code=401)
        if (refusal := self._token_refusal(token, relative, USE_FILE)) is not None:
            return JSONResponse({"error": refusal}, status_code=401)
        try:
            path = resolve_existing(self.config.documents_dir, relative)
        except ValueError as exc:
            return _refused(exc)
        return FileResponse(path, media_type=MEDIA_TYPE, filename=path.name)

    async def _save_callback(self, request: Request) -> JSONResponse:
        """Persist what the document server saved, and tell it whether that worked.

        Anything other than `{"error": 0}` makes the document server keep
        retrying, and a `0` it did not earn makes it drop the edit, so the
        answer tracks the write rather than the request.
        """
        relative = request.path_params["doc"]
        # Nothing in the body names the document, so the only thing binding this
        # save to this path is the token minted onto the callback URL itself.
        if (rejection := self._verify_document(request, relative)) is not None:
            return rejection
        try:
            body = await request.json()
        except ValueError:
            body = None
        if not isinstance(body, dict):
            return _callback_refusal("callback body is not a JSON object", 400)
        if (rejection := self._verify_callback(request, body)) is not None:
            return rejection
        try:
            path = resolve(self.config.documents_dir, relative)
        except ValueError as exc:
            logger.warning("onlyoffice callback for %s refused: %s", relative, exc)
            return _callback_refusal("the callback path was refused", 403)

        status = body.get("status")
        key = body.get("key")
        # The callback is the only thing that ever says whether a document is open,
        # and which session it is that has it open, which is what a turn reads to
        # decide it has a session to force-save. It has to be read even from a
        # session whose save is about to be refused: drop the lifecycle with the
        # payload and a closed session stays live forever, and the generation that
        # retires its key stops counting.
        if status == STATUS_EDITING:
            self.sessions.session_started(relative, key)
        elif status in (STATUS_SAVE, STATUS_CLOSED_UNCHANGED):
            self.sessions.session_ended(relative, key)

        if not self.sessions.is_current_key(relative, key):
            return self._refuse_superseded(relative)

        if status in (STATUS_SAVE_ERROR, STATUS_FORCE_SAVE_ERROR):
            logger.warning("onlyoffice reported a save error for %s: %r", relative, body)
        elif status in (STATUS_SAVE, STATUS_FORCE_SAVE):
            # A forcesave payload is the current authoritative state, so it is
            # written back exactly like an end-of-session save.
            try:
                download = await self.http.get(body["url"])
                download.raise_for_status()
                # The download runs for up to DOWNLOAD_TIMEOUT and holds no lock,
                # so a whole agent turn fits between the check above and this one,
                # and these bytes are then the ones it replaced.
                if not self.sessions.is_current_key(relative, key):
                    return self._refuse_superseded(relative)
                await asyncio.to_thread(write_atomic, path, download.content)
            except Exception as exc:  # noqa: BLE001 -- any failure here has to reach the document server as a retry
                logger.warning("onlyoffice save for %s failed: %s", relative, exc)
                return _callback_refusal("the save could not be written", 200)
            logger.info("onlyoffice saved %s (status %s)", relative, status)
            self.sessions.deliver(relative)
        elif status == STATUS_CLOSED_UNCHANGED:
            # A session that closed with nothing typed since its last save parks
            # no payload, and the file on disk is already what it held. That is
            # still the answer for a turn parked on a save.
            logger.info("onlyoffice closed %s with nothing to save", relative)
            self.sessions.deliver(relative)
        return JSONResponse({"error": 0})

    def _refuse_superseded(self, relative: str) -> JSONResponse:
        """Acknowledge a save from a session an agent turn has already moved past.

        That session is still holding the bytes the turn replaced, so writing its
        save back would undo the edit, and anything but a zero would only have the
        document server retry until it did. Warned rather than noted: whatever was
        typed into that session after the turn force-saved it goes down with the
        refusal.
        """
        logger.warning("onlyoffice refused a save for a superseded key on %s", relative)
        return JSONResponse({"error": 0})

    def _token_refusal(self, token: str, relative: str, use: str) -> str | None:
        """Say why a route token is not one this route may act on, or None when it is.

        Both public routes check the same three things, and each has a token the
        other mints, so the document and the purpose are read in one place rather
        than once per handler.
        """
        try:
            claims = jwt.verify(token, self.jwt_secret())
        except ValueError as exc:
            return str(exc)
        if claims.get("document") != relative:
            return "token was not issued for this document"
        if claims.get("use") != use:
            return "token was not issued for this route"
        return None

    def _verify_document(self, request: Request, relative: str) -> JSONResponse | None:
        """Check the callback URL's own token was minted for the document being written.

        Returns:
            A 401 response, or None when the path checks out.
        """
        if not (token := request.query_params.get("doc_token")):
            return _callback_refusal("missing doc_token")
        if (refusal := self._token_refusal(token, relative, USE_CALLBACK)) is not None:
            return _callback_refusal(refusal)
        return None

    def _verify_callback(self, request: Request, body: dict) -> JSONResponse | None:
        """Check the callback's JWT signs the body being acted on.

        A signature check alone would only prove the caller once held a valid
        token, and any observed token could then be replayed with a chosen `url`.

        Returns:
            A 401 response, or None when the callback checks out.
        """
        wrapped = False
        if not (token := body.get("token")):
            # The header form signs {"payload": <body>}, one level deeper.
            token = _bearer(request)
            wrapped = True
        if not token:
            return _callback_refusal("missing token")
        try:
            claims = jwt.verify(token, self.jwt_secret())
        except ValueError as exc:
            return _callback_refusal(str(exc))
        signed = claims.get("payload") if wrapped else claims
        # Field-by-field alone matches absence with absence, so a token signing
        # none of them would authenticate a body carrying none of them.
        if not isinstance(signed, dict) or "status" not in signed:
            return _callback_refusal("token does not sign a callback")
        if any(signed.get(f) != body.get(f) for f in ("status", "url", "key")):
            return _callback_refusal("token does not sign this callback body")
        return None

    # ── config building ──

    def jwt_secret(self) -> str:
        """The shared document-server secret, resolved from the backend on first use."""
        if self._secret is None:
            self._secret = self.config.resolve_jwt_secret()
        return self._secret

    def editor_config(self, relative: str, path: Path, user: str | None = None, theme: str | None = None) -> dict:
        """Build the signed config the browser hands to `DocsAPI.DocEditor`.

        Args:
            relative: The document's `canonical` path relative to the documents
                directory. Every identity built here is signed for it: both token
                claims, `document.url` and `callbackUrl`.
            path: The resolved file on disk.
            user: Who is viewing, from the web UI. Anything the human writes in the
                editor is authored under this, so it must never be the agent: two
                participants sharing one id are one participant to the document
                server, and to whoever reads the comments afterwards.
            theme: The tsugite theme the page is on, mapped to the editor's own.

        Returns:
            The editor config, with a `token` signing every other field.
        """
        secret = self.jwt_secret()
        base = f"{self.config.public_base_url.rstrip('/')}/api/plugins/onlyoffice"
        quoted = quote(relative)
        file_token = jwt.sign({"document": relative, "use": USE_FILE}, secret, expires_in=FILE_TOKEN_TTL)
        # No expiry: the callback has to outlive the editing session, and an
        # expired one would fail a save the editor believes it already made. The
        # purpose claim is what keeps that from also being a forever file read.
        callback_token = jwt.sign({"document": relative, "use": USE_CALLBACK}, secret, expires_in=0)
        config = {
            "document": {
                "fileType": "docx",
                # Minting goes through the sessions, which both remember the key a
                # command will name this session by and keep a retired one out.
                "key": self.sessions.open_key(relative, path),
                "title": path.name,
                "url": f"{base}/file/{quoted}?token={file_token}",
                "permissions": {"edit": True, "comment": True},
            },
            "documentType": "word",
            "editorConfig": {
                "callbackUrl": f"{base}/callback/{quoted}?doc_token={callback_token}",
                "lang": "en",
                "user": {"id": user, "name": user} if user else {"id": "viewer", "name": "Viewer"},
                # `compactToolbar` collapses the ribbon until a tab is clicked, which
                # is as small as the editor gets here: the side panels and the status
                # bar need the Developer Edition extended white-label licence, and an
                # unlicensed server ignores `customization.layout` in silence.
                "customization": {
                    "compactToolbar": True,
                    "hideRulers": True,
                    "uiTheme": "theme-light" if theme in LIGHT_THEMES else "theme-dark",
                },
            },
        }
        config["token"] = jwt.sign(config, secret)
        return config


def create_adapter(*, config, runtime, session_store, identity_map):
    """Adapter-plugin factory (the tsugite.adapters entry point).

    `config` is the daemon.yaml plugins.onlyoffice dict; runtime,
    session_store and identity_map are unused, because the adapter is not
    agent-scoped. Returns None to stay inactive, which the gateway skips.

    An enabled block the model refuses raises rather than returning None. The
    plugin loader logs what a factory raised, so the daemon says which key it
    could not read; returning None spends the config model's `extra="forbid"` on
    a typo that makes the plugin quietly not exist.

    Raises:
        ValidationError: The block is enabled and the model cannot read it.
    """
    if not config.get("enabled"):
        logger.info("onlyoffice plugin installed but disabled (set plugins.onlyoffice.enabled: true to activate)")
        return None
    return OnlyOfficeAdapter(OnlyOfficeConfig(**config))
