"""StaticMixin: static HTTP handlers for HTTPServer (split from adapters/http.py)."""

from typing import TYPE_CHECKING

from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, Response
from starlette.routing import Mount, Route

from tsugite_daemon.adapters.http.helpers import (
    WEB_DIR,
    _NoCacheStaticFiles,
    _web_assets_version,
)

if TYPE_CHECKING:
    pass


class StaticMixin:
    def _static_routes(self) -> list:
        return [
            Mount("/static", app=_NoCacheStaticFiles(directory=str(WEB_DIR)), name="static"),
            Route("/sw.js", self._serve_sw, methods=["GET"]),
            Route("/", self._serve_ui, methods=["GET"]),
        ]

    async def _serve_sw(self, request: Request) -> Response:
        sw_path = WEB_DIR / "sw.js"
        if not sw_path.exists():
            return JSONResponse({"error": "service worker not found"}, status_code=404)
        # Prepend a version constant derived from the web dir's latest mtime so the
        # browser treats the SW as updated whenever any web asset changes.
        version = _web_assets_version()
        body = f'const SW_VERSION = "{version}";\n'.encode() + sw_path.read_bytes()
        return Response(body, media_type="application/javascript", headers={"Cache-Control": "no-cache"})

    async def _serve_ui(self, request: Request) -> Response:
        ui_path = WEB_DIR / "index.html"
        if not ui_path.exists():
            return JSONResponse({"error": "web UI not found"}, status_code=404)
        return HTMLResponse(ui_path.read_text(), headers={"Cache-Control": "no-cache"})
