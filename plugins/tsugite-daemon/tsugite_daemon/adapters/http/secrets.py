"""SecretsMixin: secrets HTTP handlers for HTTPServer (split from adapters/http.py)."""

from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Mount, Route


class SecretsMixin:
    def _secrets_routes(self) -> list:
        return [
            Mount(
                "/api/secrets",
                name="secrets",
                routes=[
                    Route("/", self._secrets_list, methods=["GET"]),
                    Route("/{name:path}", self._secrets_set, methods=["POST"]),
                    Route("/{name:path}", self._secrets_delete, methods=["DELETE"]),
                ],
            ),
        ]

    async def _secrets_list(self, request: Request) -> JSONResponse:
        if err := self._check_auth(request):
            return err
        from tsugite.secrets import get_backend

        return JSONResponse({"secrets": get_backend().list_names()})

    async def _secrets_set(self, request: Request) -> JSONResponse:
        if err := self._check_auth(request):
            return err
        from tsugite.secrets import get_backend

        name = request.path_params["name"]
        body = await request.json()
        value = body.get("value")
        if not value:
            return JSONResponse({"error": "value is required"}, status_code=400)
        try:
            get_backend().set(name, value)
        except NotImplementedError:
            return JSONResponse({"error": "backend does not support writing"}, status_code=400)
        return JSONResponse({"status": "ok", "name": name})

    async def _secrets_delete(self, request: Request) -> JSONResponse:
        if err := self._check_auth(request):
            return err
        from tsugite.secrets import get_backend

        name = request.path_params["name"]
        try:
            deleted = get_backend().delete(name)
        except NotImplementedError:
            return JSONResponse({"error": "backend does not support deletion"}, status_code=400)
        if not deleted:
            return JSONResponse({"error": "secret not found"}, status_code=404)
        return JSONResponse({"status": "ok", "name": name})
