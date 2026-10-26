"""ProvidersMixin: provider status, CLI binary paths and a connection test.

API keys are written through the secrets routes under each provider's ``key.secret`` name.
"""

import asyncio

from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from tsugite.config import update_config
from tsugite.providers import get_provider, list_all_providers
from tsugite.providers.base import api_key_secret_name, configured_binary, resolve_api_key, resolve_binary
from tsugite.providers.model_registry import get_model_info, list_models
from tsugite.secrets import get_backend

TEST_TIMEOUT_SECONDS = 60


def _provider_status(name: str, secret_names: list[str] | Exception) -> dict:
    try:
        provider = get_provider(name)
        binary = None
        if default := getattr(provider, "binary", None):
            binary = {
                "default": default,
                "configured": configured_binary(name),
                "resolved": resolve_binary(name, default),
            }

        key = None
        if env := getattr(provider, "api_key_env", None):
            if isinstance(secret_names, Exception):
                raise secret_names
            secret = api_key_secret_name(env)
            is_set = bool(resolve_api_key(env))
            source = ("secret" if secret in secret_names else "env") if is_set else None
            key = {"env": env, "secret": secret, "set": is_set, "source": source}
    except Exception as e:  # noqa: BLE001 -- reported in the row's error field
        return {"name": name, "binary": None, "key": None, "error": str(e) or type(e).__name__}
    return {"name": name, "binary": binary, "key": key}


def _statuses(names: list[str]) -> list[dict]:
    try:
        secret_names = get_backend().list_names()
    except Exception as e:  # noqa: BLE001 -- reported on each row with a key
        secret_names = e
    return [_provider_status(n, secret_names) for n in names]


async def _pick_test_model(name: str, provider) -> str | None:
    # provider.list_models() returns [] on an auth failure. The registry fallback still picks a model,
    # and the test call reports the error.
    models = await provider.list_models() or [
        key.partition("/")[2] for key in list_models() if key.startswith(f"{name}/")
    ]
    costs = {
        m: info.input_cost_per_million
        for m in models
        if (info := get_model_info(name, m)) is not None and info.input_cost_per_million is not None
    }
    if costs:
        return min(costs, key=costs.get)
    return models[0] if models else None


class ProvidersMixin:
    def _providers_routes(self) -> list:
        return [
            Route("/api/providers", self._providers_list, methods=["GET"]),
            Route("/api/providers/{name}", self._providers_update, methods=["PUT"]),
            Route("/api/providers/{name}/test", self._providers_test, methods=["POST"]),
        ]

    async def _providers_list(self, request: Request) -> JSONResponse:
        if err := self._check_auth(request):
            return err
        return JSONResponse({"providers": await asyncio.to_thread(_statuses, list_all_providers())})

    async def _providers_update(self, request: Request) -> JSONResponse:
        if err := self._check_auth(request):
            return err
        name = request.path_params["name"]
        if name not in list_all_providers():
            return JSONResponse({"error": f"unknown provider: {name}"}, status_code=404)
        if not getattr(get_provider(name), "binary", None):
            return JSONResponse({"error": f"{name} has no binary to configure"}, status_code=400)

        body = await request.json()
        if not isinstance(body, dict) or not isinstance(body.get("binary"), str | None):
            return JSONResponse({"error": "body must be an object with a string binary"}, status_code=400)
        binary = (body.get("binary") or "").strip()

        def apply(cfg):
            entry = cfg.plugins.setdefault(name, {})
            if binary:
                entry["binary"] = binary
            else:
                entry.pop("binary", None)

        update_config(None, apply)
        [status] = await asyncio.to_thread(_statuses, [name])
        return JSONResponse(status)

    async def _providers_test(self, request: Request) -> JSONResponse:
        if err := self._check_auth(request):
            return err
        name = request.path_params["name"]
        if name not in list_all_providers():
            return JSONResponse({"error": f"unknown provider: {name}"}, status_code=404)

        provider = model = None
        try:
            provider = get_provider(name)
            async with asyncio.timeout(TEST_TIMEOUT_SECONDS):
                model = await _pick_test_model(name, provider)
                if not model:
                    return JSONResponse({"ok": False, "model": None, "error": f"{name} lists no models to test with"})
                response = await provider.acompletion(
                    [{"role": "user", "content": "Reply with the word OK."}], model=model
                )
        except Exception as e:  # noqa: BLE001 -- the error text is the test result
            return JSONResponse({"ok": False, "model": model, "error": str(e) or type(e).__name__})
        finally:
            if provider is not None and not getattr(provider, "cacheable", True):
                await provider.stop()
        return JSONResponse({"ok": True, "model": model, "reply": response.content})
