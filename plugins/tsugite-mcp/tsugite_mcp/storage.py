"""OAuth tokens, client registrations and user-entered clients for one server, kept in tsugite's secret store."""

import asyncio
import json
import time

from mcp.shared.auth import OAuthClientInformationFull, OAuthToken

from tsugite.secrets import get_backend
from tsugite.secrets.registry import get_registry

NO_REDIRECT = "http://localhost/"


class SecretStorage:
    """The SDK's `TokenStorage` for server `name`.

    Tokens are saved with the time they expire, because the SDK keeps expiry only in memory.
    Every token handed out is registered with the masking registry first.
    """

    def __init__(self, name: str):
        self.name = name
        self.tokens_key = f"mcp-{name}-tokens"
        self.client_key = f"mcp-{name}-client"
        # A client the user registered in the provider's console: client_id, client_secret, auth_method.
        self.preset_key = f"mcp-{name}-client-preset"
        self.using_preset = False

    async def _get(self, key: str) -> dict | None:
        raw = await asyncio.to_thread(get_backend().get, key)
        return json.loads(raw) if raw else None

    async def _set(self, key: str, record: dict) -> None:
        await asyncio.to_thread(get_backend().set, key, json.dumps(record))

    async def delete_all(self) -> None:
        for key in (self.tokens_key, self.client_key, self.preset_key):
            await asyncio.to_thread(get_backend().delete, key)

    async def delete_registration(self) -> None:
        for key in (self.tokens_key, self.client_key):
            await asyncio.to_thread(get_backend().delete, key)

    async def get_preset(self) -> dict | None:
        record = await self._get(self.preset_key)
        if record and record.get("client_secret"):
            get_registry().register(f"{self.preset_key}-secret", record["client_secret"], agent="mcp")
        return record

    async def set_preset(self, record: dict | None) -> None:
        if record is None:
            await asyncio.to_thread(get_backend().delete, self.preset_key)
        else:
            if record.get("client_secret"):
                get_registry().register(f"{self.preset_key}-secret", record["client_secret"], agent="mcp")
            await self._set(self.preset_key, record)

    async def expires_at(self) -> float | None:
        record = await self._get(self.tokens_key)
        return record and record.get("expires_at")

    async def get_tokens(self) -> OAuthToken | None:
        record = await self._get(self.tokens_key)
        if record is None:
            return None
        self._mask(record)
        return OAuthToken.model_validate(record)

    async def set_tokens(self, tokens: OAuthToken) -> None:
        previous = await self._get(self.tokens_key)
        record = tokens.model_dump(mode="json", exclude_none=True)
        # A refresh response may leave out the refresh token, which then stays valid.
        if previous and "refresh_token" not in record and "refresh_token" in previous:
            record["refresh_token"] = previous["refresh_token"]
        if tokens.expires_in is not None:
            record["expires_at"] = time.time() + tokens.expires_in
        self._mask(record)
        await self._set(self.tokens_key, record)

    async def get_client_info(self) -> OAuthClientInformationFull | None:
        """The user's own client when one is set, else the dynamic registration."""
        preset = await self.get_preset()
        self.using_preset = preset is not None
        if preset:
            return OAuthClientInformationFull(
                client_id=preset["client_id"],
                client_secret=preset.get("client_secret"),
                redirect_uris=[NO_REDIRECT],
                token_endpoint_auth_method=preset.get("auth_method")
                or ("client_secret_basic" if preset.get("client_secret") else "none"),
            )
        return await self.get_registration()

    async def get_registration(self) -> OAuthClientInformationFull | None:
        record = await self._get(self.client_key)
        if record is None:
            return None
        info = OAuthClientInformationFull.model_validate(record)
        if info.client_secret:
            get_registry().register(f"{self.client_key}-secret", info.client_secret, agent="mcp")
        return info

    async def set_client_info(self, client_info: OAuthClientInformationFull) -> None:
        await self._set(self.client_key, client_info.model_dump(mode="json", exclude_none=True))

    def _mask(self, record: dict) -> None:
        for field in ("access_token", "refresh_token"):
            if field in record:
                get_registry().register(f"{self.tokens_key}-{field}", record[field], agent="mcp")
