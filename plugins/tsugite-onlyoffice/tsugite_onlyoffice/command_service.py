"""The document server's CommandService, which is how the daemon steers a session.

Nothing here touches documents. `forcesave` asks the document server to hand back
what a live session currently holds, and `version` is a liveness probe.
"""

from __future__ import annotations

import httpx

from tsugite_onlyoffice import jwt

COMMAND_PATH = "/coauthoring/CommandService.ashx"
TIMEOUT = 10.0

# Codes that report there was nothing for the command to do. The document server
# spends them like errors, but each one describes the state the caller was asking
# for, so raising would turn a completed request into a failure.
NO_SUCH_SESSION = 1
NOTHING_TO_SAVE = 4


class CommandServiceError(RuntimeError):
    """The document server answered a command with a non-zero error code."""

    def __init__(self, command: str, code: int) -> None:
        super().__init__(f"onlyoffice CommandService rejected {command!r} with error {code}")
        self.command = command
        self.code = code


class CommandClient:
    """Client for one document server's CommandService.

    `key` is always the document key the editing session was opened with. A command
    the document server refuses raises `CommandServiceError`; a code that only
    reports there was nothing to do comes back as False instead.
    """

    def __init__(self, server_url: str, secret: str, client: httpx.AsyncClient) -> None:
        self._url = server_url.rstrip("/") + COMMAND_PATH
        self._secret = secret
        self._client = client

    async def version(self) -> str:
        """The document server's version string."""
        return (await self._post({"c": "version"})).get("version", "")

    async def forcesave(self, key: str) -> bool:
        """Ask the document server to save what the session on `key` currently holds.

        Returns:
            True when the save is on its way as a status 6 callback, False when there
            was nothing to save. That covers a session with nothing new, and also a
            key no session is on: once a turn rotates the key, the live editor is
            still on the previous one until it finishes swapping, and a key nobody is
            editing holds nothing the file on disk does not already have.
        """
        return await self._post({"c": "forcesave", "key": key}, benign={NOTHING_TO_SAVE, NO_SUCH_SESSION}) is not None

    async def _post(self, body: dict, benign: frozenset[int] | set[int] = frozenset()) -> dict | None:
        """Sign a command both ways the document server accepts, and send it.

        Returns:
            The answer, or None when the document server reported one of `benign`.

        Raises:
            CommandServiceError: The document server refused the command.
        """
        payload = {**body, "token": jwt.sign(body, self._secret)}
        headers = {"Authorization": "Bearer " + jwt.sign({"payload": body}, self._secret)}
        response = await self._client.post(self._url, json=payload, headers=headers, timeout=TIMEOUT)
        response.raise_for_status()
        answer = response.json()
        code = answer.get("error")
        if code in benign:
            return None
        if code:
            raise CommandServiceError(body["c"], code)
        return answer
