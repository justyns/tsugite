"""Reading a document a human is still typing in, and saying why that failed.

Nothing here talks to a document server: the CommandService side is an httpx
mock transport, and the callback it would send arrives through the app itself.
"""

import json

import httpx
import pytest
from onlyoffice_helpers import (
    JWT_SECRET,
    PUBLIC_BASE_URL,
    SERVER_URL,
    FakeCommands,
    asgi_client,
    callback_body,
    post_callback,
    serve_downloads,
)
from tsugite_onlyoffice import jwt
from tsugite_onlyoffice.command_service import CommandClient, CommandServiceError

DOCUMENT = "notes.docx"


def command_client(handler):
    """A real CommandClient whose requests land in `handler` instead of the network."""
    return CommandClient(SERVER_URL, JWT_SECRET, httpx.AsyncClient(transport=httpx.MockTransport(handler)))


def recorder(answer=None):
    """A mock transport handler that records each request's headers and body."""
    sent = []

    def handle(request: httpx.Request) -> httpx.Response:
        sent.append({"headers": request.headers, "body": json.loads(request.content)})
        return httpx.Response(200, json=answer if answer is not None else {"error": 0})

    return sent, handle


def waiters(adapter, relative):
    """The reads currently parked on a document, straight out of the adapter's state."""
    return adapter.sessions._state(relative).waiters


# ── the outbound command ──


@pytest.mark.asyncio
async def test_forcesave_names_the_command_and_the_document_key():
    sent, handle = recorder()
    await command_client(handle).forcesave("abc123")
    assert [(call["body"]["c"], call["body"]["key"]) for call in sent] == [("forcesave", "abc123")]


@pytest.mark.asyncio
async def test_a_forcesave_with_nothing_to_save_is_an_answer_not_a_failure():
    """Error 4 means the session had no changes, so the file on disk is already what it holds.

    The first read of a live document force-saves it, which is what leaves the
    second read nothing to save, so this is the ordinary shape of a turn rather
    than an edge case.
    """
    _sent, handle = recorder({"error": 4})
    assert await command_client(handle).forcesave("abc123") is False


@pytest.mark.asyncio
async def test_version_returns_what_the_document_server_reports():
    _sent, handle = recorder({"error": 0, "version": "9.3.1.10"})
    assert await command_client(handle).version() == "9.3.1.10"


@pytest.mark.asyncio
async def test_the_command_goes_to_the_command_service_endpoint():
    requests = []

    def handle(request):
        requests.append(str(request.url))
        return httpx.Response(200, json={"error": 0})

    await command_client(handle).forcesave("abc123")
    assert requests == [f"{SERVER_URL}/coauthoring/CommandService.ashx"]


@pytest.mark.asyncio
async def test_the_body_token_signs_the_body_it_travels_in():
    sent, handle = recorder()
    await command_client(handle).forcesave("abc123")
    (call,) = sent
    signed = jwt.verify(call["body"]["token"], JWT_SECRET)
    assert {k: v for k, v in signed.items() if k != "exp"} == {k: v for k, v in call["body"].items() if k != "token"}


@pytest.mark.asyncio
async def test_the_header_token_wraps_the_same_body_one_level_deeper():
    sent, handle = recorder()
    await command_client(handle).forcesave("abc123")
    (call,) = sent
    token = call["headers"]["authorization"].removeprefix("Bearer ")
    assert jwt.verify(token, JWT_SECRET)["payload"]["c"] == "forcesave"


@pytest.mark.asyncio
async def test_a_non_zero_error_from_the_command_service_raises_with_the_code():
    _sent, handle = recorder({"error": -6})
    with pytest.raises(CommandServiceError) as raised:
        await command_client(handle).forcesave("abc123")
    assert raised.value.code == -6
    assert "-6" in str(raised.value)


# ── the live read ──


@pytest.fixture
def live(adapter, http_server, typed_bytes):
    """Wire the adapter to a fake CommandService whose forcesave answers itself.

    The forcesave posts the status-6 callback the document server would post,
    which is the only thing that can resolve a waiting read.
    """
    serve_downloads(adapter, typed_bytes)

    async def answer(_command, key):
        await post_callback(http_server, DOCUMENT, callback_body(6, key))

    adapter.commands = FakeCommands(answer=answer)
    return adapter.commands


@pytest.mark.asyncio
async def test_a_live_read_force_saves_the_document(adapter, live):
    await adapter.sessions.read_live(DOCUMENT)
    assert [name for name, _key in live.calls] == ["forcesave"]


@pytest.mark.asyncio
async def test_the_force_save_carries_the_key_the_editor_was_given(adapter, live, http_server, headers):
    """A forcesave on any other key names a session the document server does not have."""
    async with asgi_client(http_server) as client:
        response = await client.get(f"/api/plugins/onlyoffice/config?path={DOCUMENT}", headers=headers)
    issued = response.json()["config"]["document"]["key"]

    await adapter.sessions.read_live(DOCUMENT)
    assert live.calls == [("forcesave", issued)]


@pytest.mark.asyncio
async def test_the_force_save_is_persisted_like_any_other_save(adapter, live, documents_dir, typed_bytes):
    """A live read is only a read to the agent; on disk it is the session's own save."""
    await adapter.sessions.read_live(DOCUMENT)
    assert (documents_dir / DOCUMENT).read_bytes() == typed_bytes


@pytest.mark.asyncio
async def test_a_second_live_read_does_not_wait_for_a_save_that_is_not_coming(adapter, monkeypatch):
    """The first read force-saves the session, so the next one has nothing left to save.

    Read, decide, read again is the ordinary shape of a turn, so a read that
    treats "nothing to save" as a failure breaks the common path rather than a
    corner of it.
    """
    import tsugite_onlyoffice.sessions as sessions_module

    monkeypatch.setattr(sessions_module, "SAVE_TIMEOUT", 0.05)
    adapter.commands = FakeCommands(nothing_to_do={"forcesave"})

    await adapter.sessions.read_live(DOCUMENT)
    assert waiters(adapter, DOCUMENT) == []


@pytest.mark.asyncio
async def test_a_live_read_nobody_answers_times_out_and_leaves_no_waiter(adapter, monkeypatch):
    import tsugite_onlyoffice.sessions as sessions_module

    monkeypatch.setattr(sessions_module, "SAVE_TIMEOUT", 0.05)
    adapter.commands = FakeCommands()

    with pytest.raises(RuntimeError) as raised:
        await adapter.sessions.read_live(DOCUMENT)
    assert DOCUMENT in str(raised.value)
    assert waiters(adapter, DOCUMENT) == []


@pytest.mark.asyncio
async def test_a_live_read_of_a_path_outside_the_documents_dir_is_refused(adapter):
    adapter.commands = FakeCommands()
    with pytest.raises(ValueError):
        await adapter.sessions.read_live("../outside.docx")


# ── the reachability diagnostic ──


def health_of(client, adapter, headers, answer=None, error=None):
    """Ask /health with the document server answering `answer`, or refusing to connect."""

    def handle(request):
        if error is not None:
            raise error
        return httpx.Response(200, json=answer)

    adapter.commands = command_client(handle)
    response = client.get("/api/plugins/onlyoffice/health", headers=headers)
    assert response.status_code == 200, response.text
    return response.json()


def test_health_requires_a_daemon_token(client):
    assert client.get("/api/plugins/onlyoffice/health").status_code == 401


def test_health_reports_a_document_server_that_answers(client, adapter, headers):
    report = health_of(client, adapter, headers, {"error": 0, "version": "9.3.1.10"})
    assert report["ok"] is True
    assert report["version"] == "9.3.1.10"
    assert SERVER_URL in report["message"]


def test_health_names_public_base_url_when_the_document_server_will_not_fetch_it(client, adapter, headers):
    """A bare -4 is the document server declining to make the request at all."""
    report = health_of(client, adapter, headers, {"error": -4})
    assert report["ok"] is False
    assert "public_base_url" in report["message"]
    assert PUBLIC_BASE_URL in report["message"]
    assert "-4" in report["message"]
    assert "did not attempt" in report["message"]


def test_health_names_public_base_url_when_the_document_server_rejects_the_command(client, adapter, headers):
    report = health_of(client, adapter, headers, {"error": -6})
    assert report["ok"] is False
    assert "public_base_url" in report["message"]
    assert "fetches" in report["message"]
    assert "-6" in report["message"]


def test_health_reports_a_document_server_it_cannot_reach(client, adapter, headers):
    report = health_of(client, adapter, headers, error=httpx.ConnectError("no route to host"))
    assert report["ok"] is False
    assert SERVER_URL in report["message"]
    assert "server_url" in report["message"]
