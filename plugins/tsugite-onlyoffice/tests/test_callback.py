"""The public save callback the document server POSTs to.

The response body is the whole contract: `{"error": 0}` tells the document server
the save landed and it may drop the edit, so every path through the handler is
checked for the answer it gives as well as for what it wrote.
"""

import logging

import httpx
import pytest
from onlyoffice_helpers import DOWNLOAD_URL, JWT_SECRET, callback_url, serve_downloads
from tsugite_onlyoffice import jwt

CALLBACK_URL = "/api/plugins/onlyoffice/callback/notes.docx"
LIVE_KEY = "5f2c9a1b"


@pytest.fixture
def downloads(adapter, typed_bytes):
    """Stand in for the document server's cache so no test needs a real one."""
    return serve_downloads(adapter, typed_bytes)


@pytest.fixture
def untouched(documents_dir):
    """The document as it was before the callback, for the paths that must not write."""
    return (documents_dir / "notes.docx").read_bytes()


def body_for(status, key=LIVE_KEY):
    return {"key": key, "status": status, "url": DOWNLOAD_URL, "users": ["tsugite"]}


def post(client, body, *, document="notes.docx", secret=JWT_SECRET, signs=None):
    url = callback_url(document, signs)
    return client.post(url, json={**body, "token": jwt.sign(body, secret)})


# ── the save path ──


def test_status_2_writes_the_downloaded_bytes_back(client, documents_dir, downloads, typed_bytes):
    resp = post(client, body_for(2))
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"error": 0}
    assert (documents_dir / "notes.docx").read_bytes() == typed_bytes
    assert downloads == [DOWNLOAD_URL]


def test_status_6_persists_a_force_save_the_same_way(client, documents_dir, downloads, typed_bytes):
    resp = post(client, body_for(6))
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"error": 0}
    assert (documents_dir / "notes.docx").read_bytes() == typed_bytes


def test_a_nested_document_is_written_in_place(client, documents_dir, downloads, typed_bytes):
    resp = post(client, body_for(2), document="reports/q1.docx")
    assert resp.status_code == 200, resp.text
    assert (documents_dir / "reports" / "q1.docx").read_bytes() == typed_bytes


def test_the_save_leaves_no_temporary_file_behind(client, documents_dir, downloads):
    post(client, body_for(2))
    assert sorted(p.name for p in documents_dir.iterdir()) == ["notes.docx", "readme.txt", "reports"]


# ── statuses that must not write ──


@pytest.mark.parametrize("status", [1, 4])
def test_editing_and_closed_statuses_write_nothing(client, documents_dir, downloads, untouched, status):
    resp = post(client, body_for(status))
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"error": 0}
    assert (documents_dir / "notes.docx").read_bytes() == untouched
    assert downloads == []


@pytest.mark.parametrize("status", [3, 7])
def test_save_error_statuses_are_acknowledged_and_logged(client, documents_dir, downloads, untouched, status, caplog):
    with caplog.at_level(logging.WARNING):
        resp = post(client, body_for(status))
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"error": 0}
    assert (documents_dir / "notes.docx").read_bytes() == untouched
    assert downloads == []
    assert "notes.docx" in caplog.text


def test_a_failing_download_reports_an_error_and_leaves_the_file_alone(client, documents_dir, adapter, untouched):
    serve_downloads(adapter, lambda request: httpx.Response(502))
    resp = post(client, body_for(2))
    assert resp.status_code == 200, resp.text
    assert resp.json()["error"] != 0
    assert (documents_dir / "notes.docx").read_bytes() == untouched


def test_a_save_that_cannot_be_written_names_no_server_path(client, documents_dir, downloads):
    """The document server reads the integer and nothing else, and this caller is unauthenticated."""
    resp = post(client, body_for(2), document="no-such-directory/deep.docx")
    assert resp.status_code == 200, resp.text
    assert resp.json()["error"] != 0
    assert str(documents_dir) not in resp.text, resp.text


# ── saves from a session an agent turn moved past ──


def test_a_save_on_a_key_a_turn_moved_past_is_acknowledged_and_not_written(
    client, adapter, documents_dir, downloads, untouched
):
    """A session outlives the turn that force-saved it, still holding the bytes that turn
    replaced, so writing its next save back would undo the agent's edit. Refusing with
    anything but a zero would only have the document server retry until it did."""
    adapter.sessions._state("notes.docx").key = "the-key-the-turn-rotated-to"

    resp = post(client, body_for(2, key="the-key-the-editor-is-still-on"))
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"error": 0}
    assert (documents_dir / "notes.docx").read_bytes() == untouched
    assert downloads == []


def test_a_refused_save_retires_its_own_session_and_not_the_live_one(client, adapter, documents_dir, untouched):
    """The refusal drops the payload, not the news that this session closed.

    Losing the lifecycle with the bytes leaves the closed session live forever, and
    stops the generation that retires its key from ever counting again, so the next
    config hands out a key the document server has already closed. Reading it onto
    the document as a whole is the other half of the same mistake: the close of a
    session a turn moved past would declare the tab that replaced it dead.
    """
    rotated = "the-key-the-turn-rotated-to"
    state = adapter.sessions._state("notes.docx")
    adapter.sessions.session_started("notes.docx", LIVE_KEY)
    state.key = rotated
    adapter.sessions.session_started("notes.docx", rotated)
    generation = state.generation

    resp = post(client, body_for(2, key=LIVE_KEY))
    assert resp.json() == {"error": 0}, resp.text
    assert (documents_dir / "notes.docx").read_bytes() == untouched, "the edit still stands"
    assert state.live_keys == {rotated}, "the session that closed is the one that was retired"
    assert state.generation == generation + 1, "and its key is retired with it"


def test_a_save_from_a_session_that_outlived_the_daemon_is_written(
    client, adapter, documents_dir, downloads, typed_bytes
):
    """A restart forgets the key it issued while the tab it issued to keeps editing. There is
    nothing to compare that tab's save against, and refusing it would lose the save outright."""
    adapter.sessions.session_started("notes.docx", None)
    assert adapter.sessions._state("notes.docx").key is None

    resp = post(client, body_for(2))
    assert resp.json() == {"error": 0}, resp.text
    assert (documents_dir / "notes.docx").read_bytes() == typed_bytes


def test_a_save_that_names_no_key_is_written(client, adapter, documents_dir, downloads, typed_bytes):
    """Nothing to compare is not a mismatch: a body carrying no key names no session to refuse."""
    adapter.sessions._state("notes.docx").key = LIVE_KEY

    resp = post(client, {"status": 2, "url": DOWNLOAD_URL, "users": ["tsugite"]})
    assert resp.json() == {"error": 0}, resp.text
    assert (documents_dir / "notes.docx").read_bytes() == typed_bytes


# ── authentication ──


def test_an_unsigned_callback_is_rejected(client, documents_dir, downloads, untouched):
    resp = client.post(callback_url("notes.docx"), json=body_for(2))
    assert resp.status_code == 401, resp.text
    assert (documents_dir / "notes.docx").read_bytes() == untouched
    assert downloads == []


def test_a_callback_with_no_doc_token_is_rejected(client, documents_dir, downloads, untouched):
    """The body's own token names no document, so a signed body alone is not enough."""
    signed = body_for(2)
    resp = client.post(CALLBACK_URL, json={**signed, "token": jwt.sign(signed, JWT_SECRET)})
    assert resp.status_code == 401, resp.text
    assert (documents_dir / "notes.docx").read_bytes() == untouched
    assert downloads == []


def test_a_callback_minted_for_one_document_is_refused_at_another(client, documents_dir, downloads):
    """A valid save for A, POSTed verbatim at B's path, is refused rather than overwriting B."""
    before = (documents_dir / "reports" / "q1.docx").read_bytes()
    resp = post(client, body_for(2), document="reports/q1.docx", signs="notes.docx")
    assert resp.status_code == 401, resp.text
    assert (documents_dir / "reports" / "q1.docx").read_bytes() == before
    assert downloads == []


def test_a_callback_signed_with_another_secret_is_rejected(client, documents_dir, downloads, untouched):
    resp = post(client, body_for(2), secret="not-the-shared-secret")
    assert resp.status_code == 401, resp.text
    assert (documents_dir / "notes.docx").read_bytes() == untouched
    assert downloads == []


@pytest.mark.parametrize(
    "tampered",
    [{"status": 6}, {"url": "https://elsewhere.example.test/payload.docx"}, {"key": "a-key-from-another-session"}],
)
def test_a_token_cannot_be_replayed_over_a_body_it_did_not_sign(client, documents_dir, downloads, untouched, tampered):
    """Every field the handler acts on has to be signed, or one observed token retargets the save."""
    signed = body_for(2)
    resp = client.post(callback_url("notes.docx"), json={**signed, **tampered, "token": jwt.sign(signed, JWT_SECRET)})
    assert resp.status_code == 401, resp.text
    assert (documents_dir / "notes.docx").read_bytes() == untouched
    assert downloads == []


def test_a_file_url_token_is_not_a_callback_token(client, documents_dir, downloads, untouched):
    """It signs no status, and a field-by-field check would match that against a body carrying none."""
    resp = client.post(callback_url("notes.docx"), json={"token": jwt.sign({"document": "notes.docx"}, JWT_SECRET)})
    assert resp.status_code == 401, resp.text
    assert (documents_dir / "notes.docx").read_bytes() == untouched
    assert downloads == []


def test_the_file_urls_token_is_not_a_doc_token(client, headers, documents_dir, downloads, untouched):
    """The two are minted from one claim set, so each route was accepting the other's token."""
    config = client.get("/api/plugins/onlyoffice/config?path=notes.docx", headers=headers).json()["config"]
    lifted = config["document"]["url"].partition("?token=")[2]
    signed = body_for(2, key=config["document"]["key"])
    resp = client.post(f"{CALLBACK_URL}?doc_token={lifted}", json={**signed, "token": jwt.sign(signed, JWT_SECRET)})
    assert resp.status_code == 401, resp.text
    assert (documents_dir / "notes.docx").read_bytes() == untouched
    assert downloads == []


def test_the_header_form_of_the_token_is_accepted(client, documents_dir, downloads, typed_bytes):
    # The header form signs one wrapper level more than the body form does.
    body = body_for(2)
    resp = client.post(
        callback_url("notes.docx"),
        json=body,
        headers={"Authorization": "Bearer " + jwt.sign({"payload": body}, JWT_SECRET)},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"error": 0}
    assert (documents_dir / "notes.docx").read_bytes() == typed_bytes


def test_a_callback_for_a_path_outside_the_documents_dir_is_rejected(client, tmp_path, downloads):
    outside = (tmp_path / "outside.docx").read_bytes()
    resp = post(client, body_for(2), document="%2e%2e%2foutside.docx")
    assert resp.status_code == 403, resp.text
    assert "outside.docx" not in resp.text, "the refusal is logged, not handed back"
    assert (tmp_path / "outside.docx").read_bytes() == outside
    assert downloads == []


def test_a_callback_cannot_write_a_file_that_is_not_a_document(client, documents_dir, downloads):
    """The write path resolves paths that need not exist yet, which is how a new name gets in.

    The download is wired and would succeed, so the refusal is the extension and
    nothing else.
    """
    resp = post(client, body_for(2), document="planted.sh")
    assert resp.json()["error"] != 0, resp.text
    assert not (documents_dir / "planted.sh").exists()
    assert downloads == [], "nothing should have been fetched for a name we will not write"
