"""The daemon-authed OnlyOffice surface: document listing, signed editor config, UI descriptor.

These are protocol boundaries: the web UI reads the surface descriptor, and the
document server reads the config the /config route signs, so both shapes are
asserted field by field rather than smoke-tested.
"""

import re

from onlyoffice_helpers import (
    AGENT_NAME,
    JWT_SECRET,
    PLAIN_DOCUMENT,
    PUBLIC_BASE_URL,
    SERVER_URL,
    build_docx,
    callback_body,
    callback_url,
)

CONFIG_URL = "/api/plugins/onlyoffice/config?path=notes.docx"
FILE_URL = "/api/plugins/onlyoffice/file/"
EDITOR_URL = "/api/plugins/onlyoffice/ui/editor.html"


def editor_page(client):
    resp = client.get(EDITOR_URL)
    assert resp.status_code == 200, resp.text
    return resp.text


def config_of(client, headers, relative="notes.docx"):
    resp = client.get(f"/api/plugins/onlyoffice/config?path={relative}", headers=headers)
    assert resp.status_code == 200, resp.text
    return resp.json()["config"]


def ui_theme(client, headers, theme):
    resp = client.get(f"{CONFIG_URL}&theme={theme}", headers=headers)
    assert resp.status_code == 200, resp.text
    return resp.json()["config"]["editorConfig"]["customization"]["uiTheme"]


def file_token(document, secret=JWT_SECRET):
    from tsugite_onlyoffice import jwt
    from tsugite_onlyoffice.adapter import USE_FILE

    return jwt.sign({"document": document, "use": USE_FILE}, secret)


def session(client, status, key, relative="notes.docx"):
    """Report one of the session lifecycle statuses the document server posts back."""
    resp = client.post(callback_url(relative), json=callback_body(status, key, url=None))
    assert resp.json() == {"error": 0}, resp.text


# ── auth ──


def test_config_route_requires_a_daemon_token(client):
    assert client.get(CONFIG_URL).status_code == 401


def test_docs_route_requires_a_daemon_token(client):
    assert client.get("/api/plugins/onlyoffice/docs").status_code == 401


# ── GET /docs ──


def test_docs_lists_every_docx_and_nothing_else(client, headers, documents_dir):
    resp = client.get("/api/plugins/onlyoffice/docs", headers=headers)
    assert resp.status_code == 200, resp.text
    documents = resp.json()["documents"]
    assert [d["path"] for d in documents] == ["notes.docx", "reports/q1.docx"]
    assert documents[0]["size"] == (documents_dir / "notes.docx").stat().st_size
    assert documents[0]["modified"].startswith("20")


# ── GET /config ──


def test_config_describes_the_document_for_the_editor(client, headers):
    resp = client.get(CONFIG_URL, headers=headers)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["server_url"] == SERVER_URL, "the page needs this to load api.js"
    config = body["config"]
    assert config["documentType"] == "word"
    assert config["document"]["fileType"] == "docx"
    assert config["document"]["title"] == "notes.docx"
    assert config["document"]["permissions"] == {"edit": True, "comment": True}
    assert config["editorConfig"]["lang"] == "en"


def test_config_opens_the_document_as_the_human_who_asked_for_it(client, headers):
    """Every browser session opening as the agent makes both authors one participant."""
    resp = client.get(f"{CONFIG_URL}&user=web-user-1", headers=headers)
    assert resp.status_code == 200, resp.text
    assert resp.json()["config"]["editorConfig"]["user"] == {"id": "web-user-1", "name": "web-user-1"}


def test_config_without_a_user_still_is_not_the_agent(client, headers):
    """A comment the human leaves must not be attributable to the tools' author."""
    user = config_of(client, headers)["editorConfig"]["user"]
    assert user["id"] != "tsugite"
    assert user["name"] != AGENT_NAME


def test_config_opens_the_editor_with_its_ribbon_collapsed(client, headers):
    """The surface docks beside a chat, so the editor opens as small as the licence allows."""
    customization = config_of(client, headers)["editorConfig"]["customization"]
    assert customization["compactToolbar"] is True
    assert customization["hideRulers"] is True
    # Hiding the side panels needs the Developer Edition white-label licence, so
    # sending `layout` would be config a Community Edition server silently drops.
    assert "layout" not in customization


def test_config_matches_the_editor_theme_to_the_tsugite_one(client, headers):
    """The editor fills the tab, so a mismatch is a bright rectangle inside a dark UI."""
    assert ui_theme(client, headers, "latte") == "theme-light", "latte is the only light tsugite theme"
    for dark in ("mocha", "macchiato", "frappe", "gruvbox"):
        assert ui_theme(client, headers, dark) == "theme-dark", dark


def test_config_without_a_theme_opens_dark(client, headers):
    """The web UI defaults to mocha, so dark is what a page that did not say is showing."""
    assert config_of(client, headers)["editorConfig"]["customization"]["uiTheme"] == "theme-dark"


def test_config_urls_are_absolute_and_use_the_public_base_url(client, headers):
    """The document server fetches these itself, so a daemon bind address is no substitute."""
    config = config_of(client, headers, "reports/q1.docx")
    prefix = f"{PUBLIC_BASE_URL}/api/plugins/onlyoffice"
    assert config["document"]["url"].startswith(f"{prefix}/file/reports/q1.docx?token=")
    assert config["editorConfig"]["callbackUrl"].startswith(f"{prefix}/callback/reports/q1.docx?doc_token=")


def test_config_hands_back_the_one_spelling_it_signed_everything_for(client, headers):
    """Both URLs travel to the document server, and any normalising client or proxy
    fetches `/file/notes.docx`, which a token minted for `./notes.docx` refuses. The
    announce carries the canonical spelling too, so a page holding the caller's own
    never matches one."""
    body = client.get("/api/plugins/onlyoffice/config?path=./notes.docx", headers=headers).json()
    prefix = f"{PUBLIC_BASE_URL}/api/plugins/onlyoffice"
    assert body["config"]["document"]["url"].startswith(f"{prefix}/file/notes.docx?token=")
    assert body["config"]["editorConfig"]["callbackUrl"].startswith(f"{prefix}/callback/notes.docx?doc_token=")
    assert body["path"] == "notes.docx"


def test_config_token_signs_the_config_it_travels_with(client, headers):
    from tsugite_onlyoffice import jwt

    config = config_of(client, headers)
    claims = jwt.verify(config["token"], JWT_SECRET)
    assert {k: v for k, v in claims.items() if k != "exp"} == {k: v for k, v in config.items() if k != "token"}


def test_file_url_token_names_the_document_it_was_minted_for(client, headers):
    from tsugite_onlyoffice import jwt

    config = config_of(client, headers, "reports/q1.docx")
    token = config["document"]["url"].partition("?token=")[2]
    assert jwt.verify(token, JWT_SECRET)["document"] == "reports/q1.docx"


def test_callback_url_token_names_the_document_and_does_not_expire(client, headers):
    """A callback arrives whenever the editor closes, so an expiry here would fail a real save."""
    from tsugite_onlyoffice import jwt

    config = config_of(client, headers, "reports/q1.docx")
    token = config["editorConfig"]["callbackUrl"].partition("?doc_token=")[2]
    claims = jwt.verify(token, JWT_SECRET)
    assert claims["document"] == "reports/q1.docx"
    assert "exp" not in claims


def test_document_key_survives_the_charset_the_document_server_allows(client, headers):
    key = config_of(client, headers)["document"]["key"]
    assert re.fullmatch(r"[A-Za-z0-9_-]{1,128}", key), key


def test_document_key_is_stable_until_the_bytes_change(client, headers, documents_dir):
    """The document server caches by key, so a stale key would serve stale bytes."""
    first = config_of(client, headers)["document"]["key"]
    assert config_of(client, headers)["document"]["key"] == first
    (documents_dir / "notes.docx").write_bytes(b"notes-v2-with-more-text")
    assert config_of(client, headers)["document"]["key"] != first


def test_the_key_holds_for_as_long_as_the_session_is_live(client, headers, documents_dir):
    """Two tabs on one document have to meet in one editing session.

    A force-save writes what the session holds back to disk mid-session, so a key
    read off the file alone would send the second tab into a session of its own.
    """
    opened = config_of(client, headers)["document"]["key"]
    session(client, 1, opened)
    (documents_dir / "notes.docx").write_bytes(b"what the force-save handed back")
    assert config_of(client, headers)["document"]["key"] == opened


def test_a_key_a_closed_session_retired_is_never_handed_out_again(client, headers):
    """Closing leaves the file exactly as it was, so a key read off the file alone comes
    back the same one the closed session retired. The document server refuses to reopen
    that, and nothing the page can do gets off it, a reload included."""
    opened = config_of(client, headers)["document"]["key"]
    session(client, 1, opened)
    session(client, 4, opened)

    reopened = config_of(client, headers)["document"]["key"]
    assert reopened != opened
    assert config_of(client, headers)["document"]["key"] == reopened, "and the tab can reload onto it"


def test_config_rejects_a_path_outside_the_documents_dir(client, headers):
    resp = client.get("/api/plugins/onlyoffice/config?path=../outside.docx", headers=headers)
    assert resp.status_code == 403, resp.text


def test_config_without_a_path_is_a_bad_request(client, headers):
    assert client.get("/api/plugins/onlyoffice/config", headers=headers).status_code == 400


def test_config_for_an_unknown_document_is_a_404(client, headers):
    assert client.get("/api/plugins/onlyoffice/config?path=missing.docx", headers=headers).status_code == 404


# ── GET /file: the public route document.url points at ──


def test_file_route_serves_the_document_at_the_url_the_config_minted(client, headers, documents_dir):
    url = config_of(client, headers)["document"]["url"]
    resp = client.get(url.replace(PUBLIC_BASE_URL, ""))
    assert resp.status_code == 200, resp.text
    assert resp.content == (documents_dir / "notes.docx").read_bytes()


def test_file_route_accepts_the_token_in_the_authorization_header(client, documents_dir):
    """The document server's outbound header is configurable, so both forms have to work."""
    resp = client.get(FILE_URL + "notes.docx", headers={"Authorization": "Bearer " + file_token("notes.docx")})
    assert resp.status_code == 200, resp.text
    assert resp.content == (documents_dir / "notes.docx").read_bytes()


def test_file_route_rejects_a_request_with_no_token(client):
    assert client.get(FILE_URL + "notes.docx").status_code == 401


def test_file_route_rejects_a_token_signed_with_another_secret(client):
    resp = client.get(FILE_URL + "notes.docx?token=" + file_token("notes.docx", "not-the-shared-secret"))
    assert resp.status_code == 401, resp.text


def test_file_route_rejects_a_non_ascii_token_without_raising(client):
    """The handler catches ValueError only, so anything else is an unauthenticated 500."""
    resp = client.get(FILE_URL + "notes.docx?token=" + file_token("notes.docx") + "é")
    assert resp.status_code == 401, resp.text


def test_file_route_rejects_a_token_minted_for_another_document(client):
    resp = client.get(FILE_URL + "reports/q1.docx?token=" + file_token("notes.docx"))
    assert resp.status_code == 401, resp.text


def test_file_route_rejects_the_callback_urls_token(client, headers):
    """That one never expires, so a file route taking it is a permanent read of the document.

    A callback URL is persisted by the document server for the session's life and
    turns up in its logs, so it is the token most likely to be lying around.
    """
    forever = config_of(client, headers)["editorConfig"]["callbackUrl"].partition("?doc_token=")[2]
    assert client.get(FILE_URL + "notes.docx?token=" + forever).status_code == 401


def test_file_route_rejects_a_path_outside_the_documents_dir(client):
    resp = client.get(FILE_URL + "%2e%2e%2foutside.docx?token=" + file_token("../outside.docx"))
    assert resp.status_code == 403, resp.text


def test_file_route_rejects_a_symlink_pointing_out_of_the_documents_dir(client, documents_dir):
    """The jail is a resolved-path check, which a lexical one would pass and a symlink would walk."""
    (documents_dir / "escape.docx").symlink_to(documents_dir.parent / "outside.docx")
    resp = client.get(FILE_URL + "escape.docx?token=" + file_token("escape.docx"))
    assert resp.status_code == 403


def test_file_route_404s_for_a_missing_document(client):
    resp = client.get(FILE_URL + "missing.docx?token=" + file_token("missing.docx"))
    assert resp.status_code == 404, resp.text


# ── the UI surface ──


def test_editor_surface_reaches_the_web_ui_payload(client, headers):
    resp = client.get("/api/plugins", headers=headers)
    assert resp.status_code == 200, resp.text
    surfaces = [s for s in resp.json()["ui_surfaces"] if s.get("plugin") == "onlyoffice"]
    assert len(surfaces) == 1, resp.text
    (surface,) = surfaces
    assert surface["kind"] == "plugin/onlyoffice/doc"
    assert surface["entry"] == "/api/plugins/onlyoffice/ui/editor.html"
    assert surface["label"] == "Document"
    assert surface["icon"] == "files"
    assert surface["nav"] is True
    assert surface["params"] == ["path"]
    assert surface["events"] == ["onlyoffice_document_update"], "the host forwards the page its own events"
    assert surface["mode"] == "workspace", "the editor docks beside a chat instead of taking the region"
    assert "assets" not in surface, "a server-side path must never reach the browser"


def test_editor_page_serves_at_the_entry_url_without_a_token(client):
    """The iframe loads the entry anonymously; the bridge hands it the token afterwards."""
    assert client.get(EDITOR_URL).status_code == 200


def test_editor_page_only_listens_to_its_host_frame(client):
    """`tsugite:init` carries the daemon token and repaints the page, so any framer could drive it."""
    assert "if (event.source !== parent) return;" in editor_page(client)


def test_editor_page_addresses_the_host_rather_than_whoever_framed_it(client):
    """The surface is same-origin with the host, and the daemon sets no frame-ancestors,
    so a page that framed this one is `parent` and collects everything sent to a wildcard."""
    targets = set(re.findall(r"parent\.postMessage\(.*, (.+?)\);", editor_page(client)))
    assert targets == {"location.origin"}, targets


# ── document events, which arrive over the host bridge ──


def test_editor_page_holds_the_spelling_the_daemon_handed_back(client):
    """An announce names the canonical path, and the page drops an event for any other,
    so a tab opened on a caller's own spelling would never hear that an agent edited it."""
    assert "currentPath = payload.path;" in editor_page(client)


def test_editor_page_takes_document_events_from_the_host(client):
    """A browser holds one /api/events stream per origin, and behind a reverse proxy a second
    long-lived request to that origin is never sent at all, so this page reads the host's."""
    page = editor_page(client)
    assert "/api/events" not in page, "a stream of this frame's own never reaches the daemon"
    assert "msg.type === 'tsugite:event'" in page, "the bridge is where a document event arrives"


# ── only documents ──


def test_config_will_not_mint_a_token_for_a_file_that_is_not_a_document(client, headers, documents_dir):
    """The listing filters to .docx, so a caller can only be naming one deliberately.

    A minted token is a bearer credential that travels to the document server and
    into its logs, so minting one for a private file hands that file to a third
    party the deployment only trusts with documents.
    """
    (documents_dir / "secrets.txt").write_text("PRIVATE-abc123")
    resp = client.get("/api/plugins/onlyoffice/config?path=secrets.txt", headers=headers)
    assert resp.status_code == 404, resp.text


def test_the_file_route_refuses_a_non_document_even_with_a_good_token(client, documents_dir):
    """Whoever holds the shared secret can sign any claim, so the check cannot live at minting."""
    (documents_dir / "secrets.txt").write_text("PRIVATE-abc123")
    resp = client.get(f"{FILE_URL}secrets.txt?token={file_token('secrets.txt')}")
    assert resp.status_code == 404, resp.text
    assert "PRIVATE-abc123" not in resp.text


def test_the_listing_and_the_policy_agree_on_a_shouted_extension(client, headers, documents_dir):
    """A document the picker hides but the routes serve is the same split this rule closed."""
    build_docx(documents_dir / "SHOUTED.DOCX", PLAIN_DOCUMENT)
    listed = client.get("/api/plugins/onlyoffice/docs", headers=headers).json()["documents"]
    assert "SHOUTED.DOCX" in [d["path"] for d in listed]
    assert client.get("/api/plugins/onlyoffice/config?path=SHOUTED.DOCX", headers=headers).status_code == 200


def test_the_picker_offers_nothing_the_routes_will_refuse(client, headers, documents_dir):
    """`Path('.docx').suffix` is empty, so a name the listing matched is one `resolve` declines.

    The picker renders a row per listed document, and a row that can never open is
    the failure the listing filter exists to prevent.
    """
    build_docx(documents_dir / ".docx", PLAIN_DOCUMENT)
    listed = [d["path"] for d in client.get("/api/plugins/onlyoffice/docs", headers=headers).json()["documents"]]
    assert ".docx" not in listed
    for path in listed:
        assert client.get(f"/api/plugins/onlyoffice/config?path={path}", headers=headers).status_code == 200, path
