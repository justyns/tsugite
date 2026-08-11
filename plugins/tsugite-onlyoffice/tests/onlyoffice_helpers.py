"""Helpers and docx fixtures the OnlyOffice tests import directly.

Not in `conftest.py`: every plugin's conftest is importable as the bare name
`conftest`, so a test importing from that name gets whichever one pytest
collected first. This module's name is the plugin's own, so it resolves to this
file whatever else is being collected alongside it.
"""

from __future__ import annotations

import asyncio
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from urllib.parse import unquote

JWT_SECRET = "onlyoffice-test-secret"
SECRET_NAME = "onlyoffice-jwt-secret"
SERVER_URL = "https://onlyoffice.example.net"
PUBLIC_BASE_URL = "https://tsugite.example.net"
AGENT_NAME = "Reviewer Nine"

# Where the document server parks what it saved, for the daemon to fetch.
DOWNLOAD_URL = f"{SERVER_URL}/cache/files/output.docx"


class StubSecretBackend:
    """Stands in for the daemon's configured secrets backend."""

    def get(self, name: str) -> str | None:
        return JWT_SECRET if name == SECRET_NAME else None


def serve_downloads(adapter, respond):
    """Answer the adapter's outbound traffic from a mock transport, recording the URLs.

    The adapter's own download still runs this way, which swapping a method out
    would skip.

    Args:
        adapter: The adapter whose shared client to replace.
        respond: The bytes to answer a download with, or a handler taking the
            request and returning an `httpx.Response`.

    Returns:
        The list every requested URL is appended to.
    """
    import httpx

    fetched = []

    def handle(request: httpx.Request) -> httpx.Response:
        fetched.append(str(request.url))
        return respond(request) if callable(respond) else httpx.Response(200, content=respond)

    adapter.http = httpx.AsyncClient(transport=httpx.MockTransport(handle))
    return fetched


def asgi_client(http_server):
    """An httpx client that runs the app on the caller's own event loop.

    A live read parks on a future the callback resolves, so the two halves have
    to share one loop. TestClient runs the app on a loop of its own in another
    thread, which is exactly the arrangement that cannot resolve that future.
    """
    import httpx

    transport = httpx.ASGITransport(app=http_server.app)
    return httpx.AsyncClient(transport=transport, base_url=PUBLIC_BASE_URL)


def callback_body(status, key, url=DOWNLOAD_URL, secret=JWT_SECRET):
    """A signed callback body, shaped the way the document server POSTs one.

    `url=None` is the shape of a status the document server parked nothing for:
    the field is absent rather than null, which is what a handler reaching for it
    has to survive.
    """
    from tsugite_onlyoffice import jwt

    body = {"key": key, "status": status, "users": ["tsugite"]}
    if url is not None:
        body["url"] = url
    return {**body, "token": jwt.sign(body, secret)}


def callback_url(relative, signs=None):
    """The callback URL the editor config mints, path-bound token and all.

    `signs` binds the token to some other document than the one in the path,
    which is the shape of a callback replayed at somebody else's document.
    """
    from tsugite_onlyoffice import jwt
    from tsugite_onlyoffice.adapter import USE_CALLBACK

    claims = {"document": unquote(relative) if signs is None else signs, "use": USE_CALLBACK}
    doc_token = jwt.sign(claims, JWT_SECRET, expires_in=0)
    return f"/api/plugins/onlyoffice/callback/{relative}?doc_token={doc_token}"


async def post_callback(http_server, relative, body):
    """POST a callback through the app, on the caller's own loop, and expect a zero."""
    async with asgi_client(http_server) as client:
        response = await client.post(callback_url(relative), json=body)
    assert response.json() == {"error": 0}, response.text
    return response


class FakeCommands:
    """A CommandService that records what it was asked, and answers it itself.

    `answer` stands in for the callback the document server would post back.
    Short-circuiting the callback keeps a test's step order predictable; the tests
    that care what reaches disk pass an answer that posts the real callback.
    """

    def __init__(self, calls=None, answer=None, nothing_to_do=()):
        self.calls = [] if calls is None else calls
        self.answer = answer
        # Commands the document server reports as having nothing to do, which it
        # spends an error code on and the client reads back as False.
        self.nothing_to_do = set(nothing_to_do)

    async def forcesave(self, key):
        return await self._issue("forcesave", key)

    async def _issue(self, command, key):
        self.calls.append((command, key))
        # The yield is where two turns that were not serialized would interleave.
        await asyncio.sleep(0)
        if command in self.nothing_to_do:
            return False
        if self.answer is not None:
            await self.answer(command, key)
        return True


# ── docx packages ──
#
# Built here rather than committed as binaries, so what every part of a fixture
# contains is readable in the diff that changes a test.

DECLARATION = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\r\n'

CONTENT_TYPES = DECLARATION + (
    '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
    '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
    '<Default Extension="xml" ContentType="application/xml"/>'
    '<Override PartName="/word/document.xml"'
    ' ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
    '<Override PartName="/word/styles.xml"'
    ' ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>'
    "</Types>"
)

PACKAGE_RELS = DECLARATION + (
    '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
    '<Relationship Id="rId1"'
    ' Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument"'
    ' Target="word/document.xml"/>'
    "</Relationships>"
)

DOCUMENT_RELS = DECLARATION + (
    '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
    '<Relationship Id="rId1"'
    ' Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>'
    "</Relationships>"
)

EMPTY_RELS = DECLARATION + '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"/>'

STYLES = DECLARATION + (
    '<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
    '<w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/></w:style>'
    "</w:styles>"
)

PLAIN_DOCUMENT = DECLARATION + (
    '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body>'
    "<w:p><w:r><w:t>Quarterly review</w:t></w:r></w:p>"
    "<w:p><w:r><w:t>The pilot shipped on time.</w:t></w:r></w:p>"
    "<w:p><w:r><w:t>Costs held flat.</w:t></w:r></w:p>"
    '<w:sectPr><w:pgSz w:w="11906" w:h="16838"/></w:sectPr>'
    "</w:body></w:document>"
)

# Declares wpc and r without using either, carries mc:Ignorable, and mixes run
# properties, a tab, a line break and a tracked deletion into the body.
STYLED_DOCUMENT = DECLARATION + (
    "<w:document"
    ' xmlns:wpc="http://schemas.microsoft.com/office/word/2010/wordprocessingCanvas"'
    ' xmlns:mc="http://schemas.openxmlformats.org/markup-compatibility/2006"'
    ' xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"'
    ' xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"'
    ' xmlns:w14="http://schemas.microsoft.com/office/word/2010/wordml"'
    ' xmlns:w15="http://schemas.microsoft.com/office/word/2012/wordml"'
    ' mc:Ignorable="w14 w15">'
    "<w:body>"
    '<w:p w14:paraId="1A2B3C4D">'
    '<w:r><w:rPr><w:b/></w:rPr><w:t xml:space="preserve">Bold start </w:t></w:r>'
    "<w:r><w:rPr><w:i/></w:rPr><w:t>and italic tail</w:t></w:r>"
    "</w:p>"
    "<w:p><w:r><w:t>Second</w:t></w:r><w:r><w:tab/></w:r><w:r><w:t>after a tab</w:t></w:r></w:p>"
    "<w:p><w:r><w:t>Line one</w:t></w:r><w:r><w:br/></w:r><w:r><w:t>line two</w:t></w:r></w:p>"
    '<w:p><w:del w:id="7" w:author="Reviewer" w:date="2024-02-02T10:00:00Z">'
    "<w:r><w:delText>removed sentence </w:delText></w:r></w:del>"
    "<w:r><w:t>kept sentence</w:t></w:r></w:p>"
    "<w:p/>"
    "</w:body></w:document>"
)

# What a live editing session hands back: the document plus a sentence that only
# ever existed in the editor, so a read that took the file instead shows up.
TYPED_DOCUMENT = DECLARATION + (
    '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body>'
    "<w:p><w:r><w:t>Quarterly review, typed while the session was open</w:t></w:r></w:p>"
    "<w:p><w:r><w:t>Costs held flat.</w:t></w:r></w:p>"
    "</w:body></w:document>"
)

# A comment from before commentsExtended existed: it is anchored and it has a
# body, but nothing in the package carries a paraId for it.
LEGACY_COMMENT_DOCUMENT = DECLARATION + (
    '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body>'
    '<w:p><w:commentRangeStart w:id="1"/><w:r><w:t>Quarterly review</w:t></w:r>'
    '<w:commentRangeEnd w:id="1"/><w:r><w:commentReference w:id="1"/></w:r></w:p>'
    "<w:p><w:r><w:t>The pilot shipped on time.</w:t></w:r></w:p>"
    "</w:body></w:document>"
)

LEGACY_COMMENTS = DECLARATION + (
    '<w:comments xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
    '<w:comment w:id="1" w:author="Example Reviewer" w:date="2011-05-04T09:00:00Z" w:initials="ER">'
    "<w:p><w:r><w:t>Pre-2013 note.</w:t></w:r></w:p>"
    "</w:comment></w:comments>"
)

# A comment left with nothing selected, which anchors at a point: it has a
# reference in the body and no range around any of the text. Stored second and
# anchored first, so the read has to place it by where its reference sits.
POINT_COMMENT_DOCUMENT = DECLARATION + (
    '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body>'
    '<w:p><w:r><w:t>Quarterly review</w:t></w:r><w:r><w:commentReference w:id="2"/></w:r></w:p>'
    '<w:p><w:commentRangeStart w:id="1"/><w:r><w:t>The pilot shipped on time.</w:t></w:r>'
    '<w:commentRangeEnd w:id="1"/><w:r><w:commentReference w:id="1"/></w:r></w:p>'
    "</w:body></w:document>"
)

POINT_COMMENTS = DECLARATION + (
    '<w:comments xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"'
    ' xmlns:w14="http://schemas.microsoft.com/office/word/2010/wordml">'
    '<w:comment w:id="1" w:author="Example Reviewer" w:date="2026-08-10T09:31:04Z" w:initials="ER">'
    '<w:p w14:paraId="0000C001"><w:r><w:t>On the second paragraph.</w:t></w:r></w:p>'
    "</w:comment>"
    '<w:comment w:id="2" w:author="Example Reviewer" w:date="2026-08-10T09:33:12Z" w:initials="ER">'
    '<w:p w14:paraId="0000C002"><w:r><w:t>Left at a point in the first.</w:t></w:r></w:p>'
    "</w:comment></w:comments>"
)

# Two comments the document server stored in the opposite order to the one they
# are anchored in, which is the order a reader of the document meets them in.
SHUFFLED_COMMENT_DOCUMENT = DECLARATION + (
    '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body>'
    '<w:p><w:commentRangeStart w:id="2"/><w:r><w:t>Quarterly review</w:t></w:r>'
    '<w:commentRangeEnd w:id="2"/><w:r><w:commentReference w:id="2"/></w:r></w:p>'
    '<w:p><w:commentRangeStart w:id="1"/><w:r><w:t>The pilot shipped on time.</w:t></w:r>'
    '<w:commentRangeEnd w:id="1"/><w:r><w:commentReference w:id="1"/></w:r></w:p>'
    "</w:body></w:document>"
)

SHUFFLED_COMMENTS = DECLARATION + (
    '<w:comments xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"'
    ' xmlns:w14="http://schemas.microsoft.com/office/word/2010/wordml">'
    '<w:comment w:id="1" w:author="Example Reviewer" w:date="2026-08-10T03:16:53Z" w:initials="ER">'
    '<w:p w14:paraId="0000A001"><w:r><w:t>On the second paragraph.</w:t></w:r></w:p>'
    "</w:comment>"
    '<w:comment w:id="2" w:author="Example Reviewer" w:date="2026-08-10T08:08:28Z" w:initials="ER">'
    '<w:p w14:paraId="0000A002"><w:r><w:t>On the first paragraph.</w:t></w:r></w:p>'
    "</w:comment></w:comments>"
)

# Two comments nothing in the content separates: same author, same second, same words.
TWIN_COMMENT_DOCUMENT = DECLARATION + (
    '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body>'
    '<w:p><w:commentRangeStart w:id="1"/><w:r><w:t>Quarterly review</w:t></w:r>'
    '<w:commentRangeEnd w:id="1"/><w:r><w:commentReference w:id="1"/></w:r></w:p>'
    '<w:p><w:commentRangeStart w:id="2"/><w:r><w:t>The pilot shipped on time.</w:t></w:r>'
    '<w:commentRangeEnd w:id="2"/><w:r><w:commentReference w:id="2"/></w:r></w:p>'
    "</w:body></w:document>"
)

TWIN_COMMENTS = DECLARATION + (
    '<w:comments xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"'
    ' xmlns:w14="http://schemas.microsoft.com/office/word/2010/wordml">'
    '<w:comment w:id="1" w:author="Example Reviewer" w:date="2026-08-10T08:08:28Z" w:initials="ER">'
    '<w:p w14:paraId="0000B001"><w:r><w:t>Same note, twice.</w:t></w:r></w:p>'
    "</w:comment>"
    '<w:comment w:id="2" w:author="Example Reviewer" w:date="2026-08-10T08:08:28Z" w:initials="ER">'
    '<w:p w14:paraId="0000B002"><w:r><w:t>Same note, twice.</w:t></w:r></w:p>'
    "</w:comment></w:comments>"
)

REPEATED_DOCUMENT = DECLARATION + (
    '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body>'
    "<w:p><w:r><w:t>The draft went out on Monday and the draft came back on Friday.</w:t></w:r></w:p>"
    "<w:p><w:r><w:t>A third draft follows next week.</w:t></w:r></w:p>"
    "</w:body></w:document>"
)

_ZIP_DATE = (2024, 3, 4, 9, 15, 30)


def build_docx(path: Path, document_xml: str, extra: dict[str, str] | None = None) -> Path:
    """Write a minimal valid docx package.

    Args:
        path: Where the package goes.
        document_xml: The `word/document.xml` part, as text.
        extra: Further parts, keyed by zip entry name.

    Returns:
        The path that was written.
    """
    entries = {
        "[Content_Types].xml": CONTENT_TYPES,
        "_rels/.rels": PACKAGE_RELS,
        "word/document.xml": document_xml,
        "word/_rels/document.xml.rels": DOCUMENT_RELS,
        "word/styles.xml": STYLES,
    }
    entries.update(extra or {})
    with zipfile.ZipFile(path, "w") as archive:
        for offset, (name, body) in enumerate(entries.items()):
            # Distinct timestamps and one stored entry, so a save that rebuilds the
            # zip with fresh metadata instead of the original shows up as a failure.
            date = (*_ZIP_DATE[:5], _ZIP_DATE[5] + offset * 2)
            compress = zipfile.ZIP_STORED if name == "[Content_Types].xml" else zipfile.ZIP_DEFLATED
            archive.writestr(zipfile.ZipInfo(name, date_time=date), body, compress_type=compress)
    return path


def zip_entries(path):
    """Every zip entry with the metadata a rebuild is free to get wrong."""
    with zipfile.ZipFile(path) as archive:
        return {info.filename: (archive.read(info), info.date_time, info.compress_type) for info in archive.infolist()}


def zip_part(path, name):
    """One part's raw bytes."""
    with zipfile.ZipFile(path) as archive:
        return archive.read(name)


def runs(path, number):
    """The text and the formatting marks of every run in one saved paragraph."""
    from tsugite_onlyoffice.docx import DOCUMENT_PART, w

    paragraph = list(ET.fromstring(zip_part(path, DOCUMENT_PART)).iter(w("p")))[number - 1]
    out = []
    for run in paragraph.iter(w("r")):
        text = "".join(node.text or "" for node in run.iter(w("t")))
        props = run.find(w("rPr"))
        marks = [child.tag.rpartition("}")[2] for child in props] if props is not None else []
        out.append((text, marks))
    return out
