"""Shared fixtures for the OnlyOffice plugin's tests."""

from __future__ import annotations

import pytest
from onlyoffice_helpers import (
    AGENT_NAME,
    EMPTY_RELS,
    LEGACY_COMMENT_DOCUMENT,
    LEGACY_COMMENTS,
    PLAIN_DOCUMENT,
    POINT_COMMENT_DOCUMENT,
    POINT_COMMENTS,
    PUBLIC_BASE_URL,
    REPEATED_DOCUMENT,
    SECRET_NAME,
    SERVER_URL,
    SHUFFLED_COMMENT_DOCUMENT,
    SHUFFLED_COMMENTS,
    STYLED_DOCUMENT,
    TWIN_COMMENT_DOCUMENT,
    TWIN_COMMENTS,
    TYPED_DOCUMENT,
    StubSecretBackend,
    build_docx,
)
from starlette.testclient import TestClient


@pytest.fixture
def secrets_backend():
    from tsugite.secrets import set_backend

    set_backend(StubSecretBackend())
    yield
    set_backend(None)  # force a fresh re-create for the next test in this worker


@pytest.fixture
def documents_dir(tmp_path):
    root = tmp_path / "documents"
    (root / "reports").mkdir(parents=True)
    build_docx(root / "notes.docx", PLAIN_DOCUMENT)
    build_docx(root / "reports" / "q1.docx", PLAIN_DOCUMENT)
    (root / "readme.txt").write_text("not a document")
    build_docx(tmp_path / "outside.docx", PLAIN_DOCUMENT)
    return root


@pytest.fixture
def adapter(documents_dir, secrets_backend):
    from tsugite_onlyoffice.adapter import create_adapter

    return create_adapter(
        config={
            "enabled": True,
            "server_url": SERVER_URL,
            "jwt_secret_name": SECRET_NAME,
            "public_base_url": PUBLIC_BASE_URL,
            "documents_dir": str(documents_dir),
            "agent_name": AGENT_NAME,
        },
        runtime=None,
        session_store=None,
        identity_map={},
    )


@pytest.fixture
def token_store(tmp_path):
    from tsugite_daemon.auth import TokenStore

    return TokenStore(tmp_path / "tokens.json")


@pytest.fixture
def headers(token_store):
    _record, raw = token_store.create_admin_token(name="test-request-token")
    return {"Authorization": f"Bearer {raw}"}


@pytest.fixture
def http_server(adapter, token_store, tmp_path):
    from tsugite_daemon.adapters.http import HTTPServer
    from tsugite_daemon.config import HTTPConfig
    from tsugite_daemon.plugin_wiring import attach_plugin_http
    from tsugite_daemon.webhook_store import WebhookStore

    server = HTTPServer(
        config=HTTPConfig(enabled=True, host="127.0.0.1", port=8374),
        adapter=None,
        webhook_store=WebhookStore(tmp_path / "webhooks.json"),
        token_store=token_store,
    )
    # tools.py registers this at import; the gateway reads the registry and passes
    # the same descriptor.
    from tsugite_onlyoffice.tools import DOC_SURFACE

    attach_plugin_http(server, "onlyoffice", adapter, [DOC_SURFACE])
    return server


@pytest.fixture
def client(http_server):
    return TestClient(http_server.app)


@pytest.fixture
def typed_bytes(tmp_path):
    """The package the document server would hand back from a live session."""
    return build_docx(tmp_path / "typed.docx", TYPED_DOCUMENT).read_bytes()


@pytest.fixture
def plain_docx(tmp_path):
    return build_docx(tmp_path / "plain.docx", PLAIN_DOCUMENT)


@pytest.fixture
def repeated_docx(tmp_path):
    return build_docx(tmp_path / "repeated.docx", REPEATED_DOCUMENT)


@pytest.fixture
def styled_docx(tmp_path):
    return build_docx(tmp_path / "styled.docx", STYLED_DOCUMENT)


@pytest.fixture
def legacy_comment_docx(tmp_path):
    """A commented package with no `word/commentsExtended.xml` and no paraIds."""
    return build_docx(tmp_path / "legacy.docx", LEGACY_COMMENT_DOCUMENT, {"word/comments.xml": LEGACY_COMMENTS})


@pytest.fixture
def point_comment_docx(tmp_path):
    """A package holding a comment anchored at a point rather than over a range."""
    return build_docx(tmp_path / "point.docx", POINT_COMMENT_DOCUMENT, {"word/comments.xml": POINT_COMMENTS})


@pytest.fixture
def shuffled_comment_docx(tmp_path):
    """A package whose comments are stored in a different order than they are anchored in."""
    return build_docx(tmp_path / "shuffled.docx", SHUFFLED_COMMENT_DOCUMENT, {"word/comments.xml": SHUFFLED_COMMENTS})


@pytest.fixture
def twin_comment_docx(tmp_path):
    """A package holding two comments identical in author, date and text."""
    return build_docx(tmp_path / "twins.docx", TWIN_COMMENT_DOCUMENT, {"word/comments.xml": TWIN_COMMENTS})


@pytest.fixture
def orphan_comment_rels_docx(tmp_path):
    """A package shipping comment relationships with no `word/comments.xml` behind them."""
    return build_docx(tmp_path / "orphan.docx", PLAIN_DOCUMENT, {"word/_rels/comments.xml.rels": EMPTY_RELS})
