"""fetch_text / http_request / download_file / check_url honor the web fetch denylist and one-time denies."""

from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import httpx
import pytest
from tsugite_daemon.session_runner import _current_session_id

from tsugite import permissions
from tsugite.permissions import Permissions, deny_web_fetch_for_session, set_permissions
from tsugite.tools.http import check_url, download_file, fetch_text, http_request


@contextmanager
def in_session(session_id):
    token = _current_session_id.set(session_id)
    try:
        yield
    finally:
        _current_session_id.reset(token)


@pytest.fixture(autouse=True)
def _clean_state():
    set_permissions(None)
    permissions._session_denied_hosts.clear()
    yield
    set_permissions(None)
    permissions._session_denied_hosts.clear()


@pytest.fixture
def client():
    """Mocked httpx.Client; no test reaches the network."""
    with patch("tsugite.tools.http.httpx.Client") as cls:
        c = MagicMock()
        resp = MagicMock(spec=httpx.Response)
        resp.status_code = 200
        resp.headers = httpx.Headers({"content-type": "text/plain"})
        resp.text = "fetched"
        resp.url = "https://ok.test/"
        resp.raise_for_status.return_value = None
        c.request.return_value = resp
        cls.return_value.__enter__ = MagicMock(return_value=c)
        cls.return_value.__exit__ = MagicMock(return_value=None)
        yield c


@pytest.fixture
def perms(tmp_path):
    p = Permissions(tmp_path / "permissions.yaml")
    set_permissions(p)
    return p


def test_fetch_text_refuses_denylisted_host(client, perms):
    perms.web_fetch_deny("bad.test")
    with pytest.raises(RuntimeError, match="denied fetching bad.test"):
        fetch_text("https://BAD.test/page")
    client.request.assert_not_called()


def test_http_request_refuses_denylisted_host(client, perms):
    perms.web_fetch_deny("bad.test")
    with pytest.raises(RuntimeError, match="denied fetching bad.test"):
        http_request("https://bad.test/api", method="POST", body={"a": 1})
    client.request.assert_not_called()


def test_check_url_refuses_denylisted_host(client, perms):
    perms.web_fetch_deny("bad.test")
    with pytest.raises(RuntimeError, match="denied fetching bad.test"):
        check_url("https://bad.test/")
    client.head.assert_not_called()


def test_download_file_refuses_denylisted_host(client, perms, tmp_path):
    perms.web_fetch_deny("bad.test")
    target = tmp_path / "out"
    with pytest.raises(RuntimeError, match="denied fetching bad.test"):
        download_file("https://bad.test/f", str(target))
    client.stream.assert_not_called()
    assert not target.exists()


def test_one_time_deny_blocks_later_fetch_text_without_persisting(client, perms):
    deny_web_fetch_for_session("Once.test")
    with pytest.raises(RuntimeError, match="once.test"):
        fetch_text("https://once.test/x")
    client.request.assert_not_called()
    assert not perms.runtime_path.exists()


def test_one_time_deny_blocks_fetch_with_no_store(client):
    deny_web_fetch_for_session("once.test")
    with pytest.raises(RuntimeError, match="once.test"):
        fetch_text("https://once.test/x")


def test_one_time_deny_is_scoped_to_its_session(client):
    with in_session("A"):
        deny_web_fetch_for_session("once.test")
    with in_session("B"):
        assert fetch_text("https://once.test/x") == "fetched"
    with in_session("A"):
        with pytest.raises(RuntimeError, match="once.test"):
            fetch_text("https://once.test/x")


def test_undenied_host_still_fetches(client, perms):
    perms.web_fetch_deny("bad.test")
    deny_web_fetch_for_session("once.test")
    assert fetch_text("https://ok.test/") == "fetched"
    assert http_request("https://ok.test/").text == "fetched"


def test_no_permissions_store_fetches(client):
    assert fetch_text("https://ok.test/") == "fetched"


_RealClient = httpx.Client


def _redirecting_client(**kwargs):
    def handler(request):
        if request.url.host == "ok.test":
            return httpx.Response(302, headers={"location": "https://bad.test/"})
        return httpx.Response(200, text="secret")

    return _RealClient(**kwargs, transport=httpx.MockTransport(handler))


def test_redirect_to_denylisted_host_is_refused(perms):
    perms.web_fetch_deny("bad.test")
    with patch("tsugite.tools.http.httpx.Client", _redirecting_client):
        with pytest.raises(RuntimeError, match="denied fetching bad.test"):
            fetch_text("https://ok.test/")
        result = check_url("https://ok.test/")
    assert result["accessible"] is False
    assert "denied fetching bad.test" in result["error"]


@pytest.mark.asyncio
@pytest.mark.parametrize("deny", ["session", "denylist"])
async def test_subprocess_executor_child_refuses_denied_host(reset_tool_registry, tmp_path, deny):
    from tsugite.core.subprocess_executor import SubprocessExecutor
    from tsugite.core.tools import create_tool_from_tsugite
    from tsugite.tools import _register_tool

    _register_tool(fetch_text)
    if deny == "session":
        deny_web_fetch_for_session("denied.invalid")
    else:
        p = Permissions(tmp_path / "permissions.yaml")
        p.web_fetch_deny("denied.invalid")
        set_permissions(p)

    ex = SubprocessExecutor()
    try:
        ex.set_tools([create_tool_from_tsugite("fetch_text")])
        assert "fetch_text" in ex._local_tools
        result = await ex.execute(
            "try:\n    fetch_text('https://denied.invalid/')\nexcept Exception as e:\n    print('child:', e)"
        )
    finally:
        ex.cleanup()
    assert "denied fetching denied.invalid" in result.output


async def _child_fetch_output(host):
    from tsugite.core.subprocess_executor import SubprocessExecutor
    from tsugite.core.tools import create_tool_from_tsugite

    ex = SubprocessExecutor()
    try:
        ex.set_tools([create_tool_from_tsugite("fetch_text")])
        result = await ex.execute(
            f"try:\n    fetch_text('https://{host}/', timeout=2)\nexcept Exception as e:\n    print('child:', e)"
        )
    finally:
        ex.cleanup()
    return result.output


@pytest.mark.asyncio
async def test_subprocess_executor_child_sees_only_its_sessions_denies(reset_tool_registry):
    from tsugite.tools import _register_tool

    _register_tool(fetch_text)
    with in_session("A"):
        deny_web_fetch_for_session("denied.invalid")
        assert "denied fetching denied.invalid" in await _child_fetch_output("denied.invalid")
    with in_session("B"):
        assert "denied fetching" not in await _child_fetch_output("denied.invalid")
