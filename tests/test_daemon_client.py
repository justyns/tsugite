"""DaemonClient against a stubbed urlopen; nothing here touches the network."""

import io
import json
from contextlib import contextmanager
from email.message import Message
from unittest.mock import patch
from urllib.error import HTTPError, URLError

import pytest
from typer.testing import CliRunner

from tsugite.cli import app
from tsugite.daemon_client import DaemonClient, DaemonConnectionError, DaemonError


@pytest.fixture(autouse=True)
def no_daemon_env(monkeypatch):
    for var in ("TSUGITE_DAEMON_URL", "TSUGITE_DAEMON_TOKEN", "TSUGITE_DAEMON_HOST", "TSUGITE_DAEMON_PORT"):
        monkeypatch.delenv(var, raising=False)


@contextmanager
def stub_daemon(reply=b"{}", error=None):
    with patch("tsugite.daemon_client.urlopen") as urlopen:
        if error:
            urlopen.side_effect = error
        else:
            urlopen.return_value.__enter__.return_value.read.return_value = reply
        yield urlopen


def sent(urlopen):
    return urlopen.call_args[0][0]


@contextmanager
def stub_chat(body: bytes, content_type="text/event-stream", session_id=None):
    resp = io.BytesIO(body)
    resp.headers = Message()
    resp.headers["Content-Type"] = content_type
    if session_id:
        resp.headers["X-Session-Id"] = session_id
    with patch("tsugite.daemon_client.urlopen") as urlopen:
        urlopen.return_value.__enter__.return_value = resp
        yield urlopen


def sse(*events):
    return b": keepalive\n\n" + b"".join(f"data: {json.dumps(e)}\n\n".encode() for e in (*events, {"type": "done"}))


def test_env_url_and_token_apply_without_arguments(monkeypatch):
    monkeypatch.setenv("TSUGITE_DAEMON_URL", "http://daemon.example:1234/")
    monkeypatch.setenv("TSUGITE_DAEMON_TOKEN", "env-token")
    with stub_daemon() as urlopen:
        DaemonClient().request("GET", "/x")
    req = sent(urlopen)
    assert req.full_url == "http://daemon.example:1234/x"
    assert req.get_header("Authorization") == "Bearer env-token"


def test_arguments_beat_env(monkeypatch):
    monkeypatch.setenv("TSUGITE_DAEMON_URL", "http://daemon.example:1234")
    monkeypatch.setenv("TSUGITE_DAEMON_TOKEN", "env-token")
    with stub_daemon() as urlopen:
        DaemonClient("10.0.0.5", 9100, "arg-token").request("GET", "/x")
    req = sent(urlopen)
    assert req.full_url == "http://10.0.0.5:9100/x"
    assert req.get_header("Authorization") == "Bearer arg-token"


def test_zero_timeout_is_not_unset():
    with stub_daemon() as urlopen:
        DaemonClient().request("GET", "/x", timeout=0)
    assert urlopen.call_args.kwargs["timeout"] == 0


def test_no_token_sends_no_auth_header():
    with stub_daemon() as urlopen:
        DaemonClient().request("GET", "/x")
    assert sent(urlopen).get_header("Authorization") is None


def test_create_job_posts_json():
    with stub_daemon(b'{"id": "job-1"}') as urlopen:
        job = DaemonClient().create_job("me", "do it", repo="r")
    req = sent(urlopen)
    assert req.get_method() == "POST"
    assert req.full_url.endswith("/api/jobs")
    assert json.loads(req.data) == {"user_id": "me", "task": "do it", "repo": "r"}
    assert req.get_header("Content-type") == "application/json"
    assert job == {"id": "job-1"}


def test_list_jobs_sends_filters_and_unwraps():
    with stub_daemon(b'{"jobs": [{"id": "a"}]}') as urlopen:
        jobs = DaemonClient().list_jobs(state="active", limit=5)
    assert sent(urlopen).full_url.endswith("/api/jobs?state=active&limit=5")
    assert jobs == [{"id": "a"}]


def test_retry_and_cancel_paths():
    client = DaemonClient()
    with stub_daemon() as urlopen:
        client.retry_job("j/1", hint="again")
        assert sent(urlopen).full_url.endswith("/api/jobs/j%2F1/retry")
        assert json.loads(sent(urlopen).data) == {"hint": "again"}
        client.cancel_job("j1", reason="stop")
        assert sent(urlopen).full_url.endswith("/api/jobs/j1/cancel")
        assert json.loads(sent(urlopen).data) == {"reason": "stop"}


def test_clear_primary_session_uses_query_string():
    with stub_daemon() as urlopen:
        DaemonClient().clear_primary_session("u 1")
    assert sent(urlopen).full_url.endswith("/api/sessions/clear-primary?user_id=u+1")


def test_empty_body_is_none():
    with stub_daemon(b"") as urlopen:
        assert DaemonClient().request("POST", "/x") is None
    assert sent(urlopen).data is None


def test_http_error_carries_status_and_daemon_message():
    err = HTTPError("http://x", 404, "Not Found", {}, io.BytesIO(b'{"error": "unknown job"}'))
    with stub_daemon(error=err), pytest.raises(DaemonError, match="unknown job") as exc:
        DaemonClient().cancel_job("nope")
    assert exc.value.status == 404


def test_non_json_error_body_is_used_verbatim():
    err = HTTPError("http://x", 502, "Bad Gateway", {}, io.BytesIO(b"upstream down"))
    with stub_daemon(error=err), pytest.raises(DaemonError, match="upstream down"):
        DaemonClient().list_jobs()


def test_unreachable_daemon_raises_connection_error():
    with stub_daemon(error=URLError(ConnectionRefusedError())), pytest.raises(DaemonConnectionError) as exc:
        DaemonClient("127.0.0.1", 1).list_jobs()
    assert "127.0.0.1:1" in str(exc.value)
    assert exc.value.status is None


def test_chat_stream_yields_events_until_done():
    events = [{"type": "init", "agent": "a"}, {"type": "final_result", "result": "hi"}]
    with stub_chat(sse(*events)) as urlopen:
        got = list(DaemonClient().chat_stream("hello", session_id="s1"))
    assert got == events
    req = sent(urlopen)
    assert req.get_method() == "POST"
    assert req.full_url.endswith("/api/chat")
    assert json.loads(req.data) == {"message": "hello", "session_id": "s1"}


def test_chat_stream_starts_with_the_session_id():
    with stub_chat(sse({"type": "final_result", "result": "hi"}), session_id="s-new"):
        got = list(DaemonClient().chat_stream("hello"))
    assert got[0] == {"type": "session", "session_id": "s-new"}


def test_chat_result_carries_the_session_id():
    with stub_chat(sse({"type": "final_result", "result": "hi"}), session_id="s-new"):
        assert DaemonClient().chat("hello") == {"type": "final_result", "result": "hi", "session_id": "s-new"}


def test_chat_returns_final_result():
    with stub_chat(sse({"type": "thought", "content": "x"}, {"type": "final_result", "result": "hi"})):
        assert DaemonClient().chat("hello")["result"] == "hi"


def test_chat_raises_on_error_event():
    with (
        stub_chat(sse({"type": "error", "error": "model exploded", "step": 1})),
        pytest.raises(DaemonError, match="model exploded"),
    ):
        DaemonClient().chat("hello")


def test_chat_raises_without_final_result():
    with stub_chat(sse({"type": "cancelled"})), pytest.raises(DaemonError, match="without a final result"):
        DaemonClient().chat("hello")


def test_chat_returns_queue_receipt_for_busy_session():
    receipt = {"status": "queued", "queue_id": "q-1", "position": 1}
    with stub_chat(json.dumps(receipt).encode(), "application/json"):
        assert DaemonClient().chat("hello") == receipt


def test_cli_reports_daemon_error_and_exits_1():
    err = HTTPError("http://x", 404, "Not Found", {}, io.BytesIO(b'{"error": "no such schedule"}'))
    with stub_daemon(error=err):
        result = CliRunner().invoke(app, ["daemon", "schedule", "show", "ghost"])
    assert result.exit_code == 1
    assert "Error (404)" in result.output and "no such schedule" in result.output


def test_cli_reports_unreachable_daemon():
    with stub_daemon(error=URLError(ConnectionRefusedError())):
        result = CliRunner().invoke(app, ["daemon", "schedule", "list", "--port", "9"])
    assert result.exit_code == 1
    assert "Could not connect to daemon" in result.output


def test_module_imports_nothing_outside_the_stdlib():
    import subprocess
    import sys

    code = (
        "import sys; before = set(sys.modules); import tsugite.daemon_client; "
        "print(*sorted({m.split('.')[0] for m in set(sys.modules) - before}))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert set(out.split()) <= sys.stdlib_module_names | {"tsugite"}
