"""The egress filter blocks or redacts configured strings in agent HTTP requests."""

import logging
from unittest.mock import patch

import httpx
import pytest

from tsugite.config import Config, EgressFilterConfig
from tsugite.egress_filter import EgressBlockedError, apply_egress_filter, get_egress_filter
from tsugite.tools.http import check_url, download_file, fetch_text, http_request

SECRET = "Jane.Q.Example@example.test"
STRINGS = ["unrelated", "other", SECRET]
URL = "https://api.example.test/v1"


def _client(requests):
    def handler(request):
        requests.append(request)
        return httpx.Response(200, text="ok")

    return httpx.Client(transport=httpx.MockTransport(handler), event_hooks={"request": [apply_egress_filter]})


@pytest.fixture
def filter_config():
    def configure(mode="reject", strings=STRINGS):
        config = EgressFilterConfig(mode=mode, strings=strings)
        return patch("tsugite.egress_filter.get_egress_filter", return_value=config)

    return configure


SENDS = {
    "url": lambda c: c.get(f"{URL}?contact={SECRET.upper()}"),
    "encoded url": lambda c: c.get(URL, params={"contact": SECRET}),
    "header": lambda c: c.get(URL, headers={"From": f"bot <{SECRET}>"}),
    "text body": lambda c: c.post(URL, content=f"hello {SECRET}"),
    "json body": lambda c: c.post(URL, json={"a": {"b": [SECRET]}}),
    "form body": lambda c: c.post(URL, data={"contact": SECRET}),
}


@pytest.mark.parametrize("where", SENDS)
def test_reject_sends_nothing_and_names_only_the_rule(where, filter_config, caplog):
    requests = []
    with filter_config(), caplog.at_level(logging.DEBUG), pytest.raises(EgressBlockedError) as exc:
        SENDS[where](_client(requests))
    assert requests == []
    assert "strings[2]" in str(exc.value)
    assert SECRET.lower() not in (str(exc.value) + caplog.text).lower()


@pytest.mark.parametrize("where", SENDS)
def test_warn_redacts_and_sends(where, filter_config, caplog):
    requests = []
    with filter_config("warn"), caplog.at_level(logging.DEBUG):
        assert SENDS[where](_client(requests)).status_code == 200
    (sent,) = requests
    wire = f"{sent.url} {dict(sent.headers)} {sent.read().decode()}"
    assert SECRET.lower() not in wire.lower()
    assert "REDACTED" in wire
    assert int(sent.headers.get("content-length", 0)) == len(sent.read())
    assert "strings[2]" in caplog.text
    assert SECRET.lower() not in caplog.text.lower()


def test_warn_keeps_a_binary_body_when_only_the_url_matched(filter_config):
    requests = []
    body = b"\xff\xfe\x00binary"
    with filter_config("warn"):
        _client(requests).post(f"{URL}?contact={SECRET}", content=body)
    (sent,) = requests
    assert sent.read() == body


def test_warn_keeps_other_query_params_encoded(filter_config):
    requests = []
    with filter_config("warn"):
        _client(requests).get(URL, params={"keep": "a&b=c", "contact": SECRET})
    (sent,) = requests
    assert sent.url.params["keep"] == "a&b=c"


@pytest.mark.parametrize(("mode", "strings"), [("off", STRINGS), ("reject", [])])
def test_off_and_empty_list_pass_through(mode, strings, filter_config):
    requests = []
    with filter_config(mode, strings):
        SENDS["header"](_client(requests))
    assert len(requests) == 1 and SECRET in requests[0].headers["from"]


def test_filter_failure_sends_nothing():
    requests = []
    with patch("tsugite.egress_filter.get_egress_filter", side_effect=ValueError("boom")), pytest.raises(ValueError):
        SENDS["url"](_client(requests))
    assert requests == []


def test_default_mode_is_reject():
    with patch("tsugite.egress_filter.load_config", return_value=Config()):
        get_egress_filter.cache_clear()
        assert get_egress_filter().mode == "reject"
    get_egress_filter.cache_clear()


@pytest.mark.parametrize(
    "call",
    [
        lambda: http_request(f"{URL}?q={SECRET}"),
        lambda: fetch_text(f"{URL}?q={SECRET}"),
        lambda: download_file(f"{URL}?q={SECRET}", "unused"),
        lambda: check_url(f"{URL}?q={SECRET}"),
    ],
    ids=["http_request", "fetch_text", "download_file", "check_url"],
)
def test_http_tools_run_the_filter(call, filter_config):
    requests = []
    real_client = httpx.Client

    def client(**kwargs):
        return real_client(transport=httpx.MockTransport(lambda r: requests.append(r) or httpx.Response(200)), **kwargs)

    with filter_config(), patch("tsugite.tools.http.httpx.Client", side_effect=client):
        try:
            call()
        except RuntimeError as e:
            assert "strings[2]" in str(e)
    assert requests == []
