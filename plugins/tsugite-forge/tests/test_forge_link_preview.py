"""Chat links to a configured forge's pull requests and issues preview from the cache, else a live read."""

from unittest.mock import MagicMock

import pytest
from forge_helpers import TOKEN, Clock, FakeForgejo, admin_headers, forge_http, make_adapter, make_pr, seed
from tsugite_forge.providers import ForgeError, Issue

from tsugite.link_previews import attributing_to, register_link_resolver, registered_link_resolvers

PR_URL = "https://forge.example.test/octo-org/demo/pulls/7"


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def adapter(clock):
    adapter = make_adapter(clock)
    seed(adapter, clock.now)
    for cache in adapter.forges.values():
        cache.provider = MagicMock()
    return adapter


def _provider(adapter, forge="home"):
    return adapter.forges[forge].provider


def test_a_cached_pull_request_needs_no_forge_call(adapter):
    preview = adapter.link_preview(PR_URL)

    assert preview == {
        "label": "demo#7",
        "title": "Add retry to the sync job",
        "url": PR_URL,
        "status": {"text": "CI failure", "tone": "err"},
        "badges": [
            {"text": "open", "tone": "ok"},
            {"text": "CI failure", "tone": "err"},
            {"text": "1 unresolved", "tone": "warn"},
            {"text": "changes requested", "tone": "warn"},
            {"text": "bug", "tone": "mute"},
            {"text": "sync", "tone": "mute"},
        ],
        "fields": [
            {"label": "forge", "value": "Forgejo, forge.example.test"},
            {"label": "repo", "value": "octo-org/demo"},
            {"label": "author", "value": "octocat"},
            {"label": "branch", "value": "main <- retry-sync"},
            {"label": "updated", "value": "2026-10-09T12:00:00Z"},
        ],
        "icon": "/api/plugins/forge/ui/icons/forgejo.svg",
    }
    assert _provider(adapter).mock_calls == []


def test_an_uncached_pull_request_is_read_live(adapter):
    _provider(adapter).get_pr.return_value = make_pr(number=40, state="merged", ci_state="none", labels=[])

    preview = adapter.link_preview(PR_URL.replace("/7", "/40"))

    assert preview["status"] == {"text": "merged", "tone": "info"}
    _provider(adapter).get_pr.assert_called_once_with("octo-org/demo", 40)


def test_an_issue_is_read_live(adapter):
    _provider(adapter, "gh").get_issue.return_value = Issue(
        repo="octo-org/demo",
        number=5,
        title="Upload test is flaky",
        state="closed",
        author="octocat",
        labels=["tests"],
        url="https://github.com/octo-org/demo/issues/5",
    )

    preview = adapter.link_preview("https://github.com/octo-org/demo/issues/5")

    assert preview == {
        "label": "demo#5",
        "title": "Upload test is flaky",
        "url": "https://github.com/octo-org/demo/issues/5",
        "status": {"text": "closed", "tone": "mute"},
        "badges": [{"text": "closed", "tone": "mute"}, {"text": "tests", "tone": "mute"}],
        "fields": [
            {"label": "forge", "value": "GitHub, github.com"},
            {"label": "repo", "value": "octo-org/demo"},
            {"label": "author", "value": "octocat"},
        ],
        "icon": "/api/plugins/forge/ui/icons/github.svg",
    }
    _provider(adapter, "gh").get_issue.assert_called_once_with("octo-org/demo", 5)


def test_a_live_read_is_kept_until_it_expires(adapter, clock):
    _provider(adapter).get_pr.return_value = make_pr(number=40, labels=[])
    url = PR_URL.replace("/7", "/40")

    first = adapter.link_preview(url)
    assert adapter.link_preview(url) == first
    _provider(adapter).get_pr.assert_called_once()

    clock.now += 61
    adapter.link_preview(url)
    assert _provider(adapter).get_pr.call_count == 2


def test_a_forge_error_is_no_preview(adapter):
    _provider(adapter).get_pr.side_effect = ForgeError("home: GET /repos/octo-org/demo/pulls/41 returned 404")

    assert adapter.link_preview(PR_URL.replace("/7", "/41")) is None


@pytest.mark.parametrize(
    "url",
    [
        "https://forge.example.test/other-org/demo/pulls/7",
        "https://github.com/octo-org/tools/pull/3",
        "https://elsewhere.example.test/octo-org/demo/pulls/7",
        "https://forge.example.test/octo-org/../pulls/7",
        "https://forge.example.test/octo-org/demo/commits/7",
        "https://github.com/octo-org/demo/pulls/7",
    ],
)
def test_links_outside_the_configured_repos_cause_no_call(adapter, url):
    assert adapter.link_preview(url) is None
    assert _provider(adapter).mock_calls == _provider(adapter, "gh").mock_calls == []


def test_the_token_never_reaches_the_preview_route(secrets_backend, clock, tmp_path):
    from tsugite_forge.providers.forgejo import ForgejoProvider

    adapter = make_adapter(clock, forges=("home",))
    server = FakeForgejo(
        {
            "/api/v1/repos/octo-org/demo/issues/3": {
                "number": 3,
                "title": f"leaked {TOKEN}",
                "state": "open",
                "user": {"login": "octocat"},
                "labels": [],
                "html_url": "https://forge.example.test/octo-org/demo/issues/3",
            }
        }
    )
    cache = adapter.forges["home"]
    cache.repos = ["octo-org/demo"]
    cache.provider = ForgejoProvider(cache.config, transport=server.transport)
    with attributing_to("forge"):
        register_link_resolver(name="links", hosts=["forge.example.test"], resolve=adapter.link_preview)
    client, token_store, _server = forge_http(adapter, tmp_path)

    resp = client.get(
        "/api/link-preview",
        params={"url": "https://forge.example.test/octo-org/demo/issues/3"},
        headers=admin_headers(token_store),
    )

    assert resp.json()["title"] == "leaked ***"
    assert resp.json()["icon"] == "/api/plugins/forge/ui/icons/forgejo.svg"
    assert TOKEN not in resp.text


def test_create_adapter_registers_a_resolver_for_the_configured_hosts():
    from forge_helpers import FORGES
    from tsugite_forge.adapter import create_adapter

    adapter = create_adapter(
        config={"forges": [FORGES["home"], FORGES["gh"]]}, runtime=None, session_store=None, identity_map={}
    )

    (resolver,) = registered_link_resolvers()
    assert resolver["hosts"] == ["forge.example.test", "github.com"]
    assert resolver["resolve"] == adapter.link_preview
