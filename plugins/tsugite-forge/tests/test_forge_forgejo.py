"""Forgejo provider against a fake REST API built from hand-written fixtures."""

import logging
from dataclasses import asdict

import httpx
import pytest
from forge_helpers import TOKEN, FakeForgejo, forgejo_config


@pytest.fixture
def make(secrets_backend):
    def _make(overrides=None, **config):
        from tsugite_forge.providers.forgejo import ForgejoProvider

        server = FakeForgejo(overrides)
        return ForgejoProvider(forgejo_config(**config), transport=server.transport), server

    return _make


def test_list_repos_expands_owner_patterns_and_skips_archived_and_pr_disabled(make):
    provider, _ = make(repos=["octo-org/demo", "octo-org/*"])
    assert provider.list_repos() == ["octo-org/demo", "octo-org/tools"]


def test_owner_pattern_falls_back_to_user_repos(make):
    provider, _ = make(repos=["octocat/*"])
    assert provider.list_repos() == ["octocat/dotfiles"]


def _repo_pages(*sizes, total=None):
    """Routes serving org repo pages of `sizes`, as a server whose page limit is below 50 would."""
    routes, n = {}, 0
    for page, size in enumerate(sizes, 1):
        body = [{"full_name": f"octo-org/r{n + i}", "archived": False, "has_pull_requests": True} for i in range(size)]
        headers = {} if total is None else {"X-Total-Count": str(total)}
        routes[f"/api/v1/orgs/octo-org/repos?limit=50&page={page}"] = lambda request, body=body, headers=headers: (
            httpx.Response(200, json=body, headers=headers)
        )
        n += size
    return routes


def test_owner_repos_page_until_an_empty_page(make):
    provider, server = make(_repo_pages(20, 20, 5), repos=["octo-org/*"])
    assert len(provider.list_repos()) == 45
    assert [r.url.params["page"] for r in server.requests] == ["1", "2", "3", "4"]


def test_owner_repos_stop_at_the_total_count(make):
    provider, server = make(_repo_pages(20, 20, 5, total=45), repos=["octo-org/*"])
    assert len(provider.list_repos()) == 45
    assert [r.url.params["page"] for r in server.requests] == ["1", "2", "3"]


def test_list_prs_normalizes_open_pull_requests(make):
    provider, _ = make()
    first, second = provider.list_prs(["octo-org/demo"])
    assert asdict(first) == {
        "forge": "home",
        "repo": "octo-org/demo",
        "number": 7,
        "title": "Add retry to the sync job",
        "author": "octocat",
        "state": "open",
        "draft": False,
        "head_ref": "retry-sync",
        "head_sha": "7" * 40,
        "base_ref": "main",
        "base_sha": "1" * 40,
        "labels": ["bug", "sync"],
        "requested_reviewers": ["reviewer1"],
        "ci_state": "failure",
        "unresolved_threads": 1,
        "review_state": "changes_requested",
        "created_at": "2026-10-01T12:00:00Z",
        "updated_at": "2026-10-02T08:30:00Z",
        "url": "https://forge.example.test/octo-org/demo/pulls/7",
    }
    assert (second.number, second.draft, second.author) == (8, True, "reviewer1")
    assert (second.ci_state, second.review_state, second.unresolved_threads) == ("none", "none", 0)
    assert second.requested_reviewers == []


def test_list_prs_takes_an_explicit_repo_list(make):
    provider, _ = make()
    assert provider.list_prs(["octo-org/tools"]) == []


def test_get_pr_reports_merged_and_ignores_dismissed_reviews(make):
    provider, _ = make()
    pr = provider.get_pr("octo-org/demo", 9)
    assert (pr.state, pr.review_state, pr.ci_state) == ("merged", "approved", "success")


def test_checks_and_review_threads(make):
    from tsugite_forge.providers import ReviewThreads

    provider, _ = make()
    assert provider.checks("octo-org/demo", 7) == "failure"
    # sync.py:12 is resolved; sync.py:40 has a reply but no resolver.
    assert provider.review_threads("octo-org/demo", 7) == ReviewThreads(total=2, unresolved=1)


@pytest.mark.parametrize(
    "state, total, expected",
    [("success", 1, "success"), ("pending", 1, "pending"), ("error", 1, "failure"), ("warning", 1, "failure")],
)
def test_combined_status_maps_to_ci_state(make, state, total, expected):
    route = "/api/v1/repos/octo-org/demo/commits/7777777777777777777777777777777777777777/status"
    provider, _ = make({route: {"state": state, "total_count": total, "statuses": []}})
    assert provider.checks("octo-org/demo", 7) == expected


def test_diff_reads_the_dot_diff_endpoint(make):
    provider, _ = make()
    assert provider.diff("octo-org/demo", 7).startswith("diff --git a/sync.py b/sync.py")


def test_every_request_sends_the_token_from_the_secret_store(make):
    provider, server = make()
    provider.list_prs(["octo-org/demo"])
    assert server.requests
    assert {r.headers["authorization"] for r in server.requests} == {f"token {TOKEN}"}


def test_token_is_registered_for_masking(make):
    from tsugite.secrets.registry import get_registry

    provider, _ = make()
    provider.list_repos()
    provider.get_pr("octo-org/demo", 7)
    assert get_registry().mask(f"x {TOKEN} y") == "x *** y"


def test_etag_304_reuses_the_cached_body(make):
    pulls = "/api/v1/repos/octo-org/demo/pulls?state=open&limit=50&page=1"
    server_body = FakeForgejo().routes[pulls]

    def respond(request):
        if request.headers.get("if-none-match") == '"pulls-v1"':
            return httpx.Response(304, headers={"ETag": '"pulls-v1"'})
        return httpx.Response(200, json=server_body, headers={"ETag": '"pulls-v1"'})

    provider, server = make({pulls: respond})
    first = provider.list_prs(["octo-org/demo"])
    second = provider.list_prs(["octo-org/demo"])
    assert [pr.number for pr in second] == [pr.number for pr in first] == [7, 8]
    sent = [r.headers.get("if-none-match") for r in server.requests if r.url.raw_path.decode() == pulls]
    assert sent == [None, '"pulls-v1"']


def test_a_diff_is_not_kept_for_etag_revalidation(make):
    diff = "/api/v1/repos/octo-org/demo/pulls/7.diff"
    provider, server = make({diff: lambda request: httpx.Response(200, text="+x", headers={"ETag": '"diff-v1"'})})
    provider.diff("octo-org/demo", 7)
    provider.diff("octo-org/demo", 7)
    assert [r.headers.get("if-none-match") for r in server.requests] == [None, None]


def test_missing_secret_says_how_to_set_it(make):
    from tsugite_forge.providers import ForgeError

    provider, _ = make(token_secret="absent-token")
    with pytest.raises(ForgeError, match="tsu secrets set absent-token"):
        provider.list_prs(["octo-org/demo"])


def test_http_error_never_contains_the_token(make, caplog):
    from tsugite_forge.providers import ForgeError

    caplog.set_level(logging.DEBUG)
    echo = {"__status": 401, "message": f"token {TOKEN} is not valid"}
    provider, _ = make({"/api/v1/repos/octo-org/demo/pulls/7": echo})
    with pytest.raises(ForgeError) as raised:
        provider.get_pr("octo-org/demo", 7)
    assert "401" in str(raised.value)
    assert TOKEN not in str(raised.value)
    assert TOKEN not in caplog.text


def test_unreachable_server_is_a_forge_error(make):
    from tsugite_forge.providers import ForgeError

    def refuse(request):
        raise httpx.ConnectError("connection refused", request=request)

    provider, _ = make({"/api/v1/repos/octo-org/demo/pulls/7": refuse})
    with pytest.raises(ForgeError, match="connection refused"):
        provider.get_pr("octo-org/demo", 7)


def test_viewer_is_the_token_owner(make):
    provider, _ = make()
    assert provider.viewer() == "reviewer1"
