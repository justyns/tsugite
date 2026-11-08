"""Bitbucket Cloud provider against a fake REST API built from a hand-written fixture."""

import base64
from dataclasses import asdict

import httpx
import pytest
from forge_helpers import TOKEN, FakeBitbucket, bitbucket_config

DEMO = "/2.0/repositories/octo-ws/demo"


@pytest.fixture
def make(secrets_backend):
    def _make(overrides=None, **config):
        from tsugite_forge.providers.bitbucket import BitbucketProvider

        server = FakeBitbucket(overrides)
        return BitbucketProvider(bitbucket_config(**config), transport=server.transport), server

    return _make


def test_list_repos_expands_a_workspace_across_pages(make):
    provider, _ = make(repos=["octo-ws/*", "octo-ws/demo"])
    assert provider.list_repos() == ["octo-ws/demo", "octo-ws/tools", "octo-ws/site"]


def test_unknown_workspace_is_a_forge_error(make):
    from tsugite_forge.providers import ForgeError

    provider, _ = make(repos=["nobody-here/*"])
    with pytest.raises(ForgeError, match="nobody-here"):
        provider.list_repos()


def test_requests_send_the_email_and_token_as_basic_auth(make):
    provider, server = make()
    assert provider.viewer() == "octocat"
    expected = base64.b64encode(f"me@example.test:{TOKEN}".encode()).decode()
    assert server.requests[0].headers["authorization"] == f"Basic {expected}"


def test_list_prs_normalizes_open_pull_requests(make):
    provider, _ = make()
    first, second = provider.list_prs(["octo-ws/demo"])
    assert asdict(first) == {
        "forge": "bb",
        "repo": "octo-ws/demo",
        "number": 12,
        "title": "Fix flaky upload test",
        "author": "octocat",
        "state": "open",
        "draft": False,
        "head_ref": "fix-upload",
        "head_sha": "c" * 12,
        "base_ref": "main",
        "base_sha": "d" * 12,
        "labels": [],
        "requested_reviewers": ["reviewer1", "Deploy Bot"],
        "ci_state": "failure",
        "unresolved_threads": 1,
        "review_state": "changes_requested",
        "created_at": "2026-10-05T14:00:00.000000+00:00",
        "updated_at": "2026-10-06T09:00:00.000000+00:00",
        "url": "https://bitbucket.org/octo-ws/demo/pull-requests/12",
    }
    assert (second.number, second.author, second.draft, second.ci_state) == (13, "octocat", True, "none")
    assert (second.review_state, second.unresolved_threads, second.requested_reviewers) == ("none", 0, [])


def test_list_prs_lists_authored_prs_in_watched_repos_only(make):
    provider, server = make()
    assert [(pr.repo, pr.number) for pr in provider.list_prs(["octo-ws/demo"])] == [
        ("octo-ws/demo", 12),
        ("octo-ws/demo", 13),
    ]
    assert not any("/repositories/octo-ws/demo/pullrequests?" in str(r.url) for r in server.requests)


def _detail_paths(server):
    paths = [r.url.path for r in server.requests]
    return [p.removeprefix(f"{DEMO}/pullrequests/") for p in paths if p.endswith(("/statuses", "/comments"))]


def test_list_prs_reuses_checks_and_threads_while_a_pr_is_unchanged(make):
    provider, server = make()
    first = provider.list_prs(["octo-ws/demo"])
    assert _detail_paths(server) == ["12/statuses", "12/comments", "13/statuses"]
    server.requests.clear()
    assert provider.list_prs(["octo-ws/demo"]) == first
    assert _detail_paths(server) == []


def test_list_prs_refetches_checks_and_threads_when_a_pr_is_updated(make):
    provider, server = make()
    provider.list_prs(["octo-ws/demo"])
    server.requests.clear()
    listing = server.routes[
        "/2.0/workspaces/octo-ws/pullrequests/%7Boctocat%7D?state=OPEN&pagelen=50&fields=%2Bvalues.participants%2C%2Bvalues.reviewers"
    ]
    listing["values"][0]["updated_on"] = "2026-10-08T10:00:00.000000+00:00"
    provider.list_prs(["octo-ws/demo"])
    assert _detail_paths(server) == ["12/statuses", "12/comments"]


@pytest.mark.parametrize(
    "number, state, ci_state, review_state",
    [(14, "merged", "pending", "approved"), (15, "closed", "failure", "none")],
)
def test_get_pr_maps_states(make, number, state, ci_state, review_state):
    provider, _ = make()
    pr = provider.get_pr("octo-ws/demo", number)
    assert (pr.state, pr.ci_state, pr.review_state) == (state, ci_state, review_state)
    assert pr.requested_reviewers == []


def test_checks_and_review_threads(make):
    from tsugite_forge.providers import ReviewThreads

    provider, _ = make()
    assert provider.checks("octo-ws/demo", 12) == "failure"
    assert provider.review_threads("octo-ws/demo", 12) == ReviewThreads(total=2, unresolved=1)


def test_diff_follows_the_redirect(make):
    target = f"{DEMO}/diff/octo-ws/demo:{'c' * 12}%0D{'d' * 12}"
    provider, _ = make({f"{DEMO}/pullrequests/12/diff": lambda r: httpx.Response(302, headers={"Location": target})})
    assert provider.diff("octo-ws/demo", 12).startswith("diff --git a/upload.py")


def test_errors_mask_the_token(make):
    from tsugite_forge.providers import ForgeError

    provider, _ = make({"/2.0/user": {"__status": 401, "error": {"message": f"bad token {TOKEN}"}}})
    with pytest.raises(ForgeError, match="401") as raised:
        provider.viewer()
    assert TOKEN not in str(raised.value)
