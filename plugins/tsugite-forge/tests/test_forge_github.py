"""GitHub provider against a fake `gh` executable that replays hand-written fixtures."""

import json
from dataclasses import asdict

import pytest
from forge_helpers import FIXTURES, FakeGh, github_config


@pytest.fixture
def make(tmp_path, monkeypatch):
    def _make(responses=None, **config):
        from tsugite_forge.providers.github import GithubProvider

        gh = FakeGh(tmp_path, monkeypatch, responses)
        return GithubProvider(github_config(**config)), gh

    return _make


def test_list_repos_expands_owner_and_skips_archived(make):
    provider, _ = make(repos=["octo-org/*", "octo-org/demo"])
    assert provider.list_repos() == ["octo-org/demo"]


def test_unknown_owner_is_a_forge_error(make):
    from tsugite_forge.providers import ForgeError

    provider, _ = make(repos=["nobody-here/*"])
    with pytest.raises(ForgeError, match="nobody-here"):
        provider.list_repos()


def test_list_prs_normalizes_open_pull_requests(make):
    provider, gh = make()
    first, second = provider.list_prs(["octo-org/demo"])
    assert asdict(first) == {
        "forge": "gh",
        "repo": "octo-org/demo",
        "number": 12,
        "title": "Fix flaky upload test",
        "author": "octocat",
        "state": "open",
        "draft": False,
        "head_ref": "fix-upload",
        "head_sha": "c" * 40,
        "base_ref": "main",
        "base_sha": "d" * 40,
        "labels": ["tests"],
        "requested_reviewers": ["reviewer1"],
        "ci_state": "failure",
        "unresolved_threads": 2,
        "review_state": "changes_requested",
        "created_at": "2026-10-05T14:00:00Z",
        "updated_at": "2026-10-06T09:00:00Z",
        "url": "https://github.com/octo-org/demo/pull/12",
    }
    assert (second.number, second.draft, second.ci_state, second.review_state) == (13, True, "none", "none")
    assert (second.unresolved_threads, second.labels, second.requested_reviewers) == (0, [], [])
    assert all(call[0] == "api" for call in gh.calls)


def test_list_prs_follows_pages(make):
    fixtures = json.loads((FIXTURES / "github.json").read_text())
    page_one = next(e for e in fixtures if "query OpenPullRequests" in e["match"])
    connection = page_one["stdout"]["data"]["repository"]["pullRequests"]
    connection["pageInfo"] = {"hasNextPage": True, "endCursor": "CURSOR1"}
    page_two = {
        "match": ["query OpenPullRequests", "after=CURSOR1"],
        "stdout": {
            "data": {
                "repository": {
                    "pullRequests": {
                        "pageInfo": {"hasNextPage": False, "endCursor": None},
                        "nodes": [{**connection["nodes"][1], "number": 15}],
                    }
                }
            }
        },
    }
    provider, _ = make([page_two, *fixtures])
    assert [pr.number for pr in provider.list_prs(["octo-org/demo"])] == [12, 13, 15]


def test_get_pr_maps_merged_expected_and_a_deleted_author(make):
    provider, _ = make()
    pr = provider.get_pr("octo-org/demo", 14)
    assert (pr.state, pr.ci_state, pr.review_state, pr.author) == ("merged", "pending", "approved", "ghost")


def test_review_threads_beyond_the_fetched_page_are_unknown(make):
    from tsugite_forge.providers import ReviewThreads

    provider, _ = make()
    assert provider.review_threads("octo-org/demo", 14) == ReviewThreads(total=101, unresolved=None)
    assert provider.get_pr("octo-org/demo", 14).unresolved_threads is None


def test_checks_and_review_threads(make):
    from tsugite_forge.providers import ReviewThreads

    provider, _ = make()
    assert provider.checks("octo-org/demo", 12) == "failure"
    assert provider.review_threads("octo-org/demo", 12) == ReviewThreads(total=3, unresolved=2)


def test_diff_asks_for_the_diff_media_type(make):
    provider, gh = make()
    assert provider.diff("octo-org/demo", 12).startswith("diff --git a/upload_test.py")
    assert gh.calls[-1] == ["api", "repos/octo-org/demo/pulls/12", "-H", "Accept: application/vnd.github.diff"]


def test_missing_gh_is_a_setup_error(make, tmp_path, monkeypatch):
    from tsugite_forge.providers import ForgeError
    from tsugite_forge.providers.github import GithubProvider

    empty = tmp_path / "empty-bin"
    empty.mkdir()
    monkeypatch.setenv("PATH", str(empty))
    with pytest.raises(ForgeError, match="install"):
        GithubProvider(github_config()).list_prs(["octo-org/demo"])


@pytest.mark.parametrize(
    "exit_code, stderr",
    [
        (4, "To get started with GitHub CLI, please run:  gh auth login\n"),
        (1, "gh: Bad credentials (HTTP 401)\n"),
    ],
)
def test_logged_out_gh_is_a_setup_error(make, exit_code, stderr):
    from tsugite_forge.providers import ForgeError

    provider, _ = make([{"match": [], "stderr": stderr, "exit": exit_code}])
    with pytest.raises(ForgeError, match="gh auth login"):
        provider.list_prs(["octo-org/demo"])


def test_other_gh_failures_are_forge_errors(make):
    from tsugite_forge.providers import ForgeError

    provider, _ = make()
    with pytest.raises(ForgeError, match="HTTP 404") as raised:
        provider.diff("octo-org/demo", 404)
    assert "gh auth login" not in str(raised.value)


def test_viewer_is_the_gh_login(make):
    provider, _ = make()
    assert provider.viewer() == "octocat"
