"""GitHub provider over the `gh` CLI and its login. GraphQL responses carry no ETag."""

import json
import subprocess
import time

from tsugite_forge.cache import LruCache
from tsugite_forge.providers.base import ForgeError, Provider, PullRequest, ReviewThreads, rest_issue

GH_TIMEOUT = 60
# Seconds one pull request's GraphQL node serves get_pr, checks and review_threads.
NODE_TTL = 5

PR_FIELDS = """
fragment PullRequestFields on PullRequest {
  number title url isDraft state createdAt updatedAt
  author { login }
  headRefName headRefOid baseRefName baseRefOid
  labels(first: 50) { nodes { name } }
  reviewRequests(first: 50) { nodes { requestedReviewer { ... on User { login } } } }
  reviewDecision
  reviewThreads(first: 100) { totalCount nodes { isResolved } }
  commits(last: 1) { nodes { commit { statusCheckRollup { state } } } }
}
"""

OPEN_PRS_QUERY = (
    """
query OpenPullRequests($owner: String!, $name: String!, $after: String) {
  repository(owner: $owner, name: $name) {
    pullRequests(states: OPEN, first: 50, after: $after) {
      pageInfo { hasNextPage endCursor }
      nodes { ...PullRequestFields }
    }
  }
}
"""
    + PR_FIELDS
)

PR_QUERY = (
    """
query PullRequest($owner: String!, $name: String!, $number: Int!) {
  repository(owner: $owner, name: $name) {
    pullRequest(number: $number) { ...PullRequestFields }
  }
}
"""
    + PR_FIELDS
)

OWNER_REPOS_QUERY = """
query OwnerRepos($owner: String!, $after: String) {
  repositoryOwner(login: $owner) {
    repositories(first: 100, after: $after) {
      pageInfo { hasNextPage endCursor }
      nodes { nameWithOwner isArchived }
    }
  }
}
"""

CI_STATES = {
    "SUCCESS": "success",
    "FAILURE": "failure",
    "ERROR": "failure",
    "PENDING": "pending",
    "EXPECTED": "pending",
}
REVIEW_STATES = {"APPROVED": "approved", "CHANGES_REQUESTED": "changes_requested"}


class GithubProvider(Provider):
    def __init__(self, config):
        super().__init__(config)
        self._nodes = LruCache(32)

    def _gh(self, *args: str) -> str:
        name = self.config.name
        try:
            result = subprocess.run(["gh", "api", *args], capture_output=True, text=True, timeout=GH_TIMEOUT)
        except FileNotFoundError:
            raise ForgeError(f"{name}: the gh CLI is not installed, install it from https://cli.github.com") from None
        except subprocess.TimeoutExpired:
            raise ForgeError(f"{name}: gh api {args[0]} timed out after {GH_TIMEOUT}s") from None
        # gh exits 4 when it has no login.
        if result.returncode == 4 or "HTTP 401" in result.stderr:
            raise ForgeError(f"{name}: gh is not logged in, run `gh auth login`")
        if result.returncode:
            raise ForgeError(f"{name}: gh api {args[0]} failed: {result.stderr.strip()}")
        return result.stdout

    def _graphql(self, query: str, **variables) -> dict:
        args = ["graphql", "-f", f"query={query}"]
        for key, value in variables.items():
            if value is not None:
                # -F sends an int as a number, -f sends a string as-is.
                args += ["-F" if isinstance(value, int) else "-f", f"{key}={value}"]
        return json.loads(self._gh(*args))["data"]

    def viewer(self):
        return self._graphql("query Viewer { viewer { login } }")["viewer"]["login"]

    def _owner_repos(self, owner):
        repos, after = [], None
        while True:
            data = self._graphql(OWNER_REPOS_QUERY, owner=owner, after=after)
            if data["repositoryOwner"] is None:
                raise ForgeError(f"{self.config.name}: no GitHub user or organization {owner!r}")
            page = data["repositoryOwner"]["repositories"]
            # A user's repositories include ones they only collaborate on.
            repos += [
                r["nameWithOwner"]
                for r in page["nodes"]
                if not r["isArchived"] and r["nameWithOwner"].lower().startswith(f"{owner.lower()}/")
            ]
            if not page["pageInfo"]["hasNextPage"]:
                return repos
            after = page["pageInfo"]["endCursor"]

    def _open_prs(self, repo):
        owner, name = repo.split("/")
        prs, after = [], None
        while True:
            page = self._graphql(OPEN_PRS_QUERY, owner=owner, name=name, after=after)["repository"]["pullRequests"]
            prs += [self._normalize(repo, node) for node in page["nodes"]]
            if not page["pageInfo"]["hasNextPage"]:
                return prs
            after = page["pageInfo"]["endCursor"]

    def _node(self, repo: str, number: int) -> dict:
        hit = self._nodes.get((repo, number))
        if hit is None or time.monotonic() - hit[1] > NODE_TTL:
            owner, name = repo.split("/")
            node = self._graphql(PR_QUERY, owner=owner, name=name, number=number)["repository"]["pullRequest"]
            hit = (node, time.monotonic())
            self._nodes.put((repo, number), hit)
        return hit[0]

    def get_pr(self, repo, number):
        return self._normalize(repo, self._node(repo, number))

    def get_issue(self, repo, number):
        return rest_issue(repo, json.loads(self._gh(f"repos/{repo}/issues/{number}")))

    def checks(self, repo, number):
        return _ci_state(self._node(repo, number))

    def review_threads(self, repo, number):
        return _threads(self._node(repo, number))

    def diff(self, repo, number):
        return self._gh(f"repos/{repo}/pulls/{number}", "-H", "Accept: application/vnd.github.diff")

    def _normalize(self, repo: str, node: dict) -> PullRequest:
        return PullRequest(
            forge=self.config.name,
            repo=repo,
            number=node["number"],
            title=node["title"],
            # GitHub shows a deleted account as "ghost".
            author=node["author"]["login"] if node["author"] else "ghost",
            state=node["state"].lower(),
            draft=node["isDraft"],
            head_ref=node["headRefName"],
            head_sha=node["headRefOid"],
            base_ref=node["baseRefName"],
            base_sha=node["baseRefOid"],
            labels=[label["name"] for label in node["labels"]["nodes"]],
            requested_reviewers=[
                r["requestedReviewer"]["login"] for r in node["reviewRequests"]["nodes"] if r["requestedReviewer"]
            ],
            ci_state=_ci_state(node),
            unresolved_threads=_threads(node).unresolved,
            review_state=REVIEW_STATES.get(node["reviewDecision"], "none"),
            created_at=node["createdAt"],
            updated_at=node["updatedAt"],
            url=node["url"],
        )


def _ci_state(node: dict) -> str:
    rollup = node["commits"]["nodes"][0]["commit"]["statusCheckRollup"]
    return CI_STATES[rollup["state"]] if rollup else "none"


def _threads(node: dict) -> ReviewThreads:
    threads = node["reviewThreads"]
    resolved = [thread["isResolved"] for thread in threads["nodes"]]
    # Threads past the first 100 are not fetched.
    unresolved = resolved.count(False) if len(resolved) == threads["totalCount"] else None
    return ReviewThreads(total=threads["totalCount"], unresolved=unresolved)
