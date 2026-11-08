"""Bitbucket Cloud provider over the REST API."""

import httpx

from tsugite.secrets.registry import get_registry
from tsugite_forge.providers.base import ForgeError, Provider, PullRequest, ReviewThreads, secret_token

API_URL = "https://api.bitbucket.org/2.0"
TIMEOUT = 30.0
# The pull request list leaves these out unless asked for.
PR_FIELDS = "+values.participants,+values.reviewers"

STATES = {"OPEN": "open", "MERGED": "merged", "DECLINED": "closed", "SUPERSEDED": "closed"}


def _login(account: dict) -> str:
    # App accounts (bots) have no nickname.
    return account.get("nickname") or account["display_name"]


class BitbucketProvider(Provider):
    def __init__(self, config, transport: httpx.BaseTransport | None = None):
        super().__init__(config)
        # The diff endpoint answers with a redirect.
        self._client = httpx.Client(base_url=API_URL, transport=transport, timeout=TIMEOUT, follow_redirects=True)
        self._auth: httpx.BasicAuth | None = None
        self._me: dict | None = None
        # (repo, number) -> (updated_on, ci_state, unresolved_threads) from the latest poll.
        self._polled: dict[tuple[str, int], tuple[str, str, int]] = {}

    def _get(self, path: str, params: dict | None = None) -> httpx.Response:
        if self._auth is None:
            self._auth = httpx.BasicAuth(self.config.email, secret_token(self.config))
        try:
            response = self._client.get(path, params=params, auth=self._auth)
        except httpx.HTTPError as exc:
            raise ForgeError(get_registry().mask(f"{self.config.name}: GET {path} failed: {exc}")) from None
        if response.status_code >= 400:
            raise ForgeError(
                get_registry().mask(
                    f"{self.config.name}: GET {path} returned {response.status_code}: {response.text[:200]}"
                )
            )
        return response

    def _pages(self, path: str, **params) -> list:
        # `next` is an absolute URL that already holds the query.
        items, url = [], path
        while url:
            page = self._get(url, params).json()
            items += page["values"]
            url, params = page.get("next"), None
        return items

    def _account(self) -> dict:
        if self._me is None:
            self._me = self._get("/user").json()
        return self._me

    def viewer(self):
        return _login(self._account())

    def _owner_repos(self, owner):
        return [repo["full_name"] for repo in self._pages(f"/repositories/{owner}", pagelen=100)]

    def list_prs(self, repos):
        # Bitbucket has no workspace-wide listing of PRs you review.
        uuid = self._account()["uuid"]
        raws = [
            raw
            for workspace in dict.fromkeys(repo.split("/")[0] for repo in repos)
            for raw in self._pages(
                f"/workspaces/{workspace}/pullrequests/{uuid}", state="OPEN", pagelen=50, fields=PR_FIELDS
            )
            if raw["destination"]["repository"]["full_name"] in repos
        ]
        # `updated_on` does not move on a new commit status, so CI lags until the PR is next updated.
        last, self._polled = self._polled, {}
        prs = []
        for raw in raws:
            repo = raw["destination"]["repository"]["full_name"]
            hit = last.get((repo, raw["id"]))
            if not hit or hit[0] != raw["updated_on"]:
                hit = (raw["updated_on"], *self._derived(repo, raw))
            self._polled[repo, raw["id"]] = hit
            prs.append(self._normalize(repo, raw, *hit[1:]))
        return prs

    def get_pr(self, repo, number):
        raw = self._get(f"/repositories/{repo}/pullrequests/{number}").json()
        return self._normalize(repo, raw, *self._derived(repo, raw))

    def get_issue(self, repo, number):
        raise ForgeError(f"{self.config.name}: Bitbucket Cloud has no issue tracker")

    def checks(self, repo, number):
        statuses = self._pages(f"/repositories/{repo}/pullrequests/{number}/statuses", pagelen=100)
        # Keep the newest run of each build key.
        latest = {s["key"]: s["state"] for s in sorted(statuses, key=lambda s: s["updated_on"])}
        states = set(latest.values())
        if not states:
            return "none"
        if states & {"FAILED", "STOPPED"}:
            return "failure"
        return "pending" if "INPROGRESS" in states else "success"

    def review_threads(self, repo, number):
        comments = self._pages(f"/repositories/{repo}/pullrequests/{number}/comments", pagelen=100)
        threads = [
            c for c in comments if c.get("inline") and c.get("parent") is None and not c["deleted"] and not c["pending"]
        ]
        unresolved = [c for c in threads if c.get("resolution") is None]
        return ReviewThreads(total=len(threads), unresolved=len(unresolved))

    def diff(self, repo, number):
        return self._get(f"/repositories/{repo}/pullrequests/{number}/diff").text

    def _derived(self, repo: str, raw: dict) -> tuple[str, int]:
        return self.checks(repo, raw["id"]), (
            self.review_threads(repo, raw["id"]).unresolved if raw["comment_count"] else 0
        )

    def _normalize(self, repo: str, raw: dict, ci_state: str, unresolved_threads: int) -> PullRequest:
        number = raw["id"]
        reviewed = {_login(p["user"]): p["state"] for p in raw["participants"] if p["state"]}
        if "changes_requested" in reviewed.values():
            review_state = "changes_requested"
        else:
            review_state = "approved" if reviewed else "none"
        return PullRequest(
            forge=self.config.name,
            repo=repo,
            number=number,
            title=raw["title"],
            author=_login(raw["author"]),
            state=STATES[raw["state"]],
            draft=raw["draft"],
            head_ref=raw["source"]["branch"]["name"],
            head_sha=raw["source"]["commit"]["hash"],
            base_ref=raw["destination"]["branch"]["name"],
            base_sha=raw["destination"]["commit"]["hash"],
            labels=[],
            requested_reviewers=[_login(r) for r in raw["reviewers"] if _login(r) not in reviewed],
            ci_state=ci_state,
            unresolved_threads=unresolved_threads,
            review_state=review_state,
            created_at=raw["created_on"],
            updated_at=raw["updated_on"],
            url=raw["links"]["html"]["href"],
        )
