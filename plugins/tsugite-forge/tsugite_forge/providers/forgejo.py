"""Forgejo provider over the REST API, authenticated with a token from the secrets backend."""

import httpx

from tsugite.secrets.registry import get_registry
from tsugite_forge.cache import LruCache
from tsugite_forge.providers.base import ForgeError, Provider, PullRequest, ReviewThreads, rest_issue, secret_token

# Forgejo's default [api] MAX_RESPONSE_ITEMS. A larger limit is capped to it.
PAGE_SIZE = 50
TIMEOUT = 30.0
MAX_ETAGS = 1000

REVIEW_STATES = {"APPROVED": "approved", "REQUEST_CHANGES": "changes_requested"}


class _NotFoundError(ForgeError):
    pass


class ForgejoProvider(Provider):
    def __init__(self, config, transport: httpx.BaseTransport | None = None):
        super().__init__(config)
        self._client = httpx.Client(base_url=f"{config.url.rstrip('/')}/api/v1", transport=transport, timeout=TIMEOUT)
        self._token: str | None = None
        self._etags = LruCache(MAX_ETAGS)

    def _authorization(self) -> str:
        if self._token is None:
            self._token = secret_token(self.config)
        return f"token {self._token}"

    def _get(self, path: str, params: dict | None = None, etag: bool = True) -> httpx.Response:
        key = str(httpx.URL(path, params=params))
        headers = {"Authorization": self._authorization()}
        cached = self._etags.get(key) if etag else None
        if cached:
            headers["If-None-Match"] = cached[0]
        try:
            response = self._client.get(path, params=params, headers=headers)
        except httpx.HTTPError as exc:
            raise ForgeError(get_registry().mask(f"{self.config.name}: GET {path} failed: {exc}")) from None
        if response.status_code == 304 and cached:
            return cached[1]
        if response.status_code >= 400:
            message = get_registry().mask(
                f"{self.config.name}: GET {path} returned {response.status_code}: {response.text[:200]}"
            )
            raise (_NotFoundError if response.status_code == 404 else ForgeError)(message)
        if etag and "etag" in response.headers:
            self._etags.put(key, (response.headers["etag"], response))
        return response

    def _pages(self, path: str, **params) -> list:
        # A server may cap pages below PAGE_SIZE. Stop on an empty page or at x-total-count.
        items, page = [], 1
        while True:
            response = self._get(path, {**params, "limit": PAGE_SIZE, "page": page})
            batch = response.json()
            items += batch
            total = response.headers.get("x-total-count")
            if not batch or (total is not None and len(items) >= int(total)):
                return items
            page += 1

    def viewer(self):
        return self._get("/user").json()["login"]

    def _owner_repos(self, owner):
        try:
            repos = self._pages(f"/orgs/{owner}/repos")
        except _NotFoundError:
            repos = self._pages(f"/users/{owner}/repos")
        return [repo["full_name"] for repo in repos if not repo["archived"] and repo["has_pull_requests"]]

    def _open_prs(self, repo):
        return [self._normalize(repo, raw) for raw in self._pages(f"/repos/{repo}/pulls", state="open")]

    def get_pr(self, repo, number):
        return self._normalize(repo, self._get(f"/repos/{repo}/pulls/{number}").json())

    def get_issue(self, repo, number):
        return rest_issue(repo, self._get(f"/repos/{repo}/issues/{number}").json())

    def checks(self, repo, number):
        return self._ci_state(repo, self._get(f"/repos/{repo}/pulls/{number}").json()["head"]["sha"])

    def review_threads(self, repo, number):
        return self._threads(repo, number, self._pages(f"/repos/{repo}/pulls/{number}/reviews"))

    def diff(self, repo, number):
        # Diffs are unbounded, so they skip the ETag cache.
        return self._get(f"/repos/{repo}/pulls/{number}.diff", etag=False).text

    def _normalize(self, repo: str, raw: dict) -> PullRequest:
        number = raw["number"]
        reviews = self._pages(f"/repos/{repo}/pulls/{number}/reviews")
        # `mergeable` is left out: Forgejo recomputes it in the background after a push.
        return PullRequest(
            forge=self.config.name,
            repo=repo,
            number=number,
            title=raw["title"],
            author=raw["user"]["login"],
            state="merged" if raw["merged"] else raw["state"],
            draft=raw.get("draft", False),
            head_ref=raw["head"]["ref"],
            head_sha=raw["head"]["sha"],
            base_ref=raw["base"]["ref"],
            base_sha=raw["base"]["sha"],
            labels=[label["name"] for label in raw["labels"] or []],
            requested_reviewers=[user["login"] for user in raw["requested_reviewers"] or []],
            ci_state=self._ci_state(repo, raw["head"]["sha"]),
            unresolved_threads=self._threads(repo, number, reviews).unresolved,
            review_state=_review_state(reviews),
            created_at=raw["created_at"],
            updated_at=raw["updated_at"],
            url=raw["html_url"],
        )

    def _ci_state(self, repo: str, sha: str) -> str:
        status = self._get(f"/repos/{repo}/commits/{sha}/status").json()
        if not status["total_count"]:
            return "none"
        return status["state"] if status["state"] in ("success", "pending") else "failure"

    def _threads(self, repo: str, number: int, reviews: list) -> ReviewThreads:
        # Forgejo groups a conversation by file and line, and resolving it sets
        # `resolver` on one comment in it (the first), not on every reply.
        conversations: dict[tuple, bool] = {}
        for review in reviews:
            if not review["comments_count"]:
                continue
            for comment in self._get(f"/repos/{repo}/pulls/{number}/reviews/{review['id']}/comments").json():
                key = (comment["path"], comment["position"], comment["original_position"])
                conversations[key] = conversations.get(key, False) or comment["resolver"] is not None
        return ReviewThreads(total=len(conversations), unresolved=list(conversations.values()).count(False))


def _review_state(reviews: list) -> str:
    """Combine each reviewer's latest approving or blocking review that is still standing."""
    latest = {}
    for review in reviews:
        if review["state"] in REVIEW_STATES and not review["dismissed"]:
            latest[review["user"]["login"]] = review["state"]
    if "REQUEST_CHANGES" in latest.values():
        return "changes_requested"
    return "approved" if latest else "none"
