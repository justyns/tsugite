"""Provider interface and the normalized pull request every forge kind returns."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from tsugite.secrets import get_backend
from tsugite.secrets.registry import get_registry


class ForgeError(RuntimeError):
    pass


@dataclass
class PullRequest:
    forge: str
    repo: str
    number: int
    title: str
    author: str
    state: str  # open, closed or merged
    draft: bool
    head_ref: str
    head_sha: str
    base_ref: str
    base_sha: str
    labels: list[str]
    requested_reviewers: list[str]
    ci_state: str  # success, failure, pending or none
    unresolved_threads: int | None  # None when the forge cannot tell
    review_state: str  # approved, changes_requested or none
    created_at: str
    updated_at: str
    url: str


@dataclass
class Issue:
    repo: str
    number: int
    title: str
    state: str  # open or closed
    author: str
    labels: list[str]
    url: str


def rest_issue(repo: str, raw: dict) -> Issue:
    """An issue from the REST shape Forgejo and GitHub share."""
    return Issue(
        repo=repo,
        number=raw["number"],
        title=raw["title"],
        state=raw["state"],
        author=raw["user"]["login"],
        labels=[label["name"] for label in raw["labels"] or []],
        url=raw["html_url"],
    )


def secret_token(config) -> str:
    """Read `config.token_secret` from the secrets backend and register it for masking."""
    token = get_backend().get(config.token_secret)
    if token is None:
        raise ForgeError(
            f"{config.name}: secret {config.token_secret!r} not found in secrets backend,"
            f" set it with `tsu secrets set {config.token_secret}`"
        )
    return get_registry().register(config.token_secret, token, agent="forge")


@dataclass
class ReviewThreads:
    total: int
    unresolved: int | None


class Provider(ABC):
    def __init__(self, config):
        self.config = config

    def list_repos(self) -> list[str]:
        """Expand the configured patterns into `owner/repo` names."""
        repos = []
        for pattern in self.config.repos:
            owner, name = pattern.split("/")
            for repo in self._owner_repos(owner) if name == "*" else [pattern]:
                if repo not in repos:
                    repos.append(repo)
        return repos

    def list_prs(self, repos: list[str]) -> list[PullRequest]:
        return [pr for repo in repos for pr in self._open_prs(repo)]

    @abstractmethod
    def viewer(self) -> str:
        """Login of the account the forge is read as."""

    @abstractmethod
    def _owner_repos(self, owner: str) -> list[str]: ...

    def _open_prs(self, repo: str) -> list[PullRequest]:
        raise NotImplementedError

    @abstractmethod
    def get_pr(self, repo: str, number: int) -> PullRequest: ...

    @abstractmethod
    def get_issue(self, repo: str, number: int) -> Issue: ...

    @abstractmethod
    def checks(self, repo: str, number: int) -> str: ...

    @abstractmethod
    def review_threads(self, repo: str, number: int) -> ReviewThreads: ...

    @abstractmethod
    def diff(self, repo: str, number: int) -> str: ...
