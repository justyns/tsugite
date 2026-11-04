"""Provider interface and the normalized pull request every forge kind returns."""

from abc import ABC, abstractmethod
from dataclasses import dataclass


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

    @abstractmethod
    def _open_prs(self, repo: str) -> list[PullRequest]: ...

    @abstractmethod
    def get_pr(self, repo: str, number: int) -> PullRequest: ...

    @abstractmethod
    def checks(self, repo: str, number: int) -> str: ...

    @abstractmethod
    def review_threads(self, repo: str, number: int) -> ReviewThreads: ...

    @abstractmethod
    def diff(self, repo: str, number: int) -> str: ...
