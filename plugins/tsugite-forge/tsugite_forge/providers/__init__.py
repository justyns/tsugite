"""Forge providers, keyed by the `kind` a forge is configured with."""

from tsugite_forge.providers.base import ForgeError, Provider, PullRequest, ReviewThreads
from tsugite_forge.providers.forgejo import ForgejoProvider
from tsugite_forge.providers.github import GithubProvider

kinds: dict[str, type[Provider]] = {"forgejo": ForgejoProvider, "github": GithubProvider}

__all__ = ["ForgeError", "Provider", "PullRequest", "ReviewThreads", "kinds"]
