"""Configuration model for the `plugins.forge` block of daemon.yaml."""

import re
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

REPO_PATTERN = re.compile(r"[\w.-]+/([\w.-]+|\*)")
NAME_PATTERN = r"^[A-Za-z0-9_-]+$"


class _Forge(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(pattern=NAME_PATTERN)
    repos: list[str]

    @field_validator("repos")
    @classmethod
    def _validate_repos(cls, repos: list[str]) -> list[str]:
        for pattern in repos:
            if not REPO_PATTERN.fullmatch(pattern):
                raise ValueError(f"repo pattern {pattern!r} must be owner/repo or owner/*")
        return repos


class ForgejoForge(_Forge):
    kind: Literal["forgejo"]
    url: str
    token_secret: str


class GithubForge(_Forge):
    """Talks to GitHub through the `gh` CLI and its login."""

    kind: Literal["github"]


Forge = Annotated[ForgejoForge | GithubForge, Field(discriminator="kind")]


class ForgePluginConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Read by the plugin loader, which hands the factory the same block.
    enabled: bool = True
    poll_minutes: int = Field(5, ge=1)
    forges: list[Forge] = []

    @model_validator(mode="after")
    def _validate_unique_names(self) -> "ForgePluginConfig":
        names = [forge.name for forge in self.forges]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"forge names must be unique, repeated: {', '.join(duplicates)}")
        return self
