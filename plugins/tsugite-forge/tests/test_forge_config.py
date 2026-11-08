"""tsugite-forge packaging, `plugins.forge` validation and the adapter factory."""

import importlib.metadata as md

import pytest
from pydantic import ValidationError

FORGEJO = {
    "name": "home",
    "kind": "forgejo",
    "url": "https://forge.example.test",
    "token_secret": "forgejo-token",
    "repos": ["octo-org/demo", "octo-org/*"],
}
GITHUB = {"name": "gh", "kind": "github", "repos": ["octo-org/*"]}
BITBUCKET = {
    "name": "bb",
    "kind": "bitbucket",
    "email": "me@example.test",
    "token_secret": "bitbucket-token",
    "repos": ["octo-ws/*"],
}
VALID = {"poll_minutes": 5, "forges": [FORGEJO, GITHUB, BITBUCKET]}


def _create(config):
    from tsugite_forge.adapter import create_adapter

    return create_adapter(config=config, runtime=None, session_store=None, identity_map={})


def _parse(config):
    from tsugite_forge.config import ForgePluginConfig

    return ForgePluginConfig(**config)


def test_adapter_entry_point_registered():
    assert "forge" in {ep.name for ep in md.entry_points(group="tsugite.adapters")}


def test_valid_config_parses_each_kind():
    config = _parse(VALID)
    assert config.poll_minutes == 5
    assert [(f.name, f.kind) for f in config.forges] == [("home", "forgejo"), ("gh", "github"), ("bb", "bitbucket")]
    assert config.forges[0].url == "https://forge.example.test"
    assert config.forges[0].token_secret == "forgejo-token"


def test_enabled_key_is_accepted():
    """The plugin loader reads `enabled` from the same block it hands the factory."""
    assert _parse({**VALID, "enabled": True}).forges


@pytest.mark.parametrize(
    "config, needle",
    [
        ({**VALID, "pol_minutes": 5}, "pol_minutes"),
        ({"forges": [{**GITHUB, "repo": ["octo-org/demo"]}]}, "repo"),
        ({"forges": [{**FORGEJO, "tokn_secret": "x"}]}, "tokn_secret"),
        ({"forges": [{**BITBUCKET, "url": "https://bitbucket.example.test"}]}, "url"),
    ],
)
def test_unknown_key_fails_with_its_name(config, needle):
    with pytest.raises(ValidationError) as raised:
        _parse(config)
    assert needle in str(raised.value)


@pytest.mark.parametrize("name", ["a:b", "a/b", ""])
def test_forge_name_is_letters_digits_dashes_and_underscores(name):
    with pytest.raises(ValidationError):
        _parse({"forges": [{**GITHUB, "name": name}]})


def test_forge_names_must_be_unique():
    with pytest.raises(ValidationError) as raised:
        _parse({"forges": [GITHUB, {**FORGEJO, "name": "gh"}]})
    assert "gh" in str(raised.value)


@pytest.mark.parametrize("pattern", ["octo-org", "octo-org/demo/extra", "*/demo", "octo-org/de*", "/demo", "octo-org/"])
def test_bad_repo_pattern_fails(pattern):
    with pytest.raises(ValidationError) as raised:
        _parse({"forges": [{**GITHUB, "repos": [pattern]}]})
    assert pattern in str(raised.value)


def test_no_forges_returns_none():
    assert _create({}) is None
    assert _create({"forges": []}) is None


def test_valid_config_returns_adapter():
    adapter = _create(VALID)
    assert [f.name for f in adapter.config.forges] == ["home", "gh", "bb"]
