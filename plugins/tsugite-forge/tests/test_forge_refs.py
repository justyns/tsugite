"""Pure ref helpers: remote and mention parsing, the stored `refs` value, and pin/hide actions."""

import re
from types import SimpleNamespace

import pytest
from forge_helpers import git_checkout
from tsugite_forge.config import ForgePluginConfig

FORGES = ForgePluginConfig(
    forges=[
        {
            "name": "home",
            "kind": "forgejo",
            "url": "https://forge.example.test",
            "token_secret": "t",
            "repos": ["octo-org/*"],
        },
        {"name": "gh", "kind": "github", "repos": ["octo-org/*"]},
        {"name": "bb", "kind": "bitbucket", "email": "me@example.test", "token_secret": "t", "repos": ["octo-ws/*"]},
    ]
).forges


@pytest.mark.parametrize(
    "url, expected",
    [
        ("git@forge.example.test:octo-org/demo.git", ("home", "octo-org/demo")),
        ("git@forge.example.test:octo-org/demo", ("home", "octo-org/demo")),
        ("ssh://git@forge.example.test:2222/octo-org/demo.git", ("home", "octo-org/demo")),
        ("ssh://forge.example.test/octo-org/demo", ("home", "octo-org/demo")),
        ("https://forge.example.test/octo-org/demo.git", ("home", "octo-org/demo")),
        ("https://forge.example.test:3000/octo-org/demo", ("home", "octo-org/demo")),
        ("https://deploy@forge.example.test/octo-org/demo.git", ("home", "octo-org/demo")),
        ("https://forge.example.test/octo-org/demo/", ("home", "octo-org/demo")),
        ("https://forge.example.test/git/octo-org/demo.git", ("home", "octo-org/demo")),
        ("git@github.com:octo-org/site.git", ("gh", "octo-org/site")),
        ("https://github.com/octo-org/site", ("gh", "octo-org/site")),
        ("ssh://git@github.com/octo-org/site.git", ("gh", "octo-org/site")),
        ("git@bitbucket.org:octo-ws/api.git", ("bb", "octo-ws/api")),
        ("https://me@bitbucket.org/octo-ws/api.git", ("bb", "octo-ws/api")),
        ("git@gitlab.example.test:octo-org/demo.git", None),
        ("https://forge.example.test.other.test/octo-org/demo", None),
        ("https://forge.example.test/demo", None),
        ("/srv/git/demo.git", None),
        ("", None),
    ],
)
def test_remote_ref(url, expected):
    from tsugite_forge.refs import remote_ref

    assert remote_ref(url, FORGES) == expected


TEXT = """See https://forge.example.test/octo-org/demo/pulls/7 and https://github.com/octo-org/site/pull/12.
Issues https://forge.example.test/octo-org/demo/issues/3 and <https://github.com/octo-org/site/issues/4>.
Not ours: https://gitlab.example.test/octo-org/demo/-/merge_requests/1, https://github.com/octo-org/site/pulls/9,
https://forge.example.test/octo-org/demo/pull/10, prefixed https://forge.example.test/git/octo-org/demo/pulls/11.
Bare #5 here, but not issue#6 or #issuecomment-1."""


def test_mention_refs_reads_configured_forge_urls():
    from tsugite_forge.refs import mention_refs

    assert mention_refs(TEXT, FORGES, None) == [
        ("home", "octo-org/demo", 7),
        ("gh", "octo-org/site", 12),
        ("home", "octo-org/demo", 3),
        ("gh", "octo-org/site", 4),
        ("home", "octo-org/demo", 11),
    ]


@pytest.mark.parametrize(
    "url, expected",
    [
        ("https://bitbucket.org/octo-ws/api/pull-requests/12", ("bb", "octo-ws/api", "pull-requests", 12)),
        ("https://bitbucket.org/octo-ws/api/pull-requests/12/", ("bb", "octo-ws/api", "pull-requests", 12)),
        ("https://bitbucket.org/octo-ws/api/issues/3", None),
        ("https://bitbucket.org/octo-ws/api/pulls/12", None),
    ],
)
def test_bitbucket_link_ref(url, expected):
    from tsugite_forge.previews import LINK_PATH
    from tsugite_forge.refs import link_ref

    assert link_ref(url, FORGES) == expected
    if expected:
        assert re.search(LINK_PATH, url)


def test_bare_number_needs_a_single_pinned_repo():
    from tsugite_forge.refs import mention_refs

    assert mention_refs(TEXT, FORGES, ("home", "octo-org/demo"))[-1] == ("home", "octo-org/demo", 5)
    assert ("home", "octo-org/demo", 6) not in mention_refs(TEXT, FORGES, ("home", "octo-org/demo"))


PINNED = {"forge": "home", "repo": "octo-org/demo", "pr": 7, "source": "pinned"}
HIDDEN = {"forge": "gh", "repo": "octo-org/site", "pr": None, "hidden": True}


@pytest.mark.parametrize(
    "value, expected",
    [
        (None, []),
        ("https://forge.example.test/octo-org/demo/pulls/7", []),
        ({"forge": "home"}, []),
        ([PINNED, HIDDEN], [PINNED, HIDDEN]),
        (
            [
                1,
                "x",
                {"forge": "home"},
                {**PINNED, "pr": "7"},
                {**PINNED, "pr": True},
                {**PINNED, "source": "touched"},
                PINNED,
            ],
            [PINNED],
        ),
        ([{"forge": "home", "repo": "octo-org/demo", "source": "pinned"}], [{**PINNED, "pr": None}]),
    ],
)
def test_stored_refs_tolerates_any_value(value, expected):
    from tsugite_forge.refs import stored_refs

    assert stored_refs(value) == expected


def test_actions_pin_unpin_hide_unhide():
    from tsugite_forge.refs import apply_action

    refs = apply_action([], "pin", "home", "octo-org/demo", 7)
    assert refs == [PINNED]
    assert apply_action(refs, "pin", "home", "octo-org/demo", 7) == [PINNED]
    refs = apply_action(refs, "hide", "gh", "octo-org/site", None)
    assert refs == [PINNED, HIDDEN]
    assert apply_action(refs, "hide", "gh", "octo-org/site", None) == [PINNED, HIDDEN]
    assert apply_action(refs, "unpin", "home", "octo-org/demo", 7) == [HIDDEN]
    assert apply_action(refs, "unhide", "gh", "octo-org/site", None) == [PINNED]
    assert apply_action(refs, "unpin", "home", "octo-org/demo", None) == [PINNED, HIDDEN]


def test_unknown_action_is_refused():
    from tsugite_forge.refs import apply_action

    with pytest.raises(ValueError, match="merge"):
        apply_action([], "merge", "home", "octo-org/demo", 7)


def _event(event_type, **data):
    return SimpleNamespace(type=event_type, data=data)


def _resolve(spec):
    return ("home", spec.removeprefix("home:")) if spec.startswith("home:") else None


@pytest.mark.parametrize(
    "tool, arguments, expected",
    [
        ("read_file", {"path": "{ws}/demo/src/app.py"}, ("home", "octo-org/demo", None)),
        ("edit_file", {"path": "demo/src/app.py"}, ("home", "octo-org/demo", None)),
        ("spawn_job", {"prompt": "x", "repo": "demo"}, ("home", "octo-org/demo", None)),
        ("list_files", {"path": "."}, ("home", "octo-org/tools", None)),
        ("run", {"command": "cd demo && git push"}, None),
        ("read_file", {"path": "demo/" + "x" * 500 + "…"}, None),
        ("read_file", {"path": "{ws}/../plain/x.txt"}, None),
        ("forge_pr", {"repo": "home:octo-org/demo", "number": 7}, ("home", "octo-org/demo", 7)),
        ("forge_pin", {"repo": "home:octo-org/demo", "pr": "7"}, ("home", "octo-org/demo", 7)),
        ("forge_pin", {"repo": "home:octo-org/demo", "pr": True}, ("home", "octo-org/demo", None)),
        ("forge_pr", {"repo": "elsewhere:octo-org/demo", "number": 7}, None),
    ],
)
def test_derive_each_touched_source(tmp_path, tool, arguments, expected):
    from tsugite_forge.refs import derive, git_origin

    ws = git_checkout(tmp_path / "ws", "git@forge.example.test:octo-org/tools.git")
    git_checkout(ws / "demo", "https://forge.example.test/octo-org/demo.git")
    (tmp_path / "plain").mkdir()
    arguments = {k: v.replace("{ws}", str(ws)) if isinstance(v, str) else v for k, v in arguments.items()}
    chain = [
        [
            _event("session_start", workspace=str(ws)),
            _event("code_execution", tool_calls=[{"tool": tool, "arguments": arguments}]),
        ]
    ]
    assert derive(chain, FORGES, _resolve, None, git_origin) == ([("touched", expected)] if expected else [])


def test_relative_paths_need_a_workspace(tmp_path):
    from tsugite_forge.refs import derive, git_origin

    chain = [[_event("code_execution", tool_calls=[{"tool": "read_file", "arguments": {"path": "demo/a.py"}}])]]
    assert derive(chain, FORGES, _resolve, None, git_origin) == []


def test_merge_keeps_the_most_authoritative_source():
    from tsugite_forge.refs import merge

    key = ("home", "octo-org/demo", 7)
    hidden_pin = {"forge": "gh", "repo": "octo-org/demo", "pr": None, "hidden": True}
    rows = merge([hidden_pin], [("mentioned", key), ("touched", key)], {})
    assert [(r["pr"], r["source"], r["hidden"]) for r in rows] == [(7, "touched", False)]
