"""The agent tools over the adapter's cache, with live provider calls for PR detail."""

import re
import subprocess
import sys
from datetime import UTC, datetime

import pytest
from forge_helpers import TOKEN, FakeForgejo, make_adapter, make_pr, seed

NOW = datetime(2026, 10, 10, 12, 0, tzinfo=UTC).timestamp()


@pytest.fixture
def adapter(secrets_backend):
    from tsugite_forge.providers.forgejo import ForgejoProvider
    from tsugite_forge.tools import set_forge_adapter

    adapter = make_adapter(lambda: NOW)
    seed(adapter, NOW)
    home = adapter.forges["home"]
    adapter.server = FakeForgejo()
    home.provider = ForgejoProvider(home.config, transport=adapter.server.transport)
    set_forge_adapter(adapter)
    yield adapter
    set_forge_adapter(None)


def tools():
    from tsugite_forge import tools

    return tools


def test_tools_need_the_daemon():
    from tsugite.exceptions import ToolUnavailableError

    tools().set_forge_adapter(None)
    for call in (lambda: tools().forge_prs(), lambda: tools().forge_repos(), lambda: tools().forge_pr("x/y", 1)):
        with pytest.raises(ToolUnavailableError, match="daemon"):
            call()


def test_forge_prs_prints_one_line_per_pull_request(adapter):
    assert tools().forge_prs().splitlines() == [
        "gh:octo-org/demo#12 Fix flaky upload test · octocat · 30m · CI success · threads ? · tests",
        "home:octo-org/demo#8 Rework the settings page · reviewer1 · 3h · CI none · 0 unresolved · draft",
        "home:octo-org/demo#7 Add retry to the sync job · octocat · 9d · CI failure · 1 unresolved · bug, sync",
    ]


@pytest.mark.parametrize(
    "kwargs, numbers",
    [
        ({"author": "reviewer1"}, [8]),
        ({"label": "bug"}, [7]),
        ({"needs_review": True}, [12, 7]),
        ({"repo": "octo-org/demo"}, [12, 8, 7]),
        ({"repo": "octo-org/tools"}, []),
    ],
)
def test_forge_prs_passes_its_filters_through(adapter, kwargs, numbers):
    assert [int(n) for n in re.findall(r"#(\d+) ", tools().forge_prs(**kwargs))] == numbers


def test_forge_prs_says_when_nothing_matches(adapter):
    assert tools().forge_prs(label="nope") == "No open pull requests match."


def test_forge_prs_only_reads_open_pull_requests(adapter):
    with pytest.raises(ValueError, match="open"):
        tools().forge_prs(state="closed")


def test_forge_prs_caps_rows_and_says_so(adapter):
    adapter.forges["gh"].prs = [make_pr(forge="gh", number=n) for n in range(100, 160)]
    lines = tools().forge_prs(repo="gh:octo-org/demo").splitlines()
    assert len(lines) == 51
    assert lines[-1] == "Showing 50 of 60. Narrow with repo, author, label or needs_review."


def test_forge_prs_reports_a_failing_forge(adapter):
    adapter.forges["gh"].error = "gh: gh is not logged in, run `gh auth login`"
    assert tools().forge_prs().splitlines()[-1] == (
        "gh: refresh failed, showing a 2m old cache: gh: gh is not logged in, run `gh auth login`"
    )


def test_tools_report_a_forge_that_never_refreshed(adapter):
    gh = adapter.forges["gh"]
    gh.last_refresh, gh.repos, gh.prs, gh.error = None, [], [], "gh: boom"
    assert tools().forge_prs().splitlines()[-1] == "gh: refresh failed, nothing cached yet: gh: boom"
    assert tools().forge_repos().splitlines()[-2:] == [
        "gh (github), never refreshed: no repos",
        "  last refresh failed: gh: boom",
    ]


@pytest.mark.parametrize(
    "repo", ["home:../..", "home:octo-org/demo?x=1", "home:octo-org/demo#x", "../..", "nowhere:a/b"]
)
def test_forge_pr_only_reaches_a_listed_repo(adapter, repo):
    with pytest.raises(ValueError):
        tools().forge_pr(repo, 1)
    assert adapter.server.requests == []


def test_forge_pr_needs_a_forge_when_the_repo_is_on_two(adapter):
    with pytest.raises(ValueError, match="home:octo-org/demo"):
        tools().forge_pr("octo-org/demo", 7)


def test_forge_pr_rejects_a_repo_no_forge_carries(adapter):
    with pytest.raises(ValueError, match="octo-org/missing"):
        tools().forge_pr("octo-org/missing", 7)


def test_forge_pr_shows_detail_and_live_parts(adapter):
    text = tools().forge_pr("home:octo-org/demo", 7)
    lines = text.splitlines()
    assert lines[:5] == [
        "home:octo-org/demo#7 Add retry to the sync job",
        "open · by octocat · opened 9d ago · retry-sync -> main",
        "labels: bug, sync · review requested: reviewer1 · review: changes_requested",
        "CI: failure · threads: 1 unresolved of 2",
        "https://forge.example.test/octo-org/demo/pulls/7",
    ]
    assert lines[5] == "diff:"
    assert lines[6] == "diff --git a/sync.py b/sync.py"


def test_forge_pr_include_selects_the_live_parts(adapter):
    adapter.forges["home"].prs[0].ci_state = "pending"
    text = tools().forge_pr("home:octo-org/demo", 7, include=["checks"])
    assert "CI: failure · threads: 1 unresolved" in text
    assert "unresolved of" not in text and "diff:" not in text


def test_forge_pr_rejects_an_unknown_include(adapter):
    with pytest.raises(ValueError, match="comments"):
        tools().forge_pr("home:octo-org/demo", 7, include=["comments"])


def test_forge_pr_fetches_a_pull_request_the_cache_lacks(adapter):
    assert tools().forge_pr("home:octo-org/demo", 9, include=[]).splitlines()[1].startswith("merged · by octocat")


def test_forge_pr_truncates_the_diff(adapter, monkeypatch):
    monkeypatch.setattr(adapter.forges["home"].provider, "diff", lambda repo, number: "x" * 30000)
    text = tools().forge_pr("home:octo-org/demo", 7, include=["diff"])
    assert "x" * 20000 in text and "x" * 20001 not in text
    assert text.endswith("[diff truncated at 20000 of 30000 characters]")


def test_tool_output_never_contains_the_token(adapter, monkeypatch):
    from tsugite.secrets.registry import get_registry

    get_registry().register("forgejo-token", TOKEN)
    monkeypatch.setattr(adapter.forges["home"].provider, "diff", lambda repo, number: f"+ token = {TOKEN}\n")
    adapter.forges["gh"].prs = [make_pr(forge="gh", number=12, title=f"leak {TOKEN}")]
    adapter.forges["gh"].error = f"gh: echoed {TOKEN}"
    outputs = [tools().forge_prs(), tools().forge_pr("home:octo-org/demo", 7, include=["diff"]), tools().forge_repos()]
    assert all(TOKEN not in output for output in outputs)
    assert "***" in outputs[1]


def test_forge_repos_lists_each_forge(adapter):
    adapter.forges["gh"].error = "gh: boom"
    assert tools().forge_repos().splitlines() == [
        "home (forgejo), refreshed 2m ago: octo-org/demo, octo-org/tools",
        "gh (github), refreshed 2m ago: octo-org/demo",
        "  last refresh failed: gh: boom",
    ]


def test_forge_category_is_optional_outside_the_daemon():
    from tsugite.tools import _expand_single_spec

    assert _expand_single_spec("@forge") == []


def test_tools_module_imports_without_daemon_or_starlette():
    probe = (
        "import sys, tsugite_forge.tools as t;"
        "assert 'tsugite_daemon' not in sys.modules, 'tools.py pulled in tsugite_daemon';"
        "assert 'starlette' not in sys.modules, 'tools.py pulled in starlette';"
        "t.set_forge_adapter(None)"
    )
    subprocess.run([sys.executable, "-c", probe], check=True)
