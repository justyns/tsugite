"""The two forge surfaces: their wiring, their served assets, and the pure filter helpers."""

import json
import subprocess
from pathlib import Path

import pytest
from forge_helpers import forge_http, make_adapter
from tsugite_forge.providers import kinds

UI = Path(__file__).parent.parent / "tsugite_forge" / "ui"


@pytest.fixture
def server(tmp_path):
    from tsugite_forge.tools import PRS_SURFACE, SESSION_SURFACE

    client, _tokens, http_server = forge_http(make_adapter(lambda: 0), tmp_path, [PRS_SURFACE, SESSION_SURFACE])
    client.surfaces = {surface["kind"]: surface for surface in http_server.plugin_ui_surfaces}
    return client


def test_surfaces_are_wired_as_a_nav_page_and_a_session_tab(server):
    prs, session = server.surfaces["plugin/forge/prs"], server.surfaces["plugin/forge/session-prs"]
    assert (prs["entry"], prs["nav"]) == ("/api/plugins/forge/ui/prs.html", True)
    assert (session["entry"], session["nav"], session["mode"], session["params"]) == (
        "/api/plugins/forge/ui/session.html",
        False,
        "workspace",
        ["session"],
    )


@pytest.mark.parametrize("asset", ["prs.html", "session.html", "filters.js", "surface.js", "forge.css"])
def test_page_and_its_module_are_served(server, asset):
    resp = server.get(f"/api/plugins/forge/ui/{asset}")
    assert resp.status_code == 200
    assert resp.text == (UI / asset).read_text()


@pytest.mark.parametrize("kind", sorted(kinds))
def test_every_forge_kind_has_a_served_icon(server, kind):
    resp = server.get(f"/api/plugins/forge/ui/icons/{kind}.svg")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/svg+xml"
    assert f".forge-icon[data-kind='{kind}'] {{\n  mask-image: url(icons/{kind}.svg);" in (UI / "forge.css").read_text()


PRS = [
    {
        "forge": "home",
        "repo": "octo-org/demo",
        "number": 7,
        "author": "octocat",
        "requested_reviewers": ["reviewer1"],
        "ci_state": "failure",
        "unresolved_threads": 1,
        "labels": ["needs-fix"],
    },
    {
        "forge": "home",
        "repo": "octo-org/demo",
        "number": 8,
        "author": "reviewer1",
        "requested_reviewers": [],
        "ci_state": "none",
        "unresolved_threads": 0,
        "labels": [],
    },
    {
        "forge": "gh",
        "repo": "octo-org/demo",
        "number": 12,
        "author": "octocat",
        "requested_reviewers": ["octocat"],
        "ci_state": "success",
        "unresolved_threads": None,
        "labels": ["ready-to-merge"],
    },
]
VIEWERS = {"home": "reviewer1", "gh": "octocat"}


def _node(names: str, script: str):
    """Run `script` under node with `names` imported from filters.js, and read the JSON it prints."""
    source = f"import {{ {names} }} from {json.dumps((UI / 'filters.js').as_uri())};{script}"
    out = subprocess.run(["node", "--input-type=module", "-e", source], capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


@pytest.mark.parametrize(
    "filters, numbers",
    [
        ({}, [7, 8, 12]),
        ({"mine": True}, [8, 12]),
        ({"reviewRequested": True}, [7, 12]),
        ({"ciFailing": True}, [7]),
        ({"unresolved": True}, [7]),
        ({"label": "ready-to-merge"}, [12]),
        ({"mine": True, "label": "ready-to-merge"}, [12]),
    ],
)
def test_filter_prs(filters, numbers):
    script = (
        f"const prs = filterPrs({json.dumps(PRS)}, {json.dumps(filters)}, {json.dumps(VIEWERS)});"
        "console.log(JSON.stringify(prs.map((pr) => pr.number)));"
    )
    assert _node("filterPrs", script) == numbers


def test_group_by_repo_and_labels():
    script = (
        f"console.log(JSON.stringify({{groups: groupByRepo({json.dumps(PRS)}).map(([key, group]) => [key, group.length]),"
        f" labels: labelsOf({json.dumps(PRS)})}}));"
    )
    assert _node("groupByRepo, labelsOf", script) == {
        "groups": [["gh:octo-org/demo", 1], ["home:octo-org/demo", 2]],
        "labels": ["needs-fix", "ready-to-merge"],
    }


REFS = [
    {"forge": "home", "repo": "octo-org/demo", "pr": 8, "source": "pinned", "hidden": False},
    {"forge": "home", "repo": "octo-org/demo", "pr": None, "source": "touched", "hidden": False},
    {"forge": "gh", "repo": "octo-org/site", "pr": 12, "source": "mentioned", "hidden": True},
    {"forge": "home", "repo": "octo-org/demo", "pr": 9, "source": "mentioned", "hidden": False},
]


def test_partition_refs():
    script = (
        f"const parts = partitionRefs({json.dumps(REFS)});"
        "console.log(JSON.stringify(Object.fromEntries(Object.entries(parts).map(([k, v]) => [k, v.map((r) => r.pr)]))));"
    )
    assert _node("partitionRefs", script) == {"main": [8, None], "mentioned": [9], "hidden": [12]}
