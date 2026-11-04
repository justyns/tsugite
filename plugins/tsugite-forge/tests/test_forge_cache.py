"""Caches for live pull request detail and Forgejo ETags."""

import httpx
import pytest
from forge_helpers import Clock, FakeForgejo, FakeGh, forgejo_config, make_adapter, make_pr


class CountingProvider:
    def __init__(self):
        self.calls = []

    def checks(self, repo, number):
        self.calls.append("checks")
        return "failure"

    def review_threads(self, repo, number):
        from tsugite_forge.providers import ReviewThreads

        self.calls.append("threads")
        return ReviewThreads(total=2, unresolved=1)

    def diff(self, repo, number):
        self.calls.append("diff")
        return "diff --git a/x b/x\n"


@pytest.fixture
def adapter(clock):
    adapter = make_adapter(clock, forges=["home"])
    home = adapter.forges["home"]
    home.repos, home.prs = ["octo-org/demo"], [make_pr()]
    home.provider = CountingProvider()
    return adapter


def test_checks_and_threads_are_reused_for_the_same_head_but_the_diff_is_not(adapter):
    first = adapter.pr_detail("home", "octo-org/demo", 7)
    assert adapter.pr_detail("home", "octo-org/demo", 7) == first
    assert adapter.forges["home"].provider.calls == ["checks", "threads", "diff", "diff"]


def test_a_new_head_sha_reads_everything_again(adapter):
    adapter.pr_detail("home", "octo-org/demo", 7)
    adapter.forges["home"].prs = [make_pr(head_sha="a" * 40)]
    adapter.pr_detail("home", "octo-org/demo", 7)
    assert adapter.forges["home"].provider.calls == ["checks", "threads", "diff"] * 2


def test_checks_and_threads_expire(adapter, clock):
    from tsugite_forge.adapter import DETAIL_TTL

    adapter.pr_detail("home", "octo-org/demo", 7)
    clock.now += DETAIL_TTL + 1
    adapter.pr_detail("home", "octo-org/demo", 7)
    assert adapter.forges["home"].provider.calls == ["checks", "threads", "diff"] * 2


def test_github_pr_detail_sends_one_graphql_query(tmp_path, monkeypatch):
    from tsugite_forge.tools import forge_pr, set_forge_adapter

    gh = FakeGh(tmp_path, monkeypatch)
    adapter = make_adapter(Clock(), forges=["gh"])
    adapter.forges["gh"].repos = ["octo-org/demo"]
    set_forge_adapter(adapter)
    try:
        assert "CI: failure · threads: 2 unresolved of 3" in forge_pr("gh:octo-org/demo", 12)
    finally:
        set_forge_adapter(None)
    assert [call[1] for call in gh.calls].count("graphql") == 1


def test_the_etag_cache_keeps_only_the_newest_entries(secrets_backend, monkeypatch):
    import tsugite_forge.providers.forgejo as forgejo

    monkeypatch.setattr(forgejo, "MAX_ETAGS", 3)
    routes = {
        f"/api/v1/repos/octo-org/demo/pulls/{n}": (
            lambda request, n=n: httpx.Response(200, json={}, headers={"ETag": f'"{n}"'})
        )
        for n in range(1, 6)
    }
    provider = forgejo.ForgejoProvider(forgejo_config(), transport=FakeForgejo(routes).transport)
    for n in range(1, 6):
        provider._get(f"/repos/octo-org/demo/pulls/{n}")
    assert [key.rsplit("/", 1)[1] for key in provider._etags._items] == ["3", "4", "5"]
