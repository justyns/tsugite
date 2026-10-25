"""Dict acceptance criteria: predicate timeouts, per-model verdicts, findings reviews and cheapest-first tiers."""

import json
import subprocess

import pytest
from tsugite_daemon import job_reviews as job_reviews_module
from tsugite_daemon.commands import parse_acceptance_criteria
from tsugite_daemon.job_criteria import ac_text, normalize_criteria
from tsugite_daemon.job_predicates import _evaluate_predicate, _parse_ac_predicate, partition_acs
from tsugite_daemon.job_reviews import brief_findings, format_finding, parse_findings, review_target
from tsugite_daemon.job_store import Job, JobState
from tsugite_daemon.session_store import Session, SessionStatus

from .test_jobs_orchestrator import _make_git_repo

SKIPPED = "skipped: an earlier check failed"


def _seed(store, orchestrator, criteria, **fields):
    job = store.add(Job(id="", parent_session_id="parent-1", prompt="do it", acceptance_criteria=criteria, **fields))
    orchestrator.register_worker(job.id, "worker-1", timeout_minutes=30)
    return store.get(job.id)


def _session(job, session_id, verifier=False):
    metadata = {"job_id": job.id}
    if verifier:
        metadata["verifier_for"] = "worker-1"
    return Session(id=session_id, source="spawned", status=SessionStatus.COMPLETED.value, metadata=metadata)


async def _worker_done(orchestrator, job, summary="## Summary\ndone"):
    await orchestrator.on_session_complete(_session(job, "worker-1"), summary)


async def _report(orchestrator, store, job_id, started, output):
    await orchestrator.on_session_complete(_session(store.get(job_id), started.id, verifier=True), output)


def _verdict(passed, reason="ok"):
    return json.dumps({"ac_results": [{"ac_text": "x", "pass": passed, "reason": reason}], "overall_pass": passed})


def _findings(*findings):
    return json.dumps({"findings": list(findings)})


def _finding(severity, body="a problem", kind="fix", where="app.py:3"):
    return {"severity": severity, "body": body, "kind": kind, "where": where}


def _sessions(runner, agent_file):
    return [s for s in runner.started if s.agent_file == agent_file]


def _retry_prompts(runner):
    return [s.prompt for s in runner.started if (s.metadata or {}).get("loop_attempt")]


def test_normalize_keeps_strings_and_validated_dicts():
    items = ["cmd:ls", "  ", {"cmd": " make ", "timeout": 600}, {"text": "legacy", "kind": "prose"}]
    assert normalize_criteria(items) == ["cmd:ls", {"cmd": "make", "timeout": 600}, "legacy"]


@pytest.mark.parametrize(
    "bad",
    [
        {"cmd": "ls", "bogus": 1},
        {"cmd": "ls", "models": ["m"]},
        {"bogus": 1},
        {"cmd": "ls", "timeout": 0},
        {"cmd": "ls", "timeout": 1801},
        {"cmd": "ls", "timeout": "10"},
        {"exit_code": "ls", "code": "1"},
        {"check": "x", "models": []},
        {"prompt": "p", "block_on": "huge"},
        {"models": ["m"]},
        {"cmd": ""},
    ],
)
def test_normalize_rejects_unknown_or_invalid_keys(bad):
    with pytest.raises(ValueError):
        normalize_criteria([bad])


def test_duplicate_models_collapse():
    assert normalize_criteria([{"check": "x", "models": ["a", " a", "b"]}]) == [{"check": "x", "models": ["a", "b"]}]


def test_ac_text_renders_each_kind():
    assert ac_text("plain") == "plain"
    assert ac_text({"cmd": "ls", "timeout": 5}) == "cmd:ls"
    assert ac_text({"exit_code": "grep x f", "code": 1}) == "exit_code:grep x f:1"
    assert ac_text({"file_exists": "a.txt"}) == "file_exists:a.txt"
    assert ac_text({"check": "it works", "models": ["m"]}) == "it works"
    assert ac_text({"prompt": "find bugs"}) == "review: find bugs"
    assert ac_text({"agent": "my_reviewer"}) == "review: my_reviewer"


def test_dict_predicates_parse_with_timeout():
    assert _parse_ac_predicate({"cmd": "ls", "timeout": 600}) == {"kind": "cmd", "cmd": "ls", "timeout": 600}
    assert _parse_ac_predicate({"exit_code": "grep x f", "code": 1}) == {
        "kind": "exit_code",
        "cmd": "grep x f",
        "expected": 1,
    }
    assert _parse_ac_predicate({"file_exists": "a"}) == {"kind": "file_exists", "path": "a"}
    assert _parse_ac_predicate({"check": "claim"}) is None


def test_partition_acs_sorts_dicts_into_tiers():
    predicates, verdicts, reviews = partition_acs(
        [{"cmd": "ls"}, "claim", {"check": "other", "models": ["m1"]}, {"prompt": "review it"}]
    )
    assert [p["ac_index"] for p in predicates] == [0]
    assert verdicts == [{"ac_index": 1, "ac_text": "claim"}, {"ac_index": 2, "ac_text": "other", "models": ["m1"]}]
    assert [r["ac_index"] for r in reviews] == [3]


@pytest.mark.parametrize(
    "predicate,caller_timeout,expected",
    [
        ({"timeout": 600}, None, 600),
        ({}, None, 30),
        ({"timeout": 600}, 45, 45),
        ({"timeout": 10}, 45, 10),
        ({}, 45, 45),
    ],
)
def test_predicate_timeout_override(tmp_path, monkeypatch, predicate, caller_timeout, expected):
    seen = {}

    def fake_run(argv, **kwargs):
        seen.update(kwargs)
        return subprocess.CompletedProcess(argv, 0, b"", b"")

    monkeypatch.setattr("tsugite_daemon.job_predicates.subprocess.run", fake_run)
    timeout = {} if caller_timeout is None else {"timeout": caller_timeout}
    result = _evaluate_predicate(
        {"kind": "cmd", "cmd": "true", **predicate},
        cwd=str(tmp_path),
        ac_index=0,
        ac_text="cmd:true",
        attempt=1,
        **timeout,
    )
    assert result["pass"] is True
    assert seen["timeout"] == expected


@pytest.mark.asyncio
async def test_create_job_rejects_unknown_criterion_keys_before_creating_anything(store, runner, orchestrator):
    with pytest.raises(ValueError, match="unknown key"):
        await orchestrator.create_and_start_job(
            parent_session_id="parent-1", prompt="x", acceptance_criteria=[{"cmd": "ls", "retries": 3}]
        )
    assert store.list_all() == [] and runner.started == []


@pytest.mark.asyncio
async def test_create_job_stores_dict_criteria_and_shows_their_text(store, runner, orchestrator):
    criteria = [{"cmd": "true", "timeout": 5}, "plain claim"]
    job, _ = await orchestrator.create_and_start_job(
        parent_session_id="parent-1", prompt="x", acceptance_criteria=criteria
    )
    assert store.get(job.id).acceptance_criteria == criteria
    assert job.to_payload()["ac_texts"] == ["cmd:true", "plain claim"]
    assert "1. cmd:true" in runner.started[0].prompt and "2. plain claim" in runner.started[0].prompt


def test_parse_acceptance_criteria_accepts_json_dicts():
    raw = json.dumps([{"cmd": "ls", "timeout": 9}, "claim"])
    assert parse_acceptance_criteria(raw) == [{"cmd": "ls", "timeout": 9}, "claim"]
    assert parse_acceptance_criteria("a|b") == ["a", "b"]


@pytest.mark.asyncio
async def test_dict_predicate_runs_in_the_job_flow(store, runner, orchestrator, tmp_path):
    (tmp_path / "f.txt").write_text("x")
    job = _seed(
        store,
        orchestrator,
        [{"cmd": "test -f f.txt", "timeout": 60}, {"file_exists": "f.txt"}, {"exit_code": "false", "code": 1}],
        worktree_path=str(tmp_path),
    )
    await _worker_done(orchestrator, job)
    final = store.get(job.id)
    assert final.state == JobState.DONE.value
    assert [r["ac_text"] for r in final.ac_results] == ["cmd:test -f f.txt", "file_exists:f.txt", "exit_code:false:1"]
    assert runner.started == []


@pytest.mark.asyncio
async def test_plain_strings_use_one_verifier_on_the_verifier_model(store, runner, orchestrator):
    job = _seed(store, orchestrator, ["claim one", "claim two"], verifier_model="vm")
    await _worker_done(orchestrator, job)
    [verifier] = _sessions(runner, "job_verifier")
    assert verifier.model == "vm"
    assert "claim one" in verifier.prompt and "claim two" in verifier.prompt
    assert store.get(job.id).ac_results is None


@pytest.mark.asyncio
async def test_verdict_with_models_runs_one_verifier_per_model_and_needs_all_to_pass(store, runner, orchestrator):
    job = _seed(store, orchestrator, [{"check": "the report is accurate", "models": ["m1", "m2"]}])
    await _worker_done(orchestrator, job)
    sessions = _sessions(runner, "job_verifier")
    assert [s.model for s in sessions] == ["m1", "m2"]

    await _report(orchestrator, store, job.id, sessions[0], _verdict(True, "read Output: fine"))
    assert store.get(job.id).state == JobState.VERIFYING.value

    await _report(orchestrator, store, job.id, sessions[1], _verdict(False, "numbers are off"))
    refreshed = store.get(job.id)
    [row] = refreshed.ac_results
    assert row["pass"] is False
    assert row["models"] == ["m1", "m2"]
    assert row["verdicts"] == [
        {"model": "m1", "pass": True, "reason": "read Output: fine"},
        {"model": "m2", "pass": False, "reason": "numbers are off"},
    ]
    assert refreshed.state == JobState.RUNNING.value
    assert refreshed.verify_attempts == 1
    [brief] = _retry_prompts(runner)
    assert "the report is accurate" in brief and "m2: numbers are off" in brief


@pytest.mark.asyncio
async def test_verdict_with_models_passes_when_every_model_passes(store, runner, orchestrator):
    job = _seed(store, orchestrator, [{"check": "claim", "models": ["m1", "m2"]}])
    await _worker_done(orchestrator, job)
    for session in _sessions(runner, "job_verifier"):
        await _report(orchestrator, store, job.id, session, _verdict(True))
    final = store.get(job.id)
    assert final.state == JobState.DONE.value
    assert final.result["ac_results"][0]["models"] == ["m1", "m2"]


@pytest.mark.asyncio
async def test_strings_and_model_checks_share_sessions_by_model(store, runner, orchestrator):
    job = _seed(store, orchestrator, ["plain", {"check": "pinned", "models": ["m1"]}], verifier_model="vm")
    await _worker_done(orchestrator, job)
    default, pinned = _sessions(runner, "job_verifier")
    assert default.model == "vm" and "plain" in default.prompt and "pinned" not in default.prompt
    assert pinned.model == "m1" and "pinned" in pinned.prompt and "plain" not in pinned.prompt


def test_parse_findings_is_strict():
    ok = _findings(_finding("major"), {"severity": "note", "body": "fyi"})
    assert [f["severity"] for f in parse_findings(f"prose {ok} tail")] == ["major", "note"]
    assert parse_findings(_findings()) == []
    assert parse_findings(_findings({"severity": "huge", "body": "x"})) is None
    assert parse_findings(_findings({"severity": "major", "body": ""})) is None
    assert parse_findings(_findings({"severity": "major", "body": "x", "kind": "maybe"})) is None
    assert parse_findings('{"findings": "none"}') is None
    assert parse_findings("no json here") is None


@pytest.mark.asyncio
async def test_review_below_block_on_passes(store, runner, orchestrator):
    job = _seed(store, orchestrator, [{"prompt": "find bugs", "models": ["m1"], "block_on": "major"}])
    await _worker_done(orchestrator, job)
    [reviewer] = _sessions(runner, "job_reviewer")
    assert reviewer.model == "m1" and "find bugs" in reviewer.prompt
    await _report(orchestrator, store, job.id, reviewer, _findings(_finding("minor"), _finding("note")))
    final = store.get(job.id)
    assert final.state == JobState.DONE.value
    [row] = final.ac_results
    assert row["pass"] is True and row["models"] == ["m1"] and len(row["findings"]) == 2


@pytest.mark.asyncio
async def test_review_at_block_on_fails_and_the_brief_lists_the_blocking_findings(store, runner, orchestrator):
    job = _seed(store, orchestrator, [{"prompt": "find bugs", "models": ["m1", "m2"], "block_on": "major"}])
    await _worker_done(orchestrator, job)
    first, second = _sessions(runner, "job_reviewer")
    await _report(orchestrator, store, job.id, first, _findings(_finding("minor", "style nit")))
    await _report(orchestrator, store, job.id, second, _findings(_finding("blocker", "SQL injection", where="db.py:9")))
    refreshed = store.get(job.id)
    [row] = refreshed.ac_results
    assert row["pass"] is False
    assert [f["model"] for f in row["findings"]] == ["m1", "m2"]
    assert refreshed.state == JobState.RUNNING.value and refreshed.verify_attempts == 1
    [brief] = _retry_prompts(runner)
    assert "[blocker] db.py:9: SQL injection" in brief
    assert "style nit" not in brief


@pytest.mark.asyncio
async def test_advisory_review_never_fails(store, runner, orchestrator):
    job = _seed(store, orchestrator, [{"prompt": "look around", "models": ["m1"]}])
    await _worker_done(orchestrator, job)
    [reviewer] = _sessions(runner, "job_reviewer")
    await _report(orchestrator, store, job.id, reviewer, _findings(_finding("blocker", kind="decision")))
    final = store.get(job.id)
    assert final.state == JobState.DONE.value
    assert final.ac_results[0]["pass"] is True
    assert final.ac_results[0]["findings"][0]["severity"] == "blocker"


@pytest.mark.asyncio
async def test_blocking_decision_finding_parks_the_job(store, runner, orchestrator):
    job = _seed(store, orchestrator, [{"prompt": "review", "models": ["m1"], "block_on": "major"}])
    await _worker_done(orchestrator, job)
    [reviewer] = _sessions(runner, "job_reviewer")
    await _report(
        orchestrator, store, job.id, reviewer, _findings(_finding("major", "Use sqlite or postgres?", kind="decision"))
    )
    parked = store.get(job.id)
    assert parked.state == JobState.STUCK.value
    assert parked.verify_attempts == 0
    assert "needs a decision" in parked.error and "Use sqlite or postgres?" in parked.error
    assert parked.result["ac_results"][0]["findings"][0]["kind"] == "decision"
    assert _retry_prompts(runner) == []


@pytest.mark.asyncio
async def test_unparseable_reviewer_output_is_rerun_once_then_errors(store, runner, orchestrator):
    job = _seed(store, orchestrator, [{"prompt": "review", "models": ["m1"], "block_on": "major"}])
    await _worker_done(orchestrator, job)
    [first] = _sessions(runner, "job_reviewer")
    await _report(orchestrator, store, job.id, first, "I looked and it is fine")
    rerun = _sessions(runner, "job_reviewer")
    assert len(rerun) == 2 and store.get(job.id).state == JobState.VERIFYING.value

    await _report(orchestrator, store, job.id, rerun[1], "still prose")
    errored = store.get(job.id)
    assert errored.state == JobState.ERRORED.value
    assert "reviewer produced no parseable findings" in errored.error
    [row] = errored.ac_results
    assert row["pass"] is False and "unparseable" in row["reason"]
    assert errored.attempts[-1]["verifier_pass"] is False


@pytest.mark.asyncio
async def test_custom_agent_template_runs_the_review(store, runner, orchestrator):
    job = _seed(store, orchestrator, [{"agent": "my_reviewer", "prompt": "focus on auth"}])
    await _worker_done(orchestrator, job)
    [reviewer] = _sessions(runner, "my_reviewer")
    assert "focus on auth" in reviewer.prompt


@pytest.mark.asyncio
async def test_repo_review_gets_the_diff_inline_and_the_worktree_path(store, runner, orchestrator, tmp_path):
    repo = tmp_path / "repo"
    _make_git_repo(repo)
    base = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True).stdout.strip()
    (repo / "README.md").write_text("hi\nadded line\n")
    (repo / "new.py").write_text("print(1)\n")
    job = _seed(store, orchestrator, [{"prompt": "review", "models": ["m1"]}], worktree_path=str(repo), base_sha=base)
    await _worker_done(orchestrator, job)
    [reviewer] = _sessions(runner, "job_reviewer")
    assert "+added line" in reviewer.prompt
    assert "new.py" in reviewer.prompt
    assert str(repo) in reviewer.prompt
    assert reviewer.workspace_override == str(repo)


@pytest.mark.asyncio
async def test_non_repo_review_gets_the_result_and_the_files_to_read(store, runner, orchestrator, tmp_path):
    (tmp_path / "report.md").write_text("the report body")
    job = _seed(store, orchestrator, [{"prompt": "review", "models": ["m1"]}], workspace_path=str(tmp_path))
    await _worker_done(orchestrator, job, "## Summary\nwrote it\n\n## Artifacts\n- Files changed: report.md, gone.md\n")
    [reviewer] = _sessions(runner, "job_reviewer")
    assert "wrote it" in reviewer.prompt and "report.md" in reviewer.prompt
    assert "the report body" not in reviewer.prompt
    assert reviewer.workspace_override == str(tmp_path)


@pytest.mark.asyncio
async def test_failing_predicate_spawns_no_verifier_or_reviewer(store, runner, orchestrator, tmp_path):
    job = _seed(
        store,
        orchestrator,
        [{"cmd": "false"}, "a claim", {"check": "pinned", "models": ["m1"]}, {"prompt": "review", "models": ["m1"]}],
        worktree_path=str(tmp_path),
    )
    await _worker_done(orchestrator, job)
    assert _sessions(runner, "job_verifier") == [] and _sessions(runner, "job_reviewer") == []
    refreshed = store.get(job.id)
    rows = {r["ac_index"]: r for r in refreshed.ac_results}
    assert rows[0]["pass"] is False and rows[0]["reason"] != SKIPPED
    assert all(rows[i]["reason"] == SKIPPED and rows[i]["pass"] is False for i in (1, 2, 3))
    assert refreshed.verify_attempts == 1


@pytest.mark.asyncio
async def test_failing_verdict_spawns_no_reviewer(store, runner, orchestrator):
    job = _seed(store, orchestrator, ["a claim", {"prompt": "review", "models": ["m1"], "block_on": "major"}])
    await _worker_done(orchestrator, job)
    [verifier] = _sessions(runner, "job_verifier")
    await _report(orchestrator, store, job.id, verifier, _verdict(False, "not true"))
    assert _sessions(runner, "job_reviewer") == []
    rows = {r["ac_index"]: r for r in store.get(job.id).ac_results}
    assert rows[0]["reason"] == "not true"
    assert rows[1]["reason"] == SKIPPED


@pytest.mark.asyncio
async def test_passing_verdicts_lead_to_the_review_tier(store, runner, orchestrator, tmp_path):
    job = _seed(
        store,
        orchestrator,
        ["cmd:true", "a claim", {"prompt": "review", "models": ["m1"], "block_on": "major"}],
        worktree_path=str(tmp_path),
    )
    await _worker_done(orchestrator, job)
    assert _sessions(runner, "job_reviewer") == []
    [verifier] = _sessions(runner, "job_verifier")
    await _report(orchestrator, store, job.id, verifier, _verdict(True))
    [reviewer] = _sessions(runner, "job_reviewer")
    assert store.get(job.id).state == JobState.VERIFYING.value
    await _report(orchestrator, store, job.id, reviewer, _findings())
    final = store.get(job.id)
    assert final.state == JobState.DONE.value
    assert [r["ac_index"] for r in final.result["ac_results"]] == [0, 1, 2]


def _head_sha(repo):
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True).stdout.strip()


def _spy_on_git(monkeypatch, subcommand):
    """Record the sizes of reads from `git <subcommand>` stdout; return (reads, procs)."""
    reads: list[int] = []
    procs = []
    real_popen = subprocess.Popen

    class Spy:
        def __init__(self, real):
            self._real = real

        def read(self, n=-1):
            data = self._real.read(n)
            reads.append(len(data))
            return data

        def close(self):
            self._real.close()

        @property
        def closed(self):
            return self._real.closed

    def spy_popen(*args, **kwargs):
        proc = real_popen(*args, **kwargs)
        if subcommand in args[0]:
            procs.append(proc)
            proc.stdout = Spy(proc.stdout)
        return proc

    monkeypatch.setattr("tsugite_daemon.job_reviews.subprocess.Popen", spy_popen)
    return reads, procs


def test_review_target_streams_a_huge_diff_and_stops_at_the_cap(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    _make_git_repo(repo)
    base = _head_sha(repo)
    (repo / "huge.txt").write_text("line of text\n" * 400_000)
    subprocess.run(["git", "add", "huge.txt"], cwd=repo, check=True)
    reads, procs = _spy_on_git(monkeypatch, "diff")
    target = review_target(Job(id="j", parent_session_id="p", prompt="x", base_sha=base), "", str(repo))
    assert sum(reads) <= 60_001 and "[truncated" in target
    assert procs[0].poll() is not None


def test_parse_findings_keeps_long_fields_and_formatting_clips_them():
    [finding] = parse_findings(_findings(_finding("major", "b" * 5_000, where="w" * 500)))
    assert len(finding["body"]) == 5_000 and len(finding["where"]) == 500
    assert len(format_finding(finding)) < 2_300


@pytest.mark.asyncio
async def test_ac_results_keep_the_full_finding_while_the_brief_clips_it(store, runner, orchestrator):
    job = _seed(store, orchestrator, [{"prompt": "find bugs", "models": ["m1"], "block_on": "minor"}])
    await _worker_done(orchestrator, job)
    [reviewer] = _sessions(runner, "job_reviewer")
    await _report(orchestrator, store, job.id, reviewer, _findings(_finding("major", "b" * 5_000)))
    [brief] = _retry_prompts(runner)
    assert "b" * 5_000 not in brief and "b" * 2_000 in brief
    assert store.get(job.id).ac_results[0]["findings"][0]["body"] == "b" * 5_000


def test_brief_findings_keeps_the_top_ten_and_counts_the_rest():
    findings = [_finding("minor", f"m{i}") for i in range(12)] + [_finding("blocker", "worst")]
    lines = brief_findings(findings, indent="  - ")
    assert len(lines) == 11
    assert lines[0] == "  - [blocker] app.py:3: worst"
    assert lines[-1] == "  - +3 more (full list in ac_results)"
    assert len(brief_findings(findings[:10])) == 10


def test_review_target_streams_the_untracked_listing_and_caps_it(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    _make_git_repo(repo)
    base = _head_sha(repo)
    for i in range(2_000):
        (repo / f"untracked_{i:05d}_{'n' * 40}.txt").write_text("")
    reads, _ = _spy_on_git(monkeypatch, "ls-files")
    target = review_target(Job(id="j", parent_session_id="p", prompt="x", base_sha=base), "", str(repo))
    listing = target.split("Untracked files (read them in the working directory): ")[1]
    assert reads and sum(reads) <= 4_001
    assert len(listing) < 4_200 and "[truncated" in listing


@pytest.mark.asyncio
async def test_completion_without_a_round_is_ignored(store, runner, orchestrator):
    job = _seed(store, orchestrator, ["a claim"])
    await _worker_done(orchestrator, job)
    [verifier] = _sessions(runner, "job_verifier")
    orchestrator._update_latest_attempt(job.id, round=None)

    await orchestrator._handle_verifier_complete(store.get(job.id), verifier.id, _verdict(True))
    assert store.get(job.id).state == JobState.VERIFYING.value


@pytest.mark.asyncio
async def test_second_round_session_completes_the_round_and_outsiders_are_ignored(store, runner, orchestrator):
    job = _seed(store, orchestrator, [{"check": "claim", "models": ["m1", "m2"]}])
    await _worker_done(orchestrator, job)
    first, second = _sessions(runner, "job_verifier")
    assert store.get(job.id).verifier_session_id == first.id

    outsider = Session(
        id="stranger-session",
        source="spawned",
        status=SessionStatus.FAILED.value,
        metadata={"job_id": job.id, "verifier_for": "worker-1"},
        error="boom",
    )
    await orchestrator.on_session_complete(outsider, _verdict(True))
    outsider.status = SessionStatus.COMPLETED.value
    await orchestrator.on_session_complete(outsider, _verdict(True))
    assert store.get(job.id).state == JobState.VERIFYING.value

    await _report(orchestrator, store, job.id, first, _verdict(True))
    await _report(orchestrator, store, job.id, second, _verdict(True))
    assert store.get(job.id).state == JobState.DONE.value


@pytest.mark.asyncio
async def test_hundreds_of_long_findings_give_a_bounded_brief(store, runner, orchestrator):
    job = _seed(store, orchestrator, [{"prompt": "find bugs", "models": ["m1"], "block_on": "minor"}])
    await _worker_done(orchestrator, job)
    [reviewer] = _sessions(runner, "job_reviewer")
    many = [_finding("minor", f"issue {i} " + "z" * 5_000) for i in range(300)] + [_finding("blocker", "the worst")]
    await _report(orchestrator, store, job.id, reviewer, _findings(*many))
    [brief] = _retry_prompts(runner)
    assert len(brief) < 25_000
    assert "the worst" in brief
    assert "more (full list in ac_results)" in brief
    assert len(store.get(job.id).ac_results[0]["findings"]) == 301


@pytest.mark.asyncio
async def test_hundreds_of_decision_findings_give_a_bounded_job_error(store, runner, orchestrator):
    job = _seed(store, orchestrator, [{"prompt": "review", "models": ["m1"], "block_on": "minor"}])
    await _worker_done(orchestrator, job)
    [reviewer] = _sessions(runner, "job_reviewer")
    many = [_finding("major", "pick one " + "q" * 5_000, kind="decision") for _ in range(300)]
    await _report(orchestrator, store, job.id, reviewer, _findings(*many))
    parked = store.get(job.id)
    assert parked.state == JobState.STUCK.value and len(parked.error) < 25_000


@pytest.mark.asyncio
async def test_duplicate_verdict_completion_does_not_touch_the_review_round(store, runner, orchestrator, tmp_path):
    job = _seed(
        store,
        orchestrator,
        ["a claim", {"prompt": "review", "models": ["m1"], "block_on": "major"}],
        worktree_path=str(tmp_path),
    )
    await _worker_done(orchestrator, job)
    [verifier] = _sessions(runner, "job_verifier")
    await _report(orchestrator, store, job.id, verifier, _verdict(True))
    [reviewer] = _sessions(runner, "job_reviewer")
    before = store.get(job.id).attempts[-1]["round"]

    await _report(orchestrator, store, job.id, verifier, _verdict(False, "late duplicate"))

    after = store.get(job.id)
    assert after.state == JobState.VERIFYING.value
    assert after.attempts[-1]["round"] == before
    assert list(before["sessions"]) == [reviewer.id]
    await _report(orchestrator, store, job.id, reviewer, _findings())
    assert store.get(job.id).state == JobState.DONE.value


@pytest.mark.asyncio
async def test_rerun_spawns_nothing_when_the_job_is_cancelled_while_the_target_is_built(
    store, runner, orchestrator, monkeypatch
):
    job = _seed(store, orchestrator, [{"prompt": "review", "models": ["m1"], "block_on": "major"}])
    await _worker_done(orchestrator, job)
    [first] = _sessions(runner, "job_reviewer")
    real = job_reviews_module.review_target

    def cancelling(*args):
        store.update_state(job.id, JobState.CANCELLED.value)
        return real(*args)

    monkeypatch.setattr("tsugite_daemon.jobs_orchestrator.review_target", cancelling)
    await _report(orchestrator, store, job.id, first, "prose, no findings")
    assert len(_sessions(runner, "job_reviewer")) == 1
    assert store.get(job.id).state == JobState.CANCELLED.value


def test_hung_git_helper_holding_stdout_is_killed_with_the_group(tmp_path, monkeypatch):
    import time

    repo = tmp_path / "repo"
    _make_git_repo(repo)
    real_popen = subprocess.Popen
    monkeypatch.setattr(job_reviews_module, "_GIT_TIMEOUT_SECONDS", 0.5)
    hang = ["sh", "-c", "echo partial; sleep 60 & exec sleep 60"]
    seen = []

    def hanging_popen(args, **kwargs):
        seen.append(list(args))
        return real_popen(hang if "diff" in args else args, **kwargs)

    monkeypatch.setattr("tsugite_daemon.job_reviews.subprocess.Popen", hanging_popen)
    start = time.monotonic()
    target = review_target(Job(id="j", parent_session_id="p", prompt="x", base_sha="HEAD"), "", str(repo))
    assert time.monotonic() - start < 10
    assert "partial" in target and "was killed after" in target
    assert "--no-ext-diff" in seen[0] and "--no-textconv" in seen[0]


def test_git_that_finishes_is_not_reported_as_killed(tmp_path):
    repo = tmp_path / "repo"
    _make_git_repo(repo)
    text, note = job_reviews_module._git_capped(["ls-files"], repo, 10_000)
    assert note == ""
