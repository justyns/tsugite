"""Findings reviews: the reviewer's target and prompt, strict parsing of its findings, and the pass rule."""

import os
import signal
import subprocess
import threading
from pathlib import Path
from typing import Optional

from tsugite_daemon.job_criteria import SEVERITIES, clip
from tsugite_daemon.job_store import Job
from tsugite_daemon.job_verdicts import _json_objects

REVIEWER_AGENT = "job_reviewer"

_TARGET_CAP_CHARS = 60_000
_UNTRACKED_CAP_CHARS = 4_000
_GIT_TIMEOUT_SECONDS = 30

_BODY_CAP_CHARS = 2_000
_WHERE_CAP_CHARS = 200
BRIEF_FINDINGS_PER_CRITERION = 10

_FINDING_KINDS = ("fix", "decision")


def parse_findings(raw: str) -> Optional[list[dict]]:
    """The findings of the last JSON object holding a `findings` list, or None when any entry is malformed."""
    found = [obj for obj in _json_objects(raw) if "findings" in obj]
    if not found or not isinstance(found[-1]["findings"], list):
        return None
    findings = []
    for item in found[-1]["findings"]:
        if not isinstance(item, dict) or item.get("severity") not in SEVERITIES:
            return None
        body = item.get("body")
        if not isinstance(body, str) or not body.strip():
            return None
        where = item.get("where")
        kind = item.get("kind", "fix")
        if kind not in _FINDING_KINDS or not (where is None or isinstance(where, str)):
            return None
        finding = {"severity": item["severity"], "body": body.strip(), "kind": kind}
        if where:
            finding["where"] = where
        findings.append(finding)
    return findings


def review_entry(
    entry: dict, review: dict, per_model: list[tuple[str, list[dict]]], attempt: int
) -> tuple[dict, list[dict]]:
    block_on = review.get("block_on")
    findings = [{**f, "model": model} for model, model_findings in per_model for f in model_findings]
    if block_on is None:
        blocking = []
        reason = f"advisory: {len(findings)} finding(s)"
    else:
        blocking = [f for f in findings if SEVERITIES.index(f["severity"]) >= SEVERITIES.index(block_on)]
        reason = (
            f"{len(blocking)} finding(s) at or above {block_on}" if blocking else f"no findings at or above {block_on}"
        )
    row = {
        "ac_index": entry["ac_index"],
        "ac_text": entry["ac_text"],
        "pass": not blocking,
        "reason": reason,
        "attempt": attempt,
        "models": [model for model, _ in per_model],
        "findings": findings,
        "block_on": block_on,
    }
    return row, blocking


def format_finding(finding: dict) -> str:
    where = f" {clip(finding['where'], _WHERE_CAP_CHARS)}:" if finding.get("where") else ""
    return f"[{finding['severity']}]{where} {clip(finding['body'], _BODY_CAP_CHARS)}"


def brief_findings(findings: list[dict], indent: str = "") -> list[str]:
    """Lines for the worker's brief or a job error: the BRIEF_FINDINGS_PER_CRITERION highest-severity findings."""
    ranked = sorted(findings, key=lambda f: -SEVERITIES.index(f["severity"]))
    lines = [indent + format_finding(f) for f in ranked[:BRIEF_FINDINGS_PER_CRITERION]]
    if len(ranked) > BRIEF_FINDINGS_PER_CRITERION:
        lines.append(f"{indent}+{len(ranked) - BRIEF_FINDINGS_PER_CRITERION} more (full list in ac_results)")
    return lines


def build_reviewer_prompt(review: dict, target: str, cwd: Optional[str]) -> str:
    parts = ["Instructions:", review.get("prompt") or "Review according to your instructions."]
    if cwd:
        parts += ["", f"Working directory (read-only; read_file and list_files resolve against it): {cwd}"]
    parts += [
        "",
        "Target:",
        target,
        "",
        'Reply with only a JSON object {"findings": [{"severity": "blocker|major|minor|note", '
        '"body": "...", "where": "optional path:line or free text", "kind": "fix|decision"}]}.',
    ]
    return "\n".join(parts)


def _git_capped(args: list[str], cwd: Path, limit: int) -> tuple[str, str]:
    """Output of `git args`, reading no more than `limit` characters and killing git's whole process
    group after _GIT_TIMEOUT_SECONDS. The second value is a note on a cut or timed-out read, else ""."""
    if args[0] == "diff":
        args = ["diff", "--no-ext-diff", "--no-textconv", *args[1:]]
    proc = subprocess.Popen(
        ["git", "-c", "core.fsmonitor=false", *args],
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        errors="replace",
        start_new_session=True,
    )
    timed_out = threading.Event()

    def on_timeout() -> None:
        timed_out.set()
        os.killpg(proc.pid, signal.SIGKILL)

    timer = threading.Timer(_GIT_TIMEOUT_SECONDS, on_timeout)
    timer.start()
    try:
        text = proc.stdout.read(limit + 1)
    finally:
        timer.cancel()
        timer.join()
        os.killpg(proc.pid, signal.SIGKILL)
        proc.stdout.close()
        proc.wait()
    if len(text) > limit:
        return text[:limit], f"[truncated: git {args[0]} output is longer]"
    if timed_out.is_set():
        return text, f"[git {args[0]} was killed after {_GIT_TIMEOUT_SECONDS}s]"
    return text, ""


def review_target(job: Job, worker_output: str, cwd: Optional[str]) -> str:
    """A repo job's diff from its base commit to the worktree, else the worker's result."""
    if job.base_sha and cwd:
        diff, diff_note = _git_capped(["diff", job.base_sha], Path(cwd), _TARGET_CAP_CHARS)
        untracked, untracked_note = _git_capped(
            ["ls-files", "--others", "--exclude-standard"], Path(cwd), _UNTRACKED_CAP_CHARS
        )
        parts = [f"Diff from {job.base_sha[:12]} to the working tree:", "```diff", diff, "```"]
        if diff_note:
            parts.append(diff_note)
        if untracked.strip():
            listed = ", ".join(untracked.splitlines()) + (f" {untracked_note}" if untracked_note else "")
            parts.append("Untracked files (read them in the working directory): " + listed)
        return "\n".join(parts)
    target = "Worker result:\n" + clip(worker_output.strip() or "(empty)", _TARGET_CAP_CHARS)
    if cwd:
        target += "\n\nRead the files it lists under `Files changed:` in the working directory."
    return target
