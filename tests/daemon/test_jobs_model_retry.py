"""Retry-on-a-different-model + model-escalation ladder + infra classification.

A Job that dies on a usage/rate limit is an infrastructure failure, not a
quality failure: it must be labeled as such and, when the Job carries a model
ladder, escalate to the next model instead of dead-ending on the exhausted
one. A stuck/errored Job must be retryable on a user-chosen model without
inventing a hint."""

import json

import pytest
from tsugite_daemon.job_store import JobState
from tsugite_daemon.session_store import Session, SessionStatus

from .test_jobs_orchestrator import (
    MAX_VERIFY_ATTEMPTS,
    _seed_running_job,
    _verifier_session,
    _worker_session,
)

FAIL_VERDICT = json.dumps({"ac_results": [{"ac_text": "x", "pass": False, "reason": "nope"}], "overall_pass": False})


# ── Part 1: retry on a different model ──


@pytest.mark.asyncio
async def test_retry_with_model_only_no_hint_required(store, runner, orchestrator):
    job = _seed_running_job(store, orchestrator, runner, acceptance_criteria=["x"])
    for i in range(MAX_VERIFY_ATTEMPTS):
        await orchestrator.on_session_complete(_worker_session(store.get(job.id), f"w{i}"), f"w{i}")
        await orchestrator.on_session_complete(_verifier_session(store.get(job.id), f"v{i}"), FAIL_VERDICT)
    assert store.get(job.id).state == JobState.STUCK.value

    await orchestrator.retry_with_hint(job.id, hint="", model="anthropic:claude-opus-4-8")
    refreshed = store.get(job.id)
    assert refreshed.state == JobState.RUNNING.value
    assert refreshed.model == "anthropic:claude-opus-4-8", "chosen model must persist on the Job"
    new_worker = runner.started[-1]
    assert new_worker.model == "anthropic:claude-opus-4-8", "the retry worker must actually run on the chosen model"


@pytest.mark.asyncio
async def test_retry_requires_hint_or_model(store, runner, orchestrator):
    job = _seed_running_job(store, orchestrator, runner, acceptance_criteria=["x"])
    for i in range(MAX_VERIFY_ATTEMPTS):
        await orchestrator.on_session_complete(_worker_session(store.get(job.id), f"w{i}"), f"w{i}")
        await orchestrator.on_session_complete(_verifier_session(store.get(job.id), f"v{i}"), FAIL_VERDICT)

    with pytest.raises(ValueError, match="hint or model"):
        await orchestrator.retry_with_hint(job.id, hint="")


@pytest.mark.asyncio
async def test_retry_with_verifier_model_persists(store, runner, orchestrator):
    job = _seed_running_job(store, orchestrator, runner, acceptance_criteria=["x"])
    for i in range(MAX_VERIFY_ATTEMPTS):
        await orchestrator.on_session_complete(_worker_session(store.get(job.id), f"w{i}"), f"w{i}")
        await orchestrator.on_session_complete(_verifier_session(store.get(job.id), f"v{i}"), FAIL_VERDICT)

    await orchestrator.retry_with_hint(job.id, hint="try again", verifier_model="anthropic:claude-haiku-4-5")
    assert store.get(job.id).verifier_model == "anthropic:claude-haiku-4-5"


# ── Part 3: infra/quota classification ──


@pytest.mark.asyncio
async def test_usage_limit_worker_failure_classified_as_infra(store, runner, orchestrator):
    job = _seed_running_job(store, orchestrator, runner, acceptance_criteria=["x"])
    failed = _worker_session(store.get(job.id), "worker-1", status=SessionStatus.FAILED.value)
    failed.error = "Claude AI usage limit reached|1751990400"
    await orchestrator.on_session_complete(failed, "")

    refreshed = store.get(job.id)
    assert refreshed.state == JobState.ERRORED.value
    assert "infrastructure/quota" in (refreshed.error or ""), "quota death must be labeled infra, not quality"
    assert refreshed.verify_attempts == 0, "an infra failure must not consume a verifier attempt"


# ── Part 2: model-escalation ladder ──


@pytest.mark.asyncio
async def test_ladder_create_starts_on_first_rung(store, runner, orchestrator):
    job, _worker = await orchestrator.create_and_start_job(
        parent_session_id="parent-1",
        prompt="p",
        acceptance_criteria=["x"],
        model_ladder=["claude_code:haiku", "claude_code:opus"],
    )
    fresh = store.get(job.id)
    assert fresh.model == "claude_code:haiku"
    assert fresh.model_ladder == ["claude_code:haiku", "claude_code:opus"]
    assert runner.started[-1].model == "claude_code:haiku"


@pytest.mark.asyncio
async def test_ladder_escalates_on_verifier_exhaustion(store, runner, orchestrator):
    job, _worker = await orchestrator.create_and_start_job(
        parent_session_id="parent-1",
        prompt="p",
        acceptance_criteria=["x"],
        model_ladder=["claude_code:haiku", "claude_code:opus"],
    )
    worker_id = store.get(job.id).worker_session_id
    for i in range(MAX_VERIFY_ATTEMPTS):
        wid = store.get(job.id).worker_session_id if i else worker_id
        await orchestrator.on_session_complete(
            Session(
                id=wid,
                agent="default",
                source="spawned",
                status=SessionStatus.COMPLETED.value,
                metadata={"job_id": job.id},
            ),
            f"w{i}",
        )
        await orchestrator.on_session_complete(_verifier_session(store.get(job.id), f"v{i}"), FAIL_VERDICT)

    refreshed = store.get(job.id)
    assert refreshed.state == JobState.RUNNING.value, "rung exhaustion must escalate, not go stuck"
    assert refreshed.model == "claude_code:opus"
    assert refreshed.verify_attempts == 0, "the new rung gets a fresh verifier budget"
    assert runner.started[-1].model == "claude_code:opus"
    kinds = [a.get("kind") for a in refreshed.attempts]
    assert "escalation" in kinds

    # Exhaust the last rung -> STUCK for real.
    for i in range(MAX_VERIFY_ATTEMPTS):
        wid = store.get(job.id).worker_session_id
        await orchestrator.on_session_complete(
            Session(
                id=wid,
                agent="default",
                source="spawned",
                status=SessionStatus.COMPLETED.value,
                metadata={"job_id": job.id},
            ),
            f"w2{i}",
        )
        await orchestrator.on_session_complete(_verifier_session(store.get(job.id), f"v2{i}"), FAIL_VERDICT)
    assert store.get(job.id).state == JobState.STUCK.value


@pytest.mark.asyncio
async def test_ladder_escalates_on_infra_failure_without_burning_budget(store, runner, orchestrator):
    job, _worker = await orchestrator.create_and_start_job(
        parent_session_id="parent-1",
        prompt="p",
        acceptance_criteria=["x"],
        model_ladder=["claude_code:haiku", "anthropic:claude-opus-4-8"],
    )
    wid = store.get(job.id).worker_session_id
    failed = Session(
        id=wid, agent="default", source="spawned", status=SessionStatus.FAILED.value, metadata={"job_id": job.id}
    )
    failed.error = "429 rate limit exceeded, please retry later"
    await orchestrator.on_session_complete(failed, "")

    refreshed = store.get(job.id)
    assert refreshed.state == JobState.RUNNING.value, "quota failure with a ladder must escalate immediately"
    assert refreshed.model == "anthropic:claude-opus-4-8"
    assert refreshed.verify_attempts == 0
    assert runner.started[-1].model == "anthropic:claude-opus-4-8"


@pytest.mark.asyncio
async def test_non_infra_worker_failure_does_not_escalate(store, runner, orchestrator):
    job, _worker = await orchestrator.create_and_start_job(
        parent_session_id="parent-1",
        prompt="p",
        acceptance_criteria=["x"],
        model_ladder=["claude_code:haiku", "claude_code:opus"],
    )
    wid = store.get(job.id).worker_session_id
    failed = Session(
        id=wid, agent="default", source="spawned", status=SessionStatus.FAILED.value, metadata={"job_id": job.id}
    )
    failed.error = "SyntaxError in generated code"
    await orchestrator.on_session_complete(failed, "")
    assert store.get(job.id).state == JobState.ERRORED.value, "a genuine failure must not silently burn the ladder"


@pytest.mark.asyncio
async def test_attempts_record_model_used(store, runner, orchestrator):
    job, _worker = await orchestrator.create_and_start_job(
        parent_session_id="parent-1",
        prompt="p",
        acceptance_criteria=["x"],
        model="claude_code:sonnet",
    )
    attempts = store.get(job.id).attempts
    assert attempts and attempts[0].get("model") == "claude_code:sonnet", "each attempt must record the model it ran on"
