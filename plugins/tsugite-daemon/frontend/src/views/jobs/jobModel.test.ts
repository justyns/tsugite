import { describe, expect, it } from 'vitest';
import { acCounts, acRows, attemptCount, isTerminal } from './jobModel';
import type { Job, JobAcResult } from '$lib/stores/jobs.svelte';

function job(partial: Partial<Job>): Job {
  return {
    job_id: 'job-1',
    state: 'running',
    prompt: 'p',
    acceptance_criteria: [],
    ac_results: [],
    attempts: [],
    verify_attempts: 0,
    max_attempts: 3,
    result: null,
    ...partial,
  } as Job;
}
const res = (i: number, pass: boolean, attempt = 1, reason = ''): JobAcResult => ({
  ac_index: i,
  ac_text: `ac${i}`,
  pass,
  reason,
  attempt,
});

describe('acRows', () => {
  it('folds a recorded verdict into pass/fail and carries the fail reason', () => {
    const rows = acRows(
      job({
        acceptance_criteria: ['a', 'b'],
        ac_results: [res(0, true), res(1, false, 1, 'disk at 84%')],
      }),
    );
    expect(rows[0]).toMatchObject({ state: 'pass', label: 'a' });
    expect(rows[1]).toMatchObject({ state: 'fail', label: 'b', note: 'disk at 84%' });
  });

  it('uses the latest attempt when a criterion was re-graded', () => {
    const rows = acRows(
      job({ acceptance_criteria: ['a'], ac_results: [res(0, false, 1, 'no'), res(0, true, 2)] }),
    );
    expect(rows[0]?.state).toBe('pass');
  });

  it('marks only the first ungraded criterion active while verifying', () => {
    const rows = acRows(
      job({ state: 'verifying', acceptance_criteria: ['a', 'b', 'c'], ac_results: [res(0, true)] }),
    );
    expect(rows.map((r) => r.state)).toEqual(['pass', 'active', 'pending']);
  });

  it('leaves ungraded criteria pending when not verifying', () => {
    const rows = acRows(job({ state: 'queued', acceptance_criteria: ['a', 'b'], ac_results: [] }));
    expect(rows.map((r) => r.state)).toEqual(['pending', 'pending']);
  });

  it('falls back to result.ac_results when the top-level list is empty', () => {
    const rows = acRows(
      job({
        state: 'done',
        acceptance_criteria: ['a'],
        ac_results: [],
        result: { ac_results: [res(0, true)] },
      }),
    );
    expect(rows[0]?.state).toBe('pass');
  });
});

describe('acCounts', () => {
  it('splits pass / fail / remaining and totals', () => {
    const rows = acRows(
      job({
        state: 'verifying',
        acceptance_criteria: ['a', 'b', 'c', 'd'],
        ac_results: [res(0, true), res(1, false, 1, 'x')],
      }),
    );
    // pass=1, fail=1, then first ungraded is active + one pending -> remaining=2
    expect(acCounts(rows)).toEqual({ pass: 1, fail: 1, remaining: 2, total: 4 });
  });
});

describe('attemptCount', () => {
  it('is the recorded attempts length, falling back to the counter', () => {
    expect(attemptCount(job({ attempts: [{}, {}] as never }))).toBe(2);
    expect(attemptCount(job({ attempts: [], verify_attempts: 3 }))).toBe(3);
    expect(attemptCount(job({ state: 'queued' }))).toBe(0);
  });
});

describe('isTerminal', () => {
  it('is true only for parked/finished states', () => {
    expect(['done', 'cancelled', 'stuck', 'errored'].every(isTerminal)).toBe(true);
    expect(['queued', 'running', 'verifying', 'awaiting_input'].some(isTerminal)).toBe(false);
  });
});
