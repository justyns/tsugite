/**
 * Pure derivations over a Job payload: acceptance-criteria states, criteria
 * counts, and the attempt counter. Kept out of the .svelte files so the
 * reducer is directly node-testable.
 *
 * AC data comes from the backend already: `acceptance_criteria`
 * is the ordered prompt list; `ac_results` accumulates one verdict per criterion
 * per verifier round ({ac_index, ac_text, pass, reason, attempt}). Terminal
 * snapshots also mirror the verdicts into `result.ac_results`, so that is the
 * fallback source when the top-level list is empty.
 */
import type { Job, JobAcResult } from '$lib/stores/jobs.svelte';
import type { CheckState } from '$lib/components/rows/rowState';

export interface AcRow {
  index: number;
  label: string;
  state: CheckState;
  /** Verifier's reason, surfaced on a fail. */
  note?: string;
}

const TERMINAL = new Set(['done', 'cancelled', 'stuck', 'errored']);
export function isTerminal(state: string): boolean {
  return TERMINAL.has(state);
}

function resultsFor(job: Job): JobAcResult[] {
  if (job.ac_results && job.ac_results.length) return job.ac_results;
  const mirrored = job.result?.['ac_results'];
  return Array.isArray(mirrored) ? (mirrored as JobAcResult[]) : [];
}

/** Latest verdict per criterion index (highest `attempt` wins). */
function latestByIndex(results: JobAcResult[]): Map<number, JobAcResult> {
  const latest = new Map<number, JobAcResult>();
  for (const r of results) {
    const prev = latest.get(r.ac_index);
    if (!prev || (r.attempt ?? 0) >= (prev.attempt ?? 0)) latest.set(r.ac_index, r);
  }
  return latest;
}

/**
 * One row per acceptance criterion, folded to a checklist state:
 *   - a recorded verdict -> pass | fail (fail carries the verifier's reason)
 *   - no verdict yet, job verifying -> the first such row is `active`
 *   - otherwise -> pending
 */
export function acRows(job: Job): AcRow[] {
  const criteria = job.acceptance_criteria ?? [];
  const latest = latestByIndex(resultsFor(job));
  const verifying = job.state === 'verifying';
  let activeUsed = false;
  return criteria.map((label, i) => {
    const r = latest.get(i);
    if (r) {
      return r.pass
        ? { index: i, label, state: 'pass' }
        : { index: i, label, state: 'fail', note: r.reason || undefined };
    }
    if (verifying && !activeUsed) {
      activeUsed = true;
      return { index: i, label, state: 'active' };
    }
    return { index: i, label, state: 'pending' };
  });
}

export interface AcCounts {
  pass: number;
  fail: number;
  /** pending + active - criteria with no final verdict yet. */
  remaining: number;
  total: number;
}

export function acCounts(rows: AcRow[]): AcCounts {
  let pass = 0;
  let fail = 0;
  for (const r of rows) {
    if (r.state === 'pass') pass += 1;
    else if (r.state === 'fail') fail += 1;
  }
  return { pass, fail, remaining: rows.length - pass - fail, total: rows.length };
}

/** How many verifier rounds a job has run (the recorded attempts, falling back
 *  to the running counter). Zero for a job still queued. */
export function attemptCount(job: Job): number {
  return Math.max(job.attempts?.length ?? 0, job.verify_attempts ?? 0);
}
