/**
 * Board taxonomy for the Jobs view: four kanban columns
 * (queued / active / needs-you / resolved) over the eight backend JobStates,
 * plus the summary-pill counts and the toolbar sort orders. Pure and
 * node-testable.
 *
 * The canonical state->group lists live in the data layer's jobsFilter
 * (`JOB_GROUPS`, mirroring the backend `?state=` aliases); this module reuses
 * them and only adds a standalone `queued` column, which no backend
 * alias covers, and renames the `stuck` group to "needs you".
 */
import { JOB_GROUPS, type JobLike } from '$lib/stores/jobsFilter';

export type BoardCol = 'queued' | 'active' | 'needs-you' | 'resolved';

/** Column order, left to right. */
export const BOARD_COLS: readonly BoardCol[] = ['queued', 'active', 'needs-you', 'resolved'];

export const BOARD_COL_LABEL: Record<BoardCol, string> = {
  queued: 'queued',
  active: 'active',
  'needs-you': 'needs you',
  resolved: 'resolved',
};

const COL_STATES: Record<BoardCol, readonly string[]> = {
  queued: ['queued'],
  active: JOB_GROUPS.active,
  'needs-you': JOB_GROUPS.stuck,
  resolved: JOB_GROUPS.resolved,
};

/** Which board column a job state belongs to (null for an unmapped state). */
export function boardColForState(state: string | undefined): BoardCol | null {
  if (!state) return null;
  const s = state.toLowerCase();
  for (const col of BOARD_COLS) if (COL_STATES[col].includes(s)) return col;
  return null;
}

/** The `needs you` column is the one that demands operator action, so its
 *  filter pill and column header get the warm "attention" treatment. */
export const ATTENTION_COL: BoardCol = 'needs-you';

export type FilterKey = 'all' | BoardCol;
export const FILTER_KEYS: readonly FilterKey[] = ['all', ...BOARD_COLS];

export type FilterCounts = Record<FilterKey, number>;

/** Per-pill counts. `all` is the true total (so an unmapped future state still
 *  shows up under "all"); each column counts only the states it owns. */
export function filterCounts(jobs: JobLike[]): FilterCounts {
  const counts: FilterCounts = {
    all: jobs.length,
    queued: 0,
    active: 0,
    'needs-you': 0,
    resolved: 0,
  };
  for (const j of jobs) {
    const col = boardColForState(j.state);
    if (col) counts[col] += 1;
  }
  return counts;
}

/** Restrict to a single column (or pass through for `all`). */
export function applyColumnFilter<T extends JobLike>(jobs: T[], key: FilterKey): T[] {
  if (key === 'all') return jobs;
  return jobs.filter((j) => boardColForState(j.state) === key);
}

/** Bucket jobs into their columns, preserving the incoming order within each. */
export function groupByColumn<T extends JobLike>(jobs: T[]): Record<BoardCol, T[]> {
  const out: Record<BoardCol, T[]> = { queued: [], active: [], 'needs-you': [], resolved: [] };
  for (const j of jobs) {
    const col = boardColForState(j.state);
    if (col) out[col].push(j);
  }
  return out;
}

// ---------- sorting ----------

export type SortMode = 'urgency' | 'updated' | 'created';
export const SORT_MODES: readonly SortMode[] = ['urgency', 'updated', 'created'];
export const SORT_LABEL: Record<SortMode, string> = {
  urgency: 'sort: urgency',
  updated: 'sort: updated',
  created: 'sort: created',
};

// Urgency puts the columns that need a human first; resolved sinks to the
// bottom. Ties break on most-recently-updated.
const URGENCY_RANK: Record<BoardCol, number> = {
  'needs-you': 0,
  active: 1,
  queued: 2,
  resolved: 3,
};

type Sortable = JobLike & { updated_at?: string; created_at?: string };

// ISO-8601 timestamps at a fixed offset sort lexicographically by time, newest
// first with a descending string compare.
function byDesc(a: string | undefined, b: string | undefined): number {
  return (b ?? '').localeCompare(a ?? '');
}

export function sortJobs<T extends Sortable>(jobs: T[], mode: SortMode): T[] {
  const copy = jobs.slice();
  if (mode === 'created') {
    copy.sort((a, b) => byDesc(a.created_at, b.created_at));
  } else if (mode === 'updated') {
    copy.sort((a, b) => byDesc(a.updated_at, b.updated_at));
  } else {
    copy.sort((a, b) => {
      const ra = URGENCY_RANK[boardColForState(a.state) ?? 'resolved'];
      const rb = URGENCY_RANK[boardColForState(b.state) ?? 'resolved'];
      return ra !== rb ? ra - rb : byDesc(a.updated_at, b.updated_at);
    });
  }
  return copy;
}
