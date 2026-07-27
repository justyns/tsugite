import { describe, expect, it } from 'vitest';
import {
  BOARD_COLS,
  applyColumnFilter,
  boardColForState,
  filterCounts,
  groupByColumn,
  sortJobs,
} from './board';
import type { JobLike } from '$lib/stores/jobsFilter';

type J = JobLike & { updated_at?: string; created_at?: string };
const j = (state: string, extra: Partial<J> = {}): J => ({ job_id: state, state, ...extra });

describe('boardColForState', () => {
  it('maps all eight backend states onto the four design columns', () => {
    expect(boardColForState('queued')).toBe('queued');
    expect(boardColForState('running')).toBe('active');
    expect(boardColForState('verifying')).toBe('active');
    expect(boardColForState('awaiting_input')).toBe('needs-you');
    expect(boardColForState('stuck')).toBe('needs-you');
    expect(boardColForState('errored')).toBe('needs-you');
    expect(boardColForState('done')).toBe('resolved');
    expect(boardColForState('cancelled')).toBe('resolved');
  });

  it('is case-insensitive and null for unknown/empty', () => {
    expect(boardColForState('RUNNING')).toBe('active');
    expect(boardColForState('bogus')).toBeNull();
    expect(boardColForState(undefined)).toBeNull();
  });

  it('every column owns at least one state', () => {
    for (const col of BOARD_COLS) {
      const states = [
        'queued',
        'running',
        'verifying',
        'awaiting_input',
        'stuck',
        'errored',
        'done',
        'cancelled',
      ];
      expect(states.some((s) => boardColForState(s) === col)).toBe(true);
    }
  });
});

describe('filterCounts', () => {
  it('counts per column and totals everything under all', () => {
    const jobs = [
      j('queued'),
      j('running'),
      j('verifying'),
      j('awaiting_input'),
      j('stuck'),
      j('done'),
    ];
    const c = filterCounts(jobs);
    expect(c).toEqual({ all: 6, queued: 1, active: 2, 'needs-you': 2, resolved: 1 });
  });

  it('all counts an unmapped state even when no column claims it', () => {
    const c = filterCounts([j('queued'), j('mystery')]);
    expect(c.all).toBe(2);
    expect(c.queued + c.active + c['needs-you'] + c.resolved).toBe(1);
  });
});

describe('applyColumnFilter / groupByColumn', () => {
  const jobs = [j('queued'), j('running'), j('stuck'), j('done')];
  it('all passes through unchanged', () => {
    expect(applyColumnFilter(jobs, 'all')).toHaveLength(4);
  });
  it('a column key restricts to that column', () => {
    expect(applyColumnFilter(jobs, 'needs-you').map((x) => x.state)).toEqual(['stuck']);
  });
  it('groups jobs into their columns preserving order', () => {
    const g = groupByColumn([j('running'), j('verifying'), j('queued')]);
    expect(g.active.map((x) => x.state)).toEqual(['running', 'verifying']);
    expect(g.queued.map((x) => x.state)).toEqual(['queued']);
    expect(g['needs-you']).toEqual([]);
  });
});

describe('sortJobs', () => {
  it('urgency ranks needs-you > active > queued > resolved', () => {
    const jobs = [
      j('done', { updated_at: '2026-01-01T05:00:00Z' }),
      j('queued', { updated_at: '2026-01-01T04:00:00Z' }),
      j('running', { updated_at: '2026-01-01T03:00:00Z' }),
      j('stuck', { updated_at: '2026-01-01T02:00:00Z' }),
    ];
    expect(sortJobs(jobs, 'urgency').map((x) => x.state)).toEqual([
      'stuck',
      'running',
      'queued',
      'done',
    ]);
  });

  it('urgency breaks ties by most-recently-updated', () => {
    const jobs = [
      j('running', { updated_at: '2026-01-01T01:00:00Z' }),
      j('verifying', { updated_at: '2026-01-01T09:00:00Z' }),
    ];
    expect(sortJobs(jobs, 'urgency').map((x) => x.state)).toEqual(['verifying', 'running']);
  });

  it('updated / created sort newest-first and do not mutate the input', () => {
    const jobs = [
      j('a', { updated_at: '2026-01-01T01:00:00Z', created_at: '2026-01-01T09:00:00Z' }),
      j('b', { updated_at: '2026-01-01T08:00:00Z', created_at: '2026-01-01T02:00:00Z' }),
    ];
    expect(sortJobs(jobs, 'updated').map((x) => x.state)).toEqual(['b', 'a']);
    expect(sortJobs(jobs, 'created').map((x) => x.state)).toEqual(['a', 'b']);
    expect(jobs.map((x) => x.state)).toEqual(['a', 'b']);
  });
});
