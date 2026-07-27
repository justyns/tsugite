import { describe, it, expect } from 'vitest';
import type { Schedule } from '$lib/stores/schedules.svelte';
import {
  deriveRunStatus,
  sortSchedules,
  summarize,
  nextUp,
  buildSpark,
  formatNextRun,
  formatAgo,
  formatDuration,
  formatStamp,
  recentRuns,
} from './schedulesView';

const NOW = Date.parse('2026-07-14T12:00:00Z'); // a Tuesday

function sched(partial: Partial<Schedule>): Schedule {
  return {
    id: 'x',
    agent: 'smoke',
    prompt: 'p',
    schedule_type: 'cron',
    cron_expr: '0 * * * *',
    run_at: null,
    enabled: true,
    created_at: '2026-01-01T00:00:00Z',
    last_run: null,
    next_run: null,
    last_status: null,
    last_error: null,
    timezone: 'UTC',
    execution_type: 'agent',
    command: null,
    run_count: 0,
    disabled_reason: null,
    ...partial,
  };
}

function iso(offsetMs: number): string {
  return new Date(NOW + offsetMs).toISOString();
}

describe('deriveRunStatus', () => {
  it('maps enabled + last_status, and disabled -> off', () => {
    expect(deriveRunStatus({ enabled: false, last_status: 'success' })).toBe('off');
    expect(deriveRunStatus({ enabled: true, last_status: 'success' })).toBe('done');
    expect(deriveRunStatus({ enabled: true, last_status: 'error' })).toBe('errored');
    expect(deriveRunStatus({ enabled: true, last_status: 'skipped' })).toBe('skipped');
    expect(deriveRunStatus({ enabled: true, last_status: null })).toBe('queued');
  });
});

describe('sortSchedules', () => {
  const a = sched({ id: 'a', next_run: iso(4 * 60_000) });
  const b = sched({ id: 'b', next_run: iso(60 * 60_000) });
  const off = sched({ id: 'off', next_run: null });

  it('ascending: soonest first, null-next last', () => {
    expect(sortSchedules([b, off, a], 'ascending').map((s) => s.id)).toEqual(['a', 'b', 'off']);
  });

  it('descending: furthest first, null-next first', () => {
    expect(sortSchedules([a, off, b], 'descending').map((s) => s.id)).toEqual(['off', 'b', 'a']);
  });

  it('sorts a disabled schedule to the end even when it still carries a next_run', () => {
    const disabled = sched({ id: 'dis', enabled: false, next_run: iso(2 * 60_000) });
    expect(sortSchedules([disabled, a, b], 'ascending').map((s) => s.id)).toEqual([
      'a',
      'b',
      'dis',
    ]);
  });

  it('is stable for equal keys and does not mutate input', () => {
    const x = sched({ id: 'x', next_run: null });
    const y = sched({ id: 'y', next_run: null });
    const input = [x, y];
    expect(sortSchedules(input, 'ascending').map((s) => s.id)).toEqual(['x', 'y']);
    expect(input.map((s) => s.id)).toEqual(['x', 'y']);
  });
});

describe('summarize', () => {
  it('counts total, actively-failing (enabled+error), and disabled', () => {
    const list = [
      sched({ enabled: true, last_status: 'success' }),
      sched({ enabled: true, last_status: 'error' }),
      sched({ enabled: false, last_status: 'error' }), // disabled, not counted as failing
      sched({ enabled: false, last_status: 'success' }),
    ];
    expect(summarize(list)).toEqual({ total: 4, failing: 1, disabled: 2 });
  });
});

describe('nextUp', () => {
  it('picks the soonest enabled future run, ignoring disabled/past/null', () => {
    const soon = sched({ id: 'soon', next_run: iso(5 * 60_000) });
    const later = sched({ id: 'later', next_run: iso(60 * 60_000) });
    const past = sched({ id: 'past', next_run: iso(-60_000) });
    const disabled = sched({ id: 'd', enabled: false, next_run: iso(60_000) });
    const res = nextUp([later, disabled, past, soon], NOW);
    expect(res?.schedule.id).toBe('soon');
  });

  it('returns null when nothing is upcoming', () => {
    expect(nextUp([sched({ enabled: false, next_run: iso(60_000) })], NOW)).toBeNull();
  });
});

describe('buildSpark', () => {
  it('labels an empty / non-array history', () => {
    expect(buildSpark([])).toEqual({ points: [], label: 'no recent runs' });
    expect(buildSpark(undefined).label).toBe('no recent runs');
    expect(buildSpark(null).label).toBe('no recent runs');
  });

  it('maps statuses and summarizes all-ok', () => {
    const res = buildSpark([{ status: 'success' }, { status: 'success' }]);
    expect(res.label).toBe('last 2 runs: all ok');
    expect(res.points.map((p) => p.status)).toEqual(['ok', 'ok']);
  });

  it('counts failures and skips', () => {
    const hist = [
      { status: 'success' },
      { status: 'error' },
      { status: 'skipped' },
      { status: 'error' },
    ];
    const res = buildSpark(hist);
    expect(res.label).toBe('last 4 runs: 2 failed, 1 skipped');
    expect(res.points.map((p) => p.status)).toEqual(['ok', 'fail', 'skip', 'fail']);
  });

  it('caps to the last 10 runs (most recent kept)', () => {
    const hist = Array.from({ length: 12 }, (_, i) => ({ status: i === 11 ? 'error' : 'success' }));
    const res = buildSpark(hist);
    expect(res.points).toHaveLength(10);
    expect(res.label).toBe('last 10 runs: 1 failed');
    expect(res.points.at(-1)?.status).toBe('fail');
  });
});

describe('formatNextRun', () => {
  it('returns an em-dash when there is no next run', () => {
    expect(formatNextRun(null, 'UTC', NOW)).toBe('—');
    expect(formatNextRun('not-a-date', 'UTC', NOW)).toBe('—');
  });

  it('renders sub-hour and multi-hour countdowns', () => {
    expect(formatNextRun(iso(4 * 60_000), 'UTC', NOW)).toBe('in 4m');
    expect(formatNextRun(iso(22 * 60_000), 'UTC', NOW)).toBe('in 22m');
    expect(formatNextRun(iso(2 * 3_600_000), 'UTC', NOW)).toBe('in 2h');
    expect(formatNextRun(iso(90 * 60_000), 'UTC', NOW)).toBe('in 1h 30m');
    expect(formatNextRun(iso(751 * 60_000), 'UTC', NOW)).toBe('in 12h 31m');
    expect(formatNextRun(iso(-1000), 'UTC', NOW)).toBe('due');
  });

  it('switches to an absolute weekday/time past the relative horizon', () => {
    // +18h from Tue 12:00Z -> Wed 06:00 UTC
    expect(formatNextRun(iso(18 * 3_600_000), 'UTC', NOW)).toBe('wed 06:00');
  });

  it('renders in the schedule timezone, not UTC', () => {
    // 2026-07-15T18:00Z is 14:00 in New York (EDT, UTC-4), still Wednesday.
    const at = new Date(Date.parse('2026-07-15T18:00:00Z')).toISOString();
    expect(formatNextRun(at, 'America/New_York', NOW)).toBe('wed 14:00');
  });

  it('renders a short date beyond a week', () => {
    expect(formatNextRun(iso(10 * 24 * 3_600_000), 'UTC', NOW)).toBe('jul 24');
  });

  it('falls back to UTC for an unknown timezone rather than throwing', () => {
    expect(formatNextRun(iso(18 * 3_600_000), 'Mars/Olympus', NOW)).toBe('wed 06:00');
  });
});

describe('formatAgo', () => {
  it('renders never / relative buckets / old date', () => {
    expect(formatAgo(null, NOW)).toBe('never');
    expect(formatAgo(iso(-30_000), NOW)).toBe('just now');
    expect(formatAgo(iso(-5 * 60_000), NOW)).toBe('5m ago');
    expect(formatAgo(iso(-3 * 3_600_000), NOW)).toBe('3h ago');
    expect(formatAgo(iso(-3 * 24 * 3_600_000), NOW)).toBe('3d ago');
    expect(formatAgo(iso(-40 * 24 * 3_600_000), NOW)).toBe('jun 4');
  });
});

describe('formatStamp', () => {
  it('renders a short local month/day/time, empty for missing/bad input', () => {
    // Built in local time so the local-tz render round-trips regardless of runner TZ.
    const local = new Date(2026, 6, 11, 3, 0).toISOString();
    expect(formatStamp(local)).toBe('jul 11 03:00');
    expect(formatStamp(null)).toBe('');
    expect(formatStamp('nope')).toBe('');
  });
});

describe('formatDuration', () => {
  it('formats seconds/minutes/hours', () => {
    expect(formatDuration(4_000)).toBe('4s');
    expect(formatDuration(9 * 60_000 + 12_000)).toBe('9m 12s');
    expect(formatDuration(3_600_000 + 3 * 60_000)).toBe('1h 03m');
    expect(formatDuration(-50)).toBe('0s');
  });
});

describe('recentRuns', () => {
  const at = (day: number) => `2026-07-${String(day).padStart(2, '0')}T08:00:00Z`;

  it('orders newest first and caps at the limit', () => {
    const runs = Array.from({ length: 14 }, (_, i) => ({ id: `r${i}`, created_at: at(i + 1) }));
    const out = recentRuns(runs);
    expect(out).toHaveLength(10);
    expect(out[0]!.id).toBe('r13');
    expect(out[9]!.id).toBe('r4');
  });

  it('sorts unparsable stamps last and keeps ties stable', () => {
    const out = recentRuns([
      { id: 'bad', created_at: null },
      { id: 'a', created_at: at(2) },
      { id: 'b', created_at: at(2) },
      { id: 'new', created_at: at(9) },
    ]);
    expect(out.map((r) => r.id)).toEqual(['new', 'a', 'b', 'bad']);
  });
});
