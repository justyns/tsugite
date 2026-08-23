import { describe, expect, it } from 'vitest';
import type { ActivityEntry } from '$lib/stores/activity.svelte';
import { entryRoute, groupByDay } from './activityFeed';

// Local noon on a Tuesday, so the day buckets can't straddle a UTC boundary.
const NOW = new Date(2026, 7, 18, 12, 0).getTime();
const DAY = 24 * 3_600_000;

function entry(partial: Partial<ActivityEntry> & { id: string }): ActivityEntry {
  return {
    type: 'session',
    timestamp: new Date(NOW).toISOString(),
    title: 'a chat',
    summary: '',
    status: 'ok',
    label: 'completed',
    session_id: null,
    job_id: null,
    schedule_id: null,
    ...partial,
  };
}

describe('groupByDay', () => {
  it('buckets a newest-first feed into today / yesterday / dated days', () => {
    const groups = groupByDay(
      [
        entry({ id: 'a', timestamp: new Date(NOW).toISOString() }),
        entry({ id: 'b', timestamp: new Date(NOW - 3_600_000).toISOString() }),
        entry({ id: 'c', timestamp: new Date(NOW - DAY).toISOString() }),
        entry({ id: 'd', timestamp: new Date(NOW - 4 * DAY).toISOString() }),
      ],
      NOW,
    );

    expect(groups.map((g) => g.label)).toEqual(['today', 'yesterday', 'fri aug 14']);
    expect(groups[0]!.entries.map((e) => e.id)).toEqual(['a', 'b']);
    expect(groups[1]!.entries.map((e) => e.id)).toEqual(['c']);
    expect(groups[2]!.entries.map((e) => e.id)).toEqual(['d']);
  });

  it('returns no groups for an empty feed', () => {
    expect(groupByDay([], NOW)).toEqual([]);
  });
});

describe('entryRoute', () => {
  it('sends a job row to the jobs board and a session row to its chat', () => {
    expect(
      entryRoute(entry({ id: 'j', type: 'job', job_id: 'job-1', session_id: 'chat-1' })),
    ).toEqual({ view: 'jobs', params: {} });
    expect(entryRoute(entry({ id: 's', session_id: 'chat-1' }))).toEqual({
      view: 'chats',
      params: { sessionId: 'chat-1' },
    });
  });

  it('sends a schedule run to its session, falling back to the schedules view', () => {
    expect(
      entryRoute(entry({ id: 'r', type: 'schedule', schedule_id: 'feeds', session_id: 'run-1' })),
    ).toEqual({ view: 'chats', params: { sessionId: 'run-1' } });
    expect(entryRoute(entry({ id: 'r2', type: 'schedule', schedule_id: 'feeds' }))).toEqual({
      view: 'schedules',
      params: {},
    });
  });

  it('links nowhere when the row names no target', () => {
    expect(entryRoute(entry({ id: 'x' }))).toBeNull();
  });
});
