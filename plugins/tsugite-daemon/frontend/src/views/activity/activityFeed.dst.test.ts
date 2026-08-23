// DST regression for the "yesterday" heading. Own file: it pins process.env.TZ,
// which is process-wide, so it must not share a file with TZ-neutral tests.
import { afterAll, beforeAll, expect, test } from 'vitest';
import type { ActivityEntry } from '$lib/stores/activity.svelte';
import { groupByDay } from './activityFeed';

const REAL_TZ = process.env.TZ;
beforeAll(() => {
  process.env.TZ = 'America/New_York';
});
afterAll(() => {
  if (REAL_TZ === undefined) delete process.env.TZ;
  else process.env.TZ = REAL_TZ;
});

function entry(id: string, timestamp: string): ActivityEntry {
  return {
    id,
    type: 'session',
    timestamp,
    title: 't',
    summary: '',
    status: null,
    label: 'completed',
    session_id: null,
    job_id: null,
    schedule_id: null,
  };
}

test('the yesterday heading survives spring-forward (23-hour day)', () => {
  // 00:30 the night after DST began (Mar 8 2026): now - 24h is still Mar 7.
  const now = new Date(2026, 2, 9, 0, 30).getTime();
  const groups = groupByDay([entry('a', new Date(2026, 2, 8, 12, 0).toISOString())], now);
  expect(groups.map((g) => g.label)).toEqual(['yesterday']);
});

test('the yesterday heading survives fall-back (25-hour day)', () => {
  // 23:30 on the day DST ended (Nov 1 2026): now - 24h is still Nov 1, so a
  // wall-clock "yesterday" never matches Oct 31.
  const now = new Date(2026, 10, 1, 23, 30).getTime();
  const groups = groupByDay([entry('a', new Date(2026, 9, 31, 12, 0).toISOString())], now);
  expect(groups.map((g) => g.label)).toEqual(['yesterday']);
});
