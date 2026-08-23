/// <reference types="@vitest/browser/context" />
import { page } from '@vitest/browser/context';
import { render } from 'vitest-browser-svelte';
import { afterEach, beforeEach, expect, test, vi } from 'vitest';

// The feed fetches on mount and again on every filter change; serve it from the
// seed, honouring the `types` filter the way the endpoint does.
vi.mock('$lib/api/client', () => ({
  authHeaders: () => ({}),
  api: {
    get: vi.fn(async (path: string) => {
      const types = new URL(path, 'http://x').searchParams.get('types');
      return { entries: types ? SEED.filter((e) => e.type === types) : SEED };
    }),
    post: vi.fn(async () => ({})),
    patch: vi.fn(async () => ({})),
    put: vi.fn(async () => ({})),
    del: vi.fn(async () => ({})),
  },
}));

import View from './View.svelte';
import { activity, type ActivityEntry } from '$lib/stores/activity.svelte';
import { TESTID } from '$lib/testids';

const NOW = Date.now();
const HOUR = 3_600_000;

function entry(partial: Partial<ActivityEntry> & { id: string }): ActivityEntry {
  return {
    type: 'session',
    timestamp: new Date(NOW).toISOString(),
    title: 'untitled',
    summary: '',
    status: 'ok',
    label: 'completed',
    session_id: null,
    job_id: null,
    schedule_id: null,
    ...partial,
  };
}

// Newest first, exactly as the endpoint returns it.
const SEED: ActivityEntry[] = [
  entry({ id: 'session:chat-1:9', title: 'Morning triage', session_id: 'chat-1' }),
  entry({
    id: 'job:job-1',
    type: 'job',
    title: 'ship the release',
    label: 'done',
    job_id: 'job-1',
    session_id: 'chat-1',
    timestamp: new Date(NOW - HOUR).toISOString(),
  }),
  entry({
    id: 'schedule:feeds:1',
    type: 'schedule',
    title: 'feeds',
    summary: 'timed out',
    status: 'error',
    label: 'error',
    schedule_id: 'feeds',
    session_id: 'run-1',
    timestamp: new Date(NOW - 2 * HOUR).toISOString(),
  }),
];

beforeEach(async () => {
  await page.viewport(1280, 800);
  activity.entries = [];
  activity.error = null;
  location.hash = '';
});

afterEach(() => {
  activity.entries = [];
  location.hash = '';
});

test('the feed renders every source newest-first under a day heading', async () => {
  render(View);
  const feed = page.getByTestId(TESTID.activityFeed);
  await expect.element(feed).toBeInTheDocument();
  await expect
    .element(page.getByTestId(TESTID.activityEntry('session:chat-1:9')))
    .toBeInTheDocument();

  const rows = Array.from(
    document.querySelectorAll<HTMLElement>('[data-testid^="activity-entry-"]'),
  );
  expect(rows.map((r) => r.dataset.testid)).toEqual([
    'activity-entry-session:chat-1:9',
    'activity-entry-job:job-1',
    'activity-entry-schedule:feeds:1',
  ]);
  expect(document.querySelector('[data-testid^="activity-day-"] h3')?.textContent).toBe('today');
  await page.screenshot({ path: '__screenshots__/activity-feed.png' });
});

test('a type filter narrows the feed to that kind', async () => {
  render(View);
  await expect.element(page.getByTestId(TESTID.activityEntry('job:job-1'))).toBeInTheDocument();

  await page.getByTestId(TESTID.activityFilter('schedule')).click();
  await expect
    .element(page.getByTestId(TESTID.activityEntry('schedule:feeds:1')))
    .toBeInTheDocument();
  await expect.element(page.getByTestId(TESTID.activityEntry('job:job-1'))).not.toBeInTheDocument();
});

test('clicking a session row opens that chat', async () => {
  render(View);
  await page.getByTestId(TESTID.activityEntry('session:chat-1:9')).click();
  expect(location.hash).toBe('#chats?sessionId=chat-1');
});

test('a rev bump refetches, and keeps the active filter', async () => {
  // The bare `activity.rev;` in View.svelte's effect is the whole live-update mechanism, and
  // every "remove the dead statement" cleanup wants to delete it. This is what turns red.
  const { api } = await import('$lib/api/client');
  render(View);
  await expect.element(page.getByTestId(TESTID.activityEntry('job:job-1'))).toBeInTheDocument();

  await page.getByTestId(TESTID.activityFilter('schedule')).click();
  await expect
    .element(page.getByTestId(TESTID.activityEntry('schedule:feeds:1')))
    .toBeInTheDocument();

  const before = vi.mocked(api.get).mock.calls.length;
  activity.rev += 1;
  await vi.waitFor(() => expect(vi.mocked(api.get).mock.calls.length).toBeGreaterThan(before));
  expect(vi.mocked(api.get).mock.calls.at(-1)?.[0]).toContain('types=schedule');
});
