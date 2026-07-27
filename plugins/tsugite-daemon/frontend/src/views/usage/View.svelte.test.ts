/// <reference types="vitest/browser" />
import { page } from 'vitest/browser';
import { render } from 'vitest-browser-svelte';
import { afterEach, beforeEach, expect, test, vi } from 'vitest';
import View from './View.svelte';
import { usage } from '$lib/stores/usage.svelte';
import { api } from '$lib/api/client';

const SUMMARY = [
  {
    period: '2026-07-14',
    runs: 4,
    total_tokens: 86754,
    total_cost: 2.14,
    input_tokens: 1,
    output_tokens: 1,
    cache_creation_tokens: 12000,
    cache_read_tokens: 50000,
    total_duration_ms: 1,
  },
  {
    period: '2026-07-13',
    runs: 2,
    total_tokens: 40000,
    total_cost: 2.06,
    input_tokens: 1,
    output_tokens: 1,
    cache_creation_tokens: 0,
    cache_read_tokens: 0,
    total_duration_ms: 1,
  },
];
const AGENTS = [
  {
    agent: 'odyn',
    runs: 5,
    total_tokens: 90000,
    total_cost: 3.2,
    cache_creation_tokens: 15000,
    cache_read_tokens: 60000,
  },
];
const MODELS = [
  {
    model: 'claude_code:opus',
    runs: 5,
    total_tokens: 90000,
    total_cost: 3.2,
    cache_creation_tokens: 0,
    cache_read_tokens: 0,
  },
];
const SCHEDULES = [
  {
    schedule_name: 'morning-report',
    runs: 12,
    total_tokens: 340000,
    total_cost: 1.1,
    cache_creation_tokens: 1000,
    cache_read_tokens: 9000,
    last_run: '2026-07-09T08:00:00+00:00',
  },
  {
    schedule_name: null,
    runs: 3,
    total_tokens: 5000,
    total_cost: 0.05,
    cache_creation_tokens: 0,
    cache_read_tokens: 0,
    last_run: '2026-07-05T22:30:00+00:00',
  },
];
const TOTAL = {
  runs: 6,
  total_tokens: 126754,
  total_cost: 4.2,
  input_tokens: 1,
  output_tokens: 1,
  cache_creation_tokens: 12000,
  cache_read_tokens: 50000,
};

function okResponses(path: string): unknown {
  if (path.startsWith('/api/usage/summary')) return SUMMARY;
  if (path.startsWith('/api/usage/agents')) return AGENTS;
  if (path.startsWith('/api/usage/models')) return MODELS;
  if (path.startsWith('/api/usage/schedules')) return SCHEDULES;
  if (path.startsWith('/api/usage/total')) return TOTAL;
  throw new Error(`unexpected path: ${path}`);
}

function mockGet(impl: (path: string) => Promise<unknown>) {
  return vi.spyOn(api, 'get').mockImplementation(impl as never);
}

// `usage` is a module-level singleton (shared across every test in this file),
// so each test starts it from a known state rather than trusting leftovers
// from whichever test ran before it.
beforeEach(() => {
  usage.summary = [];
  usage.agents = [];
  usage.models = [];
  usage.schedules = [];
  usage.total = null;
  usage.loading = false;
  usage.error = null;
  usage.range = { sinceDays: 30, period: 'day' };
});

afterEach(() => {
  vi.restoreAllMocks();
});

test('loads on mount and renders totals, top tables, and the per-day meter table', async () => {
  mockGet((path) => Promise.resolve(okResponses(path)));
  await render(View);

  await expect.element(page.getByText('$4.20')).toBeInTheDocument();
  await expect.element(page.getByText('odyn')).toBeInTheDocument();
  await expect.element(page.getByText('claude_code:opus')).toBeInTheDocument();
  await expect.element(page.getByText('jul 14')).toBeInTheDocument();
  await expect.element(page.getByRole('meter', { name: 'jul 14 tokens' })).toBeInTheDocument();
});

test('renders the scheduled-tasks table with named and unattributed rows', async () => {
  mockGet((path) => Promise.resolve(okResponses(path)));
  await render(View);

  await expect.element(page.getByText('scheduled tasks')).toBeInTheDocument();
  await expect.element(page.getByText('morning-report')).toBeInTheDocument();
  await expect.element(page.getByText('(unattributed)')).toBeInTheDocument();
  // last_run is formatted from the ISO timestamp (UTC, tz-safe).
  await expect.element(page.getByText('jul 09 08:00')).toBeInTheDocument();
});

test('surfaces the cache split as per-day cache columns', async () => {
  mockGet((path) => Promise.resolve(okResponses(path)));
  await render(View);

  // "cache rd"/"cache wr" headers repeat across every usage table, so scope the
  // lookup to the per-day table to keep the assertion unambiguous.
  const perDay = page.getByRole('table', { name: 'Usage per day' });
  await expect.element(perDay.getByRole('columnheader', { name: 'cache rd' })).toBeInTheDocument();
  await expect.element(perDay.getByRole('columnheader', { name: 'cache wr' })).toBeInTheDocument();
  // jul 14 row: 50000 cache reads -> "50k", 12000 cache writes -> "12k".
  await expect.element(page.getByText('50k')).toBeInTheDocument();
  await expect.element(page.getByText('12k')).toBeInTheDocument();
});

test('top agents table carries the cache split (payload already SUMs it per agent)', async () => {
  mockGet((path) => Promise.resolve(okResponses(path)));
  await render(View);

  const agentsTbl = page.getByRole('table', { name: 'Top agents by cost' });
  await expect
    .element(agentsTbl.getByRole('columnheader', { name: 'cache rd' }))
    .toBeInTheDocument();
  await expect
    .element(agentsTbl.getByRole('columnheader', { name: 'cache wr' }))
    .toBeInTheDocument();
  // odyn: 60000 reads -> "60k", 15000 writes -> "15k".
  await expect.element(agentsTbl.getByText('60k')).toBeInTheDocument();
  await expect.element(agentsTbl.getByText('15k')).toBeInTheDocument();
});

test('top models table carries the cache split', async () => {
  mockGet((path) => Promise.resolve(okResponses(path)));
  await render(View);

  const modelsTbl = page.getByRole('table', { name: 'Top models by cost' });
  await expect
    .element(modelsTbl.getByRole('columnheader', { name: 'cache rd' }))
    .toBeInTheDocument();
  await expect
    .element(modelsTbl.getByRole('columnheader', { name: 'cache wr' }))
    .toBeInTheDocument();
});

test('scheduled-tasks table carries the cache split', async () => {
  mockGet((path) => Promise.resolve(okResponses(path)));
  await render(View);

  const schedTbl = page.getByRole('table', { name: 'Usage by scheduled task' });
  await expect
    .element(schedTbl.getByRole('columnheader', { name: 'cache rd' }))
    .toBeInTheDocument();
  await expect
    .element(schedTbl.getByRole('columnheader', { name: 'cache wr' }))
    .toBeInTheDocument();
  // morning-report: 9000 reads -> "9k", 1000 writes -> "1k".
  await expect.element(schedTbl.getByText('9k')).toBeInTheDocument();
  await expect.element(schedTbl.getByText('1k')).toBeInTheDocument();
});

test('hides the scheduled-tasks table when no schedules ran', async () => {
  mockGet((path) =>
    Promise.resolve(path.startsWith('/api/usage/schedules') ? [] : okResponses(path)),
  );
  await render(View);

  await expect.element(page.getByText('$4.20')).toBeInTheDocument();
  expect(page.getByText('scheduled tasks').query()).toBeNull();
});

test('switching the range Seg reloads with the new window', async () => {
  const spy = mockGet((path) => Promise.resolve(okResponses(path)));
  await render(View);
  await expect.element(page.getByText('$4.20')).toBeInTheDocument();
  expect(usage.range.sinceDays).toBe(30);

  spy.mockClear();
  await page.getByRole('button', { name: '90 days' }).click();

  await expect.poll(() => usage.range.sinceDays).toBe(90);
  await expect
    .poll(() => spy.mock.calls.some((c) => (c[0] as string).startsWith('/api/usage/total')))
    .toBe(true);
  const totalCall = spy.mock.calls.find((c) => (c[0] as string).startsWith('/api/usage/total'));
  const since = new URLSearchParams((totalCall![0] as string).split('?')[1]).get('since');
  expect(since).toMatch(/^\d{4}-\d{2}-\d{2}$/);
});

test('shows the error pane with a working retry', async () => {
  const spy = mockGet(() => Promise.reject(new Error('network down')));
  await render(View);

  await expect.element(page.getByText("Couldn't load usage")).toBeInTheDocument();
  await expect.element(page.getByText('network down')).toBeInTheDocument();

  spy.mockImplementation((path: string) => Promise.resolve(okResponses(path)) as never);
  await page.getByRole('button', { name: 'Retry' }).click();

  await expect.element(page.getByText('$4.20')).toBeInTheDocument();
});

test('shows the empty state when nothing was recorded in range', async () => {
  mockGet((path) =>
    Promise.resolve(
      path.startsWith('/api/usage/total')
        ? { runs: 0, total_tokens: 0, total_cost: 0, input_tokens: 0, output_tokens: 0 }
        : [],
    ),
  );
  await render(View);

  await expect.element(page.getByText('No usage recorded yet')).toBeInTheDocument();
});

test('a null cost_usd sum (unset on some usage rows) renders as $0.00 instead of crashing', async () => {
  mockGet((path) => {
    if (path.startsWith('/api/usage/agents')) {
      return Promise.resolve([
        { agent: 'no-cost-agent', runs: 2, total_tokens: 500, total_cost: null },
      ]);
    }
    return Promise.resolve(okResponses(path));
  });
  await render(View);

  await expect.element(page.getByText('no-cost-agent')).toBeInTheDocument();
  await expect.element(page.getByText('$0.00')).toBeInTheDocument();
});
