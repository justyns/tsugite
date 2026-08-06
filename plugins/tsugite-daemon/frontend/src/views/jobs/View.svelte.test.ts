/// <reference types="@vitest/browser/context" />
import { page } from '@vitest/browser/context';
import { render } from 'vitest-browser-svelte';
import { afterEach, beforeEach, expect, test, vi } from 'vitest';

// The board fires incidental loads (jobs, executors, agents) on mount; serve
// them from the seeded store instead of letting them fail into the error state.
vi.mock('$lib/api/client', () => ({
  authHeaders: () => ({}),
  api: {
    get: vi.fn(async (path: string) => {
      if (path.startsWith('/api/jobs')) return { jobs: SEED };
      if (path.startsWith('/api/executors')) return { executors: ['agent'] };
      return { agents: [] };
    }),
    post: vi.fn(async () => ({})),
    patch: vi.fn(async () => ({})),
    put: vi.fn(async () => ({})),
    del: vi.fn(async () => ({})),
  },
}));

import View from './View.svelte';
import { jobs, type Job } from '$lib/stores/jobs.svelte';
import { router } from '$lib/router.svelte';
import { TESTID } from '$lib/testids';

function job(job_id: string, parent_session_id: string | null, prompt: string): Job {
  return {
    job_id,
    parent_session_id,
    worker_session_id: null,
    verifier_session_id: null,
    state: 'running',
    prompt,
    verify_attempts: 0,
    max_attempts: 3,
    notify_when: null,
    error: null,
    error_detail: null,
    pending_question: null,
    attempts: [],
    acceptance_criteria: [],
    ac_results: [],
    result: null,
    agent: 'coder',
    model: null,
    effort: null,
    model_ladder: null,
    ladder_index: null,
    verifier_model: null,
    repo: null,
    created_at: '2026-08-01T10:00:00Z',
    updated_at: '2026-08-01T10:05:00Z',
    resolved_at: null,
    spawned_by: 'user',
    executor: 'agent',
    worker_terminal_id: null,
  } as Job;
}

const SEED: Job[] = [
  job('job-1', 'sess-a', 'from this chat'),
  job('job-2', 'sess-b', 'from another chat'),
  job('job-3', 'sess-a', 'also from this chat'),
];

beforeEach(() => {
  jobs.jobs = SEED;
  router.view = 'jobs';
  router.params = {};
});

afterEach(() => {
  jobs.jobs = [];
  router.params = {};
  location.hash = '';
});

test('a session filter in the route shows only that chat’s jobs, visibly', async () => {
  // Arriving from a chat's jobs chip: #jobs?q=session:sess-a.
  await page.viewport(1280, 800);
  router.params = { q: 'session:sess-a' };
  render(View);
  await expect.element(page.getByTestId(TESTID.jobCard('job-1'))).toBeInTheDocument();
  await expect.element(page.getByTestId(TESTID.jobCard('job-3'))).toBeInTheDocument();
  await page.screenshot({ path: '__screenshots__/jobs-session-filter.png' });
  await expect.element(page.getByTestId(TESTID.jobCard('job-2'))).not.toBeInTheDocument();
  // Visible and removable: the filter sits in the board's own search field.
  const search = document.querySelector(
    `[data-testid="${TESTID.jobsSearch}"] input`,
  ) as HTMLInputElement;
  expect(search.value).toBe('session:sess-a');
});

test('clearing the filter drops it from the route so a reload does not restore it', async () => {
  // Driven through the search field the way a user clears it, so the test pins
  // the behaviour rather than whichever binding currently backs the box.
  router.params = { q: 'session:sess-a' };
  location.hash = '#jobs?q=session%3Asess-a';
  render(View);
  const search = page.getByRole('searchbox', { name: 'Search jobs' });
  await expect.poll(() => (search.element() as HTMLInputElement).value).toBe('session:sess-a');

  await search.fill('');
  await expect.element(page.getByTestId(TESTID.jobCard('job-2'))).toBeInTheDocument();
  await expect.poll(() => location.hash).toBe('#jobs');
});

test('typing a filter puts it in the route, so the view is linkable and reload-safe', async () => {
  render(View);
  const search = page.getByRole('searchbox', { name: 'Search jobs' });
  await search.fill('session:sess-b');
  await expect.element(page.getByTestId(TESTID.jobCard('job-2'))).toBeInTheDocument();
  await expect.element(page.getByTestId(TESTID.jobCard('job-1'))).not.toBeInTheDocument();
  await expect.poll(() => location.hash).toBe('#jobs?q=session%3Asess-b');
});
