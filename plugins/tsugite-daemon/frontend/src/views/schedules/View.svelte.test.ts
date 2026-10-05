/// <reference types="@vitest/browser/context" />
import { page } from '@vitest/browser/context';
import { render } from 'vitest-browser-svelte';
import { afterEach, expect, test, vi } from 'vitest';

const SEED = [
  {
    id: 'dev-land',
    agent: 'assistant',
    prompt: 'land it',
    schedule_type: 'cron',
    cron_expr: '*/10 * * * *',
    run_at: null,
    enabled: true,
    created_at: '2026-07-17T00:00:00Z',
    last_run: null,
    next_run: null,
    last_status: null,
    last_error: null,
    timezone: 'UTC',
    execution_type: 'agent',
    command: null,
    run_count: 0,
    disabled_reason: null,
  },
];

vi.mock('$lib/api/client', () => ({
  authHeaders: () => ({}),
  api: {
    get: vi.fn(async (path: string) => {
      if (path.startsWith('/api/schedules/dev-land/sessions')) return { sessions: [] };
      if (path.startsWith('/api/schedules')) return { schedules: SEED };
      return { files: [] };
    }),
    post: vi.fn(async () => ({})),
    patch: vi.fn(async () => ({})),
    put: vi.fn(async () => ({})),
    del: vi.fn(async () => ({})),
  },
}));

import View from './View.svelte';
import { router } from '$lib/router.svelte';
import { TESTID } from '$lib/testids';

afterEach(() => {
  router.view = '';
  router.params = {};
});

test("an id route param opens that schedule's drawer", async () => {
  router.view = 'schedules';
  router.params = { id: 'dev-land' };
  const { container } = await render(View);
  await expect.element(page.getByTestId(TESTID.scheduleForm)).toBeInTheDocument();
  expect(container.querySelector('.t-drawer.is-open')).not.toBeNull();
});

test('without an id param the drawer stays closed', async () => {
  router.view = 'schedules';
  const { container } = await render(View);
  await expect.element(page.getByTestId(TESTID.scheduleRow('dev-land'))).toBeInTheDocument();
  expect(container.querySelector('.t-drawer.is-open')).toBeNull();
});
