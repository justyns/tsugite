/// <reference types="@vitest/browser/context" />
import { page } from '@vitest/browser/context';
import { render, cleanup } from 'vitest-browser-svelte';
import { afterEach, beforeEach, expect, test, vi } from 'vitest';

vi.mock('$lib/api/client', () => ({
  api: { get: vi.fn(), patch: vi.fn() },
  authHeaders: () => ({}),
}));

import { api } from '$lib/api/client';
import { sessions } from '$lib/stores/sessions.svelte';
import ModelEffort from './ModelEffort.svelte';

function mockBackend(opts: { levels: string[] | null; effort?: string | null }) {
  vi.mocked(api.get).mockImplementation((path: string) => {
    if (path.endsWith('/settings'))
      return Promise.resolve({
        model: null,
        reasoning_effort: opts.effort ?? null,
      });
    if (path.includes('/effort-levels'))
      return Promise.resolve({
        model: 'anthropic:claude-sonnet-4-5',
        supported_effort_levels: opts.levels,
      });
    if (path === '/api/models') return Promise.resolve({ models: [] });
    return Promise.reject(new Error(`unexpected GET ${path}`));
  });
  vi.mocked(api.patch).mockImplementation(() =>
    Promise.resolve({ model: null, reasoning_effort: 'low' }),
  );
}

afterEach(cleanup);
beforeEach(() => {
  vi.clearAllMocks();
});

test('names the resolved default model on the chip', async () => {
  mockBackend({ levels: ['low', 'medium', 'high'] });
  render(ModelEffort, { sessionId: 's1' });
  await expect
    .element(page.getByTestId('chat-model-trigger'))
    .toHaveTextContent('default · claude-sonnet-4-5');
});

test('renders the model-specific effort levels and persists a pick', async () => {
  mockBackend({ levels: ['minimal', 'low', 'medium', 'high'], effort: 'high' });
  render(ModelEffort, { sessionId: 's1' });

  const seg = page.getByTestId('chat-effort-seg');
  await expect.element(seg.getByRole('button', { name: 'min' })).toBeInTheDocument();
  await expect
    .element(seg.getByRole('button', { name: 'high' }))
    .toHaveAttribute('aria-pressed', 'true');

  await seg.getByRole('button', { name: 'low' }).click();
  expect(vi.mocked(api.patch)).toHaveBeenCalledWith('/api/sessions/s1/settings', {
    reasoning_effort: 'low',
  });
});

test('a settings broadcast refetches the effort seg live', async () => {
  let effort: string | null = 'high';
  vi.mocked(api.get).mockImplementation((path: string) => {
    if (path.endsWith('/settings'))
      return Promise.resolve({ model: null, reasoning_effort: effort });
    if (path.includes('/effort-levels'))
      return Promise.resolve({
        model: 'anthropic:claude-sonnet-4-5',
        supported_effort_levels: ['low', 'medium', 'high'],
      });
    if (path === '/api/models') return Promise.resolve({ models: [] });
    return Promise.reject(new Error(`unexpected GET ${path}`));
  });
  render(ModelEffort, { sessionId: 's-eff' });
  const seg = page.getByTestId('chat-effort-seg');
  await expect
    .element(seg.getByRole('button', { name: 'high' }))
    .toHaveAttribute('aria-pressed', 'true');

  // A /effort change elsewhere bumps settingsRev - the seg reflects it live.
  effort = 'low';
  sessions.applySessionUpdate({
    action: 'settings',
    id: 's-eff',
    model: null,
    reasoning_effort: 'low',
  });
  await expect
    .element(seg.getByRole('button', { name: 'low' }))
    .toHaveAttribute('aria-pressed', 'true');
});

test('a model without effort levels gets no effort seg', async () => {
  mockBackend({ levels: null });
  render(ModelEffort, { sessionId: 's1' });
  await expect.element(page.getByTestId('chat-model-trigger')).toBeInTheDocument();
  await expect.element(page.getByTestId('chat-effort-seg')).not.toBeInTheDocument();
});
