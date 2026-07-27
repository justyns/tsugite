/// <reference types="@vitest/browser/context" />
import { page } from '@vitest/browser/context';
import { render, cleanup } from 'vitest-browser-svelte';
import { afterEach, beforeEach, expect, test, vi } from 'vitest';

vi.mock('$lib/api/client', () => ({
  api: { get: vi.fn() },
  authHeaders: () => ({}),
}));

import { api } from '$lib/api/client';
import { pluginsMeta } from '$lib/stores/pluginsMeta.svelte';
import View from './View.svelte';

afterEach(cleanup);

beforeEach(() => {
  vi.mocked(api.get).mockReset();
  pluginsMeta.plugins = [];
  pluginsMeta.available = true;
  pluginsMeta.loading = false;
  pluginsMeta.error = null;
});

function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (err: unknown) => void;
  const promise = new Promise<T>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

function notFound(): Error & { status: number } {
  return Object.assign(new Error('not found'), { status: 404 });
}

test('shows a loading pane while the initial fetch is in flight', async () => {
  const gate = deferred<{ plugins: never[] }>();
  vi.mocked(api.get).mockReturnValue(gate.promise);
  const { container } = await render(View);
  await expect.element(page.getByText('Plugins')).toBeInTheDocument();
  expect(container.querySelector('.t-skel')).not.toBeNull();
  gate.resolve({ plugins: [] });
});

test('renders a truthful empty state when the daemon has zero plugins installed', async () => {
  vi.mocked(api.get).mockResolvedValue({ plugins: [] });
  render(View);
  await expect.element(page.getByText('No plugins installed')).toBeInTheDocument();
});

test('a 404 renders the plain empty state without faking rows', async () => {
  vi.mocked(api.get).mockRejectedValue(notFound());
  render(View);
  await expect.element(page.getByText('No plugins installed')).toBeInTheDocument();
  expect(page.getByRole('table').elements()).toHaveLength(0);
});

test('renders an error pane with retry on a non-404 failure, and retry re-fetches', async () => {
  vi.mocked(api.get).mockRejectedValueOnce(new Error('network down'));
  render(View);
  await expect.element(page.getByText('network down')).toBeInTheDocument();

  vi.mocked(api.get).mockResolvedValueOnce({
    plugins: [{ name: 'pty', group: 'tsugite.plugins', enabled: true, loaded: false, error: null }],
  });
  await page.getByRole('button', { name: /retry/i }).click();
  await expect.element(page.getByText('pty')).toBeInTheDocument();
  expect(api.get).toHaveBeenCalledTimes(2);
});

test('renders name, stripped group, and enabled/loaded pills per plugin, sorted by group then name', async () => {
  vi.mocked(api.get).mockResolvedValue({
    plugins: [
      { name: 'web', group: 'tsugite.plugins', enabled: true, loaded: false, error: null },
      { name: 'youtube', group: 'tsugite.attachments', enabled: false, loaded: false, error: null },
      { name: 'pty', group: 'tsugite.plugins', enabled: true, loaded: true, error: null },
    ],
  });
  const { container } = await render(View);
  await expect.element(page.getByText('pty')).toBeInTheDocument();

  const rows = container.querySelectorAll('tbody tr');
  expect(rows).toHaveLength(3);
  // sorted: tsugite.attachments:youtube, tsugite.plugins:pty, tsugite.plugins:web
  expect(rows[0]?.textContent).toContain('youtube');
  expect(rows[0]?.textContent).toContain('attachments');
  expect(rows[1]?.textContent).toContain('pty');
  expect(rows[2]?.textContent).toContain('web');

  // pty: enabled + loaded
  expect(rows[1]?.textContent).toContain('enabled');
  expect(rows[1]?.textContent).toContain('loaded');
  // youtube: disabled, not loaded
  expect(rows[0]?.textContent).toContain('disabled');
  expect(rows[0]?.textContent).toContain('not loaded');
});

test('never renders an enable/disable toggle (not wired server-side)', async () => {
  vi.mocked(api.get).mockResolvedValue({
    plugins: [{ name: 'pty', group: 'tsugite.plugins', enabled: true, loaded: false, error: null }],
  });
  const { container } = await render(View);
  await expect.element(page.getByText('pty')).toBeInTheDocument();
  expect(container.querySelector('[role="switch"]')).toBeNull();
  expect(page.getByRole('checkbox').elements()).toHaveLength(0);
});

test('renders an error callout only for plugins that actually errored', async () => {
  vi.mocked(api.get).mockResolvedValue({
    plugins: [
      { name: 'ok-one', group: 'tsugite.tools', enabled: true, loaded: true, error: null },
      {
        name: 'broken',
        group: 'tsugite.tools',
        enabled: true,
        loaded: false,
        error: 'ImportError: no module named foo',
      },
    ],
  });
  render(View);
  await expect.element(page.getByText('ImportError: no module named foo')).toBeInTheDocument();
  const okRow = page.getByRole('row', { name: /ok-one/ });
  await expect.element(okRow).toBeInTheDocument();
});
