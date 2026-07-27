/// <reference types="vitest/browser" />
import { page } from 'vitest/browser';
import { render } from 'vitest-browser-svelte';
import { afterEach, beforeEach, expect, test, vi } from 'vitest';
import View from './View.svelte';
import { tools, type ToolInfo } from '$lib/stores/tools.svelte';
import { api } from '$lib/api/client';

const registry: ToolInfo[] = [
  { name: 'exec', category: 'run', description: 'Run a shell command.', source: 'builtin' },
  { name: 'web.fetch', category: 'web', description: 'Fetch a URL over HTTP.', source: 'plugin' },
];

// `tools` is a module-level singleton (shared across every test in this file),
// so each test starts it from a known state rather than trusting leftovers
// from whichever test ran before it.
beforeEach(() => {
  tools.tools = [];
  tools.available = true;
  tools.loading = false;
  tools.error = null;
});

afterEach(() => {
  vi.restoreAllMocks();
});

function mockGet(impl: () => Promise<unknown>) {
  return vi.spyOn(api, 'get').mockImplementation(impl as never);
}

test('renders every registered tool with its category chip, description and source', async () => {
  mockGet(() => Promise.resolve({ tools: registry }));
  await render(View);

  const table = page.getByRole('table', { name: 'Tools' });
  await expect.element(table.getByText('exec')).toBeInTheDocument();
  await expect.element(table.getByText('web.fetch')).toBeInTheDocument();
  // exact: substring matching would also hit "Run a shell command." below
  await expect.element(table.getByText('run', { exact: true })).toBeInTheDocument();
  await expect.element(table.getByText('Fetch a URL over HTTP.')).toBeInTheDocument();
  await expect.element(table.getByText('builtin')).toBeInTheDocument();
  await expect.element(table.getByText('plugin')).toBeInTheDocument();
});

test('typing in the search box filters rows by name/category/description/source', async () => {
  mockGet(() => Promise.resolve({ tools: registry }));
  await render(View);

  const table = page.getByRole('table', { name: 'Tools' });
  await expect.element(table.getByText('exec')).toBeInTheDocument();

  await page.getByLabelText('Search tools').fill('web');

  await expect.element(table.getByText('web.fetch')).toBeInTheDocument();
  await expect.element(table.getByText('exec')).not.toBeInTheDocument();
});

test('a search with no matches shows the empty state with a way to clear it', async () => {
  mockGet(() => Promise.resolve({ tools: registry }));
  await render(View);
  await expect.element(page.getByRole('table', { name: 'Tools' })).toBeInTheDocument();

  await page.getByLabelText('Search tools').fill('nonexistent-xyz');

  await expect.element(page.getByText('No matching tools')).toBeInTheDocument();
  await page.getByRole('button', { name: 'Clear search' }).click();

  await expect.element(page.getByRole('table', { name: 'Tools' })).toBeInTheDocument();
  await expect
    .element(page.getByRole('table', { name: 'Tools' }).getByText('exec'))
    .toBeInTheDocument();
});

test('a 404 renders the plain empty state, not an error', async () => {
  mockGet(() => Promise.reject(Object.assign(new Error('not found'), { status: 404 })));
  await render(View);

  await expect.element(page.getByText('No tools registered')).toBeInTheDocument();
  await expect.element(page.getByText("Couldn't load tools")).not.toBeInTheDocument();
});

test('a real fetch failure renders the error state, and retry recovers', async () => {
  const spy = mockGet(() => Promise.reject(new Error('network boom')));
  await render(View);

  await expect.element(page.getByText("Couldn't load tools")).toBeInTheDocument();
  await expect.element(page.getByText('network boom')).toBeInTheDocument();

  spy.mockImplementation(() => Promise.resolve({ tools: registry }) as never);
  await page.getByRole('button', { name: 'Retry' }).click();

  await expect
    .element(page.getByRole('table', { name: 'Tools' }).getByText('exec'))
    .toBeInTheDocument();
});

test('an empty registry renders a truthful empty state, not a blank table', async () => {
  mockGet(() => Promise.resolve({ tools: [] }));
  await render(View);

  await expect.element(page.getByText('No tools registered')).toBeInTheDocument();
});

test('the search box is disabled while there is nothing to search', async () => {
  mockGet(() => Promise.resolve({ tools: [] }));
  await render(View);

  await expect.element(page.getByText('No tools registered')).toBeInTheDocument();
  await expect.element(page.getByLabelText('Search tools')).toBeDisabled();
});
