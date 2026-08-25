/// <reference types="@vitest/browser/context" />
import { page } from '@vitest/browser/context';
import { render, cleanup } from 'vitest-browser-svelte';
import { afterEach, beforeEach, expect, test, vi } from 'vitest';
import View from './View.svelte';
import { webhooks, type Webhook } from '$lib/stores/webhooks.svelte';
import { agentsMeta, type RuntimeInfo } from '$lib/stores/agentsMeta.svelte';

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

function agent(name: string): RuntimeInfo {
  return {
    agent_file: name,
    workspace_dir: `/tmp/${name}`,
    model: null,
    context_limit: null,
    running_tasks: 0,
  };
}

function hook(overrides: Partial<Webhook> = {}): Webhook {
  return {
    token: 'tok-abc123',
    source: 'inbox-forward',
    created_at: new Date().toISOString(),
    ...overrides,
  };
}

// The view calls both stores' .load() from onMount; stub them so tests control
// state directly instead of racing a real fetch.
beforeEach(() => {
  webhooks.list = [];
  webhooks.loading = false;
  webhooks.error = null;
  agentsMeta.runtime = null;
  vi.spyOn(webhooks, 'load').mockResolvedValue(undefined);
  vi.spyOn(agentsMeta, 'load').mockResolvedValue(undefined);
});

test('shows the empty state, with the single New webhook action in the header', async () => {
  agentsMeta.runtime = {
    agent_file: 'smoke',
    workspace_dir: '/ws',
    model: null,
    context_limit: null,
    running_tasks: 0,
  };
  await render(View, {});
  await expect.element(page.getByText('No webhooks yet')).toBeVisible();
  await expect.element(page.getByRole('button', { name: 'New webhook' })).toBeEnabled();
});

test('shows a retryable error pane when the initial load fails and nothing is cached', async () => {
  webhooks.error = 'network unreachable';
  await render(View, {});
  await expect.element(page.getByText("Couldn't load webhooks")).toBeVisible();
  await expect.element(page.getByText('network unreachable')).toBeVisible();
  const retry = page.getByRole('button', { name: 'Retry' });
  await retry.click();
  expect(webhooks.load).toHaveBeenCalled();
});

test('renders configured webhooks with the delivery URL masked by default', async () => {
  webhooks.list = [hook({ token: 'secret-token-1', source: 'inbox-forward' })];
  await render(View, {});
  await expect.element(page.getByText('inbox-forward')).toBeVisible();
  await expect.element(page.getByText('/webhook/secret-token-1')).not.toBeInTheDocument();

  await page.getByRole('button', { name: 'show' }).click();
  await expect.element(page.getByText('/webhook/secret-token-1')).toBeVisible();

  await page.getByRole('button', { name: 'hide' }).click();
  await expect.element(page.getByText('/webhook/secret-token-1')).not.toBeInTheDocument();
});

test('a webhook never tested this session shows a neutral hint, not a fabricated status', async () => {
  webhooks.list = [hook()];
  await render(View, {});
  await expect.element(page.getByText('not tested this session')).toBeVisible();
});

test('create: validates the source client-side before calling the store', async () => {
  agentsMeta.runtime = {
    agent_file: 'smoke',
    workspace_dir: '/ws',
    model: null,
    context_limit: null,
    running_tasks: 0,
  };
  const createSpy = vi.spyOn(webhooks, 'create');
  await render(View, {});

  await page.getByRole('button', { name: 'New webhook' }).click();
  const dialog = page.getByRole('dialog', { name: 'New webhook' });
  await expect.element(dialog).toBeVisible();

  await dialog.getByLabelText('source').fill('bad source!');
  await dialog.getByRole('button', { name: 'Create' }).click();

  await expect
    .element(
      page.getByText('Source must be 1-64 chars: letters, digits, dot, underscore, or dash.'),
    )
    .toBeVisible();
  expect(createSpy).not.toHaveBeenCalled();
});

test('create: a valid submit calls the store with the source, then closes', async () => {
  agentsMeta.runtime = {
    agent_file: 'smoke',
    workspace_dir: '/ws',
    model: null,
    context_limit: null,
    running_tasks: 0,
  };
  vi.spyOn(webhooks, 'create').mockResolvedValue(hook({ source: 'gh-events' }));
  await render(View, {});

  await page.getByRole('button', { name: 'New webhook' }).click();
  const dialog = page.getByRole('dialog', { name: 'New webhook' });
  await dialog.getByLabelText('source').fill('gh-events');
  await dialog.getByRole('button', { name: 'Create' }).click();

  await expect.poll(() => webhooks.create).toHaveProperty('mock.calls.length', 1);
  expect(webhooks.create).toHaveBeenCalledWith({ source: 'gh-events' });
  // display:none removes the dialog from the a11y tree entirely once closed -
  // a stronger guarantee than "not visible" (see Conn.svelte.test.ts).
  await expect.element(dialog).not.toBeInTheDocument();
});

test('create: a store rejection surfaces inline and leaves the modal open', async () => {
  agentsMeta.runtime = {
    agent_file: 'smoke',
    workspace_dir: '/ws',
    model: null,
    context_limit: null,
    running_tasks: 0,
  };
  vi.spyOn(webhooks, 'create').mockRejectedValue(new Error('source already exists'));
  await render(View, {});

  await page.getByRole('button', { name: 'New webhook' }).click();
  const dialog = page.getByRole('dialog', { name: 'New webhook' });
  await dialog.getByLabelText('source').fill('gh-events');
  await dialog.getByRole('button', { name: 'Create' }).click();

  await expect.element(page.getByText('source already exists')).toBeVisible();
  await expect.element(dialog).toBeVisible();
});

test('delete: requires confirmation and then calls remove with the token', async () => {
  webhooks.list = [hook({ token: 'secret-token-1', source: 'inbox-forward' })];
  vi.spyOn(webhooks, 'remove').mockResolvedValue(undefined);
  await render(View, {});

  await page.getByRole('button', { name: 'Delete webhook inbox-forward' }).click();
  const dialog = page.getByRole('dialog', { name: 'Delete webhook?' });
  await expect.element(dialog).toBeVisible();
  await expect.element(dialog.getByText('inbox-forward')).toBeVisible();

  await dialog.getByRole('button', { name: 'Delete webhook' }).click();
  await expect.poll(() => webhooks.remove).toHaveProperty('mock.calls.length', 1);
  expect(webhooks.remove).toHaveBeenCalledWith('secret-token-1');
});

test('delete: cancel leaves the webhook in place', async () => {
  webhooks.list = [hook({ token: 'secret-token-1', source: 'inbox-forward' })];
  const removeSpy = vi.spyOn(webhooks, 'remove');
  await render(View, {});

  await page.getByRole('button', { name: 'Delete webhook inbox-forward' }).click();
  const dialog = page.getByRole('dialog', { name: 'Delete webhook?' });
  await dialog.getByRole('button', { name: 'Cancel' }).click();

  await expect.element(dialog).not.toBeInTheDocument();
  expect(removeSpy).not.toHaveBeenCalled();
});

test('test fire: a real POST to the public delivery path records an ok result and logs it', async () => {
  webhooks.list = [hook({ token: 'secret-token-1', source: 'inbox-forward' })];
  const fetchMock = vi.fn(async (input: RequestInfo | URL, _init?: RequestInit) => {
    expect(String(input)).toBe('/webhook/secret-token-1');
    return new Response(JSON.stringify({ status: 'accepted', file: '2026-x.json' }), {
      status: 202,
      headers: { 'Content-Type': 'application/json' },
    });
  });
  vi.stubGlobal('fetch', fetchMock);
  await render(View, {});

  await page.getByRole('button', { name: 'test fire' }).click();

  // "202" and the filename also appear in the toast and the log line, so scope
  // each assertion to the specific region being verified.
  await expect.element(page.getByRole('table').getByText('202')).toBeVisible();
  await expect.element(page.getByRole('log').getByText(/saved as 2026-x\.json/)).toBeVisible();
  expect(fetchMock).toHaveBeenCalledTimes(1);
  const [, init] = fetchMock.mock.calls[0]!;
  expect(init?.method).toBe('POST');
  const body = JSON.parse(init?.body as string);
  expect(body).toMatchObject({ event: 'test', source: 'inbox-forward' });
});

test('test fire: a failed delivery records an error result without crashing the row', async () => {
  webhooks.list = [hook({ token: 'secret-token-1', source: 'inbox-forward' })];
  vi.stubGlobal(
    'fetch',
    vi.fn(
      async () =>
        new Response(JSON.stringify({ error: 'webhook agent not configured' }), { status: 500 }),
    ),
  );
  await render(View, {});

  await page.getByRole('button', { name: 'test fire' }).click();

  await expect.element(page.getByRole('table').getByText('500')).toBeVisible();
  await expect
    .element(page.getByRole('log').getByText(/webhook agent not configured/))
    .toBeVisible();
});
