/// <reference types="@vitest/browser/context" />
import { page, userEvent } from '@vitest/browser/context';
import { render, cleanup } from 'vitest-browser-svelte';
import { afterEach, beforeEach, expect, test, vi } from 'vitest';
import { toasts } from '$lib/components/feedback/toast-store.svelte';
import { TESTID } from '$lib/testids';
import ProvidersSettings from './ProvidersSettings.svelte';

vi.mock('$lib/api/client', () => ({
  api: { get: vi.fn(), post: vi.fn(), put: vi.fn(), del: vi.fn() },
}));

import { api } from '$lib/api/client';

const claude = {
  name: 'claude_code',
  binary: { default: 'claude', configured: null, resolved: null },
  key: null,
};
const openai = {
  name: 'openai',
  binary: null,
  key: { env: 'OPENAI_API_KEY', secret: 'openai-api-key', set: false, source: null },
};
const anthropic = {
  name: 'anthropic',
  binary: null,
  key: { env: 'ANTHROPIC_API_KEY', secret: 'anthropic-api-key', set: true, source: 'env' },
};

const row = (name: string) => page.getByTestId(TESTID.settingsProvider(name));

afterEach(cleanup);

beforeEach(() => {
  toasts.items = [];
  vi.mocked(api.get)
    .mockReset()
    .mockResolvedValue({ providers: [claude, openai, anthropic] });
  vi.mocked(api.post).mockReset();
  vi.mocked(api.put).mockReset();
  vi.mocked(api.del).mockReset();
});

test('shows binary and key status, with Clear only for a stored key', async () => {
  await render(ProvidersSettings);
  await expect
    .element(row('claude_code').getByText('not found', { exact: true }))
    .toBeInTheDocument();
  await expect
    .element(row('claude_code').getByText('Not found: claude on PATH'))
    .toBeInTheDocument();
  await expect.element(row('openai').getByText('no key', { exact: true })).toBeInTheDocument();
  await expect.element(row('anthropic').getByText('key set', { exact: true })).toBeInTheDocument();
  await expect
    .element(row('anthropic').getByText(/Read from ANTHROPIC_API_KEY/))
    .toBeInTheDocument();
  expect(row('anthropic').getByRole('button', { name: 'Clear' }).query()).toBeNull();
});

test('committing a binary path saves it and shows the resolved path', async () => {
  vi.mocked(api.put).mockResolvedValue({
    ...claude,
    binary: { default: 'claude', configured: '/opt/claude', resolved: '/opt/claude' },
  });
  await render(ProvidersSettings);
  const input = page.getByLabelText('Binary path');
  await input.fill('/opt/claude');
  await userEvent.keyboard('{Tab}');

  expect(api.put).toHaveBeenCalledWith('/api/providers/claude_code', { binary: '/opt/claude' });
  await expect.element(row('claude_code').getByText('Runs /opt/claude')).toBeInTheDocument();
  await expect.element(row('claude_code').getByText('found', { exact: true })).toBeInTheDocument();
});

test('saving a key stores it as the provider secret and reloads the status', async () => {
  vi.mocked(api.post).mockResolvedValue({ status: 'ok' });
  await render(ProvidersSettings);
  vi.mocked(api.get).mockResolvedValue({
    providers: [
      claude,
      { ...openai, key: { ...openai.key, set: true, source: 'secret' } },
      anthropic,
    ],
  });

  await row('openai').getByLabelText('API key').fill('sk-new');
  await row('openai').getByRole('button', { name: 'Save' }).click();

  expect(api.post).toHaveBeenCalledWith('/api/secrets/openai-api-key', { value: 'sk-new' });
  await expect.element(row('openai').getByText('key set', { exact: true })).toBeInTheDocument();
  await expect.element(row('openai').getByRole('button', { name: 'Clear' })).toBeInTheDocument();
});

test('saving a key keeps unsaved drafts in the other rows', async () => {
  vi.mocked(api.post).mockResolvedValue({ status: 'ok' });
  await render(ProvidersSettings);

  await row('anthropic').getByLabelText('Replace API key').fill('sk-draft');
  await row('openai').getByLabelText('API key').fill('sk-new');
  await row('openai').getByRole('button', { name: 'Save' }).click();

  await vi.waitFor(() => expect(api.get).toHaveBeenCalledTimes(2));
  await expect.element(row('openai').getByLabelText('API key')).toHaveValue('');
  await expect.element(row('anthropic').getByLabelText('Replace API key')).toHaveValue('sk-draft');
});

test('a backend that cannot store keys reports the failure', async () => {
  vi.mocked(api.post).mockRejectedValue(new Error('backend does not support writing'));
  await render(ProvidersSettings);
  await row('openai').getByLabelText('API key').fill('sk-new');
  await row('openai').getByRole('button', { name: 'Save' }).click();

  await vi.waitFor(() =>
    expect(toasts.items.map((t) => [t.title, t.body])).toContainEqual([
      'Key not saved',
      'backend does not support writing',
    ]),
  );
});

test('Test shows the error text from a failed request', async () => {
  vi.mocked(api.post).mockResolvedValue({
    ok: false,
    model: 'gpt-5-nano',
    error: '401 invalid api key',
  });
  await render(ProvidersSettings);
  await page.getByRole('button', { name: 'Test openai' }).click();

  expect(api.post).toHaveBeenCalledWith('/api/providers/openai/test', {});
  await expect
    .element(row('openai').getByRole('status'))
    .toHaveTextContent('gpt-5-nano: 401 invalid api key');
});
