/// <reference types="@vitest/browser/context" />
import { page, userEvent } from '@vitest/browser/context';
import { render, cleanup } from 'vitest-browser-svelte';
import { afterEach, beforeEach, expect, test, vi } from 'vitest';
import { TESTID } from '$lib/testids';
import { welcomeDismissed } from '$lib/stores/welcomeDismissed.svelte';
import { auth } from '$lib/stores/auth.svelte';
import Welcome from './Welcome.svelte';

vi.mock('$lib/api/client', () => ({
  api: { get: vi.fn(), post: vi.fn(), put: vi.fn(), patch: vi.fn(), del: vi.fn() },
}));

import { api } from '$lib/api/client';

const providers = [
  {
    name: 'openai',
    binary: null,
    key: { env: 'OPENAI_API_KEY', secret: 'openai-api-key', set: false, source: null },
  },
  {
    name: 'claude_code',
    binary: { default: 'claude', configured: null, resolved: '/usr/bin/claude' },
    key: null,
  },
];
const models = [
  { id: 'claude_code:opus', provider: 'claude_code' },
  { id: 'claude_code:sonnet', provider: 'claude_code' },
  { id: 'openai:gpt-4o', provider: 'openai' },
];
const agentList = [
  { id: 'helper', name: 'helper', color: 'green', model: null, is_default: false },
  { id: 'desktop', name: 'desktop', color: 'blue', model: null, is_default: true },
];

function serve(
  runtime: { model?: string | null; effective_model: string | null },
  { failModels = false } = {},
) {
  vi.mocked(api.get).mockImplementation(async (path: string) => {
    if (path === '/api/runtime') return { model: null, ...runtime };
    if (path === '/api/models') {
      if (failModels) throw new Error('daemon unreachable');
      return { models };
    }
    if (path === '/api/providers') return { providers };
    if (path === '/api/agents') return { agents: agentList };
    throw new Error(`unexpected GET ${path}`);
  });
}

const runtimeGets = () => vi.mocked(api.get).mock.calls.filter(([p]) => p === '/api/runtime');

afterEach(cleanup);

beforeEach(() => {
  welcomeDismissed.set(false);
  auth.setUserId('web-user-1');
  vi.mocked(api.get).mockReset();
  vi.mocked(api.patch).mockReset().mockResolvedValue({});
});

test('opens itself on a fresh daemon, loads the runtime once, and saves model and user ID', async () => {
  serve({ effective_model: null });
  await render(Welcome, {});

  const welcome = page.getByTestId(TESTID.welcome);
  await expect.element(welcome.getByLabelText('Provider')).toHaveValue('claude_code');
  await expect.element(welcome.getByLabelText('Default model')).toHaveValue('claude_code:opus');
  await expect
    .element(page.getByTestId(TESTID.settingsProvider('claude_code')).getByText('found'))
    .toBeInTheDocument();
  expect(page.getByTestId(TESTID.settingsProvider('openai')).query()).toBeNull();
  expect(runtimeGets()).toHaveLength(1);

  await expect.element(page.getByLabelText('User ID')).toHaveValue('web-user-1');
  await userEvent.fill(page.getByLabelText('User ID'), ' sam ');
  await page.getByRole('button', { name: 'Finish' }).click();

  await expect.element(welcome).not.toBeVisible();
  expect(api.patch).toHaveBeenCalledWith('/api/runtime/config', { model: 'claude_code:opus' });
  expect(auth.userId).toBe('sam');
  expect(localStorage.getItem('tsugite_user_id')).toBe('sam');
  expect(welcomeDismissed.enabled).toBe(true);
});

test('stays closed when a global or agent-file default model is in effect', async () => {
  serve({ model: null, effective_model: 'claude_code:sonnet' });
  await render(Welcome, {});

  await vi.waitFor(() => expect(runtimeGets()).toHaveLength(1));
  await new Promise((resolve) => setTimeout(resolve));
  await expect.element(page.getByTestId(TESTID.welcome)).not.toBeVisible();
});

test('stays closed and skips the check in a browser that dismissed it', async () => {
  welcomeDismissed.set(true);
  serve({ effective_model: null });
  await render(Welcome, {});

  await expect.element(page.getByTestId(TESTID.welcome)).not.toBeVisible();
  expect(api.get).not.toHaveBeenCalled();
});

test('says when it replaces the effective default model, and keeps an unchanged model', async () => {
  serve({ model: null, effective_model: 'claude_code:sonnet' });
  await render(Welcome, { open: true });

  const modelSelect = page.getByLabelText('Default model');
  await expect.element(modelSelect).toHaveValue('claude_code:sonnet');

  await modelSelect.selectOptions('claude_code:opus');
  await expect
    .element(page.getByText('Replaces the current default, claude_code:sonnet.'))
    .toBeInTheDocument();
  await modelSelect.selectOptions('claude_code:sonnet');

  await page.getByRole('button', { name: 'Finish' }).click();
  await expect.element(page.getByTestId(TESTID.welcome)).not.toBeVisible();
  expect(api.patch).not.toHaveBeenCalled();
});

test('Finish renames the default agent and sets its color and model', async () => {
  serve({ effective_model: 'claude_code:sonnet' });
  await render(Welcome, { open: true });

  const name = page.getByLabelText('Agent name');
  await expect.element(name).toHaveValue('desktop');
  await expect.element(page.getByLabelText('Color')).toHaveValue('blue');
  await userEvent.fill(name, ' Rook ');
  await page.getByLabelText('Color').selectOptions('purple');
  await userEvent.fill(page.getByLabelText('Agent model'), 'claude_code:opus');
  await page.getByRole('button', { name: 'Finish' }).click();

  await expect.element(page.getByTestId(TESTID.welcome)).not.toBeVisible();
  expect(api.patch).toHaveBeenCalledTimes(1);
  expect(api.patch).toHaveBeenCalledWith('/api/agents/desktop', {
    name: 'Rook',
    color: 'purple',
    model: 'claude_code:opus',
  });
});

test('Finish sends only the agent fields that changed', async () => {
  serve({ effective_model: 'claude_code:sonnet' });
  await render(Welcome, { open: true });

  await expect.element(page.getByLabelText('Agent name')).toHaveValue('desktop');
  await page.getByLabelText('Color').selectOptions('teal');
  await page.getByRole('button', { name: 'Finish' }).click();

  await expect.element(page.getByTestId(TESTID.welcome)).not.toBeVisible();
  expect(api.patch).toHaveBeenCalledWith('/api/agents/desktop', { color: 'teal' });
});

test('Finish stays disabled when the load fails', async () => {
  serve({ effective_model: null }, { failModels: true });
  await render(Welcome, { open: true });

  await vi.waitFor(() => expect(api.get).toHaveBeenCalledWith('/api/models'));
  await expect.element(page.getByRole('button', { name: 'Finish' })).toBeDisabled();
});

test('Skip records the dismissal in this browser without saving', async () => {
  serve({ effective_model: null });
  await render(Welcome, { open: true });

  await page.getByRole('button', { name: 'Skip' }).click();

  await expect.element(page.getByTestId(TESTID.welcome)).not.toBeVisible();
  expect(welcomeDismissed.enabled).toBe(true);
  expect(localStorage.getItem('tsugite_welcome_dismissed')).toBe('true');
  expect(api.patch).not.toHaveBeenCalled();
});

test('Esc records the dismissal like Skip', async () => {
  serve({ effective_model: null });
  await render(Welcome, { open: true });
  await page.getByLabelText('User ID').click();
  await userEvent.keyboard('{Escape}');

  await expect.element(page.getByTestId(TESTID.welcome)).not.toBeVisible();
  expect(welcomeDismissed.enabled).toBe(true);
  expect(api.patch).not.toHaveBeenCalled();
});
