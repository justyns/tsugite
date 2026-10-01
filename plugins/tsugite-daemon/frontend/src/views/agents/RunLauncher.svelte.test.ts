/// <reference types="@vitest/browser/context" />
import { page } from '@vitest/browser/context';
import { render, cleanup } from 'vitest-browser-svelte';
import { afterEach, beforeEach, expect, test, vi } from 'vitest';

vi.mock('$lib/api/client', () => ({
  api: { get: vi.fn(), post: vi.fn(), patch: vi.fn() },
  authHeaders: () => ({}),
}));

import { api } from '$lib/api/client';
import { agentsMeta } from '$lib/stores/agentsMeta.svelte';
import { sessions } from '$lib/stores/sessions.svelte';
import { toasts } from '$lib/components/feedback/toast-store.svelte';
import { composerPrefill } from '../chats/composerPrefill.svelte';
import RunLauncher from './RunLauncher.svelte';

const FILES = [
  {
    path: '/ws/default.md',
    name: 'default',
    source: 'builtin',
    readonly: true,
    description: '',
    chat: true,
    default: true,
  },
  {
    path: '/ws/hollis.md',
    name: 'hollis',
    source: 'project',
    readonly: false,
    description: 'Operator.',
    chat: true,
  },
  {
    path: '/ws/nightly.md',
    name: 'nightly',
    source: 'project',
    readonly: false,
    description: 'Batch.',
    chat: false,
  },
];
const MODELS = {
  models: [{ id: 'openai:gpt-5.4', provider: 'openai', context_window: 400_000 }],
};

const onOpen = vi.fn();
const onClose = vi.fn();

function mount(props: { initialAgent?: string | null } = {}) {
  return render(RunLauncher, { open: true, onOpen, onClose, ...props });
}

beforeEach(() => {
  vi.mocked(api.get).mockReset();
  vi.mocked(api.post).mockReset();
  vi.mocked(api.patch).mockReset();
  onOpen.mockReset();
  onClose.mockReset();
  toasts.items.length = 0;
  vi.mocked(api.get).mockImplementation(((path: string) => {
    if (path === '/api/agent-files') return Promise.resolve({ files: FILES });
    if (path === '/api/models') return Promise.resolve(MODELS);
    return Promise.reject(new Error('unexpected GET ' + path));
  }) as never);
  vi.mocked(api.post).mockImplementation((async (path: string) =>
    path === '/api/chat/sessions/new' ? { id: 'new-chat' } : { id: 'run-1' }) as never);
  vi.mocked(api.patch).mockResolvedValue({} as never);
  vi.spyOn(sessions, 'load').mockResolvedValue(undefined as never);
  agentsMeta.agentFiles = [];
});

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

const optionTexts = () =>
  [...document.querySelectorAll<HTMLOptionElement>('select[aria-label="Agent"] option')].map(
    (o) => o.value,
  );

test('chat mode lists only chat-capable agents with the default preselected', async () => {
  mount();
  await vi.waitFor(() => expect(optionTexts()).toEqual(['default', 'hollis']));
  const select = document.querySelector<HTMLSelectElement>('select[aria-label="Agent"]')!;
  expect(select.value).toBe('default');
});

test('left at defaults, Start chat creates a plain chat and applies no settings', async () => {
  mount();
  await vi.waitFor(() => expect(optionTexts()).toContain('hollis'));
  await page.getByTestId('launcher-start').click();

  await vi.waitFor(() => expect(onOpen).toHaveBeenCalledWith('new-chat'));
  expect(api.post).toHaveBeenCalledTimes(1);
  expect(api.post).toHaveBeenCalledWith('/api/chat/sessions/new', {});
  expect(api.patch).not.toHaveBeenCalled();
  expect(composerPrefill.pending).toBeNull();
});

test('a chosen agent, model and effort are sent with the create call', async () => {
  mount();
  await vi.waitFor(() => expect(optionTexts()).toContain('hollis'));
  await page.getByLabelText('Agent', { exact: true }).selectOptions('hollis');
  await page.getByTestId('chat-model-trigger').click();
  await page.getByTestId('chat-model-opt-openai:gpt-5.4').click();
  await page.getByRole('button', { name: 'high' }).click();
  await page.getByLabelText('Starting message').fill('  hello there ');
  await page.getByTestId('launcher-start').click();

  await vi.waitFor(() => expect(onOpen).toHaveBeenCalledWith('new-chat'));
  expect(api.post).toHaveBeenCalledWith('/api/chat/sessions/new', {
    agent_file: '/ws/hollis.md',
    model: 'openai:gpt-5.4',
    reasoning_effort: 'high',
  });
  expect(api.patch).not.toHaveBeenCalled();
  expect(composerPrefill.pending).toEqual({
    sessionId: 'new-chat',
    text: 'hello there',
    run: true,
  });
  composerPrefill.consume('new-chat');
});

test('run once lists every agent file and needs a prompt', async () => {
  mount();
  await page.getByRole('button', { name: 'run once' }).click();
  await vi.waitFor(() => expect(optionTexts()).toEqual(['default', 'hollis', 'nightly']));
  await expect.element(page.getByTestId('launcher-start')).toBeDisabled();

  await page.getByLabelText('Agent', { exact: true }).selectOptions('nightly');
  await page.getByLabelText('Run prompt').fill('rotate the logs');
  await expect.element(page.getByTestId('launcher-start')).toBeEnabled();
});

test('run once posts a background run, never creating a chat, and links to the session', async () => {
  mount({ initialAgent: '/ws/nightly.md' });
  await page.getByLabelText('Run prompt').fill('rotate the logs');
  await page.getByRole('button', { name: 'high' }).click();
  await page.getByTestId('launcher-start').click();

  await vi.waitFor(() => expect(onClose).toHaveBeenCalled());
  expect(api.post).toHaveBeenCalledTimes(1);
  expect(api.post).toHaveBeenCalledWith('/api/sessions', {
    prompt: 'rotate the logs',
    agent_file: '/ws/nightly.md',
    reasoning_effort: 'high',
  });
  expect(onOpen).not.toHaveBeenCalled();
  const toast = toasts.items.find((t) => t.title === 'Run started');
  toast?.onAction?.();
  expect(onOpen).toHaveBeenCalledWith('run-1');
});

test('opened on a chat-capable agent it starts in chat mode on that agent', async () => {
  mount({ initialAgent: '/ws/hollis.md' });
  await vi.waitFor(() => expect(optionTexts()).toEqual(['default', 'hollis']));
  expect(document.querySelector<HTMLSelectElement>('select[aria-label="Agent"]')!.value).toBe(
    'hollis',
  );
  await expect.element(page.getByLabelText('Starting message')).toBeInTheDocument();
});

test('a failed start shows an error toast and keeps the dialog open', async () => {
  vi.mocked(api.post).mockRejectedValue(new Error('boom'));
  mount({ initialAgent: '/ws/nightly.md' });
  await page.getByLabelText('Run prompt').fill('go');
  await page.getByTestId('launcher-start').click();

  await vi.waitFor(() => expect(toasts.items.some((t) => t.variant === 'err')).toBe(true));
  expect(onClose).not.toHaveBeenCalled();
});

test('a failed agent list toasts instead of rejecting unhandled', async () => {
  vi.mocked(api.get).mockImplementation(((path: string) =>
    path === '/api/models'
      ? Promise.resolve(MODELS)
      : Promise.reject(new Error('offline'))) as never);
  mount();
  await vi.waitFor(() =>
    expect(toasts.items.map((t) => [t.title, t.body])).toEqual([
      ['Could not load agents', 'offline'],
    ]),
  );
  await expect.element(page.getByTestId('launcher-start')).toBeDisabled();
});
