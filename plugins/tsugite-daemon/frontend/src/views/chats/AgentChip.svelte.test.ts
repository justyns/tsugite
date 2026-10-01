/// <reference types="@vitest/browser/context" />
import { page } from '@vitest/browser/context';
import { render, cleanup } from 'vitest-browser-svelte';
import { afterEach, beforeEach, expect, test, vi } from 'vitest';

vi.mock('$lib/api/client', () => ({
  api: { get: vi.fn(), patch: vi.fn() },
  authHeaders: () => ({}),
}));

import { api } from '$lib/api/client';
import { agentsMeta } from '$lib/stores/agentsMeta.svelte';
import { toasts } from '$lib/components/feedback/toast-store.svelte';
import AgentChip from './AgentChip.svelte';

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
    description: '',
    chat: true,
  },
  {
    path: '/ws/nightly.md',
    name: 'nightly',
    source: 'project',
    readonly: false,
    description: '',
    chat: false,
  },
];

let settings: { agent_file: string | null };

beforeEach(() => {
  settings = { agent_file: null };
  vi.mocked(api.get).mockReset();
  vi.mocked(api.patch).mockReset();
  vi.mocked(api.get).mockImplementation(((path: string) => {
    if (path === '/api/runtime') return Promise.resolve({ agent_file: 'default' });
    if (path === '/api/agent-files') return Promise.resolve({ files: FILES });
    if (path.endsWith('/settings')) return Promise.resolve(settings);
    return Promise.reject(new Error('unexpected GET ' + path));
  }) as never);
  vi.mocked(api.patch).mockImplementation((async (
    _p: string,
    body: { agent_file: string | null },
  ) => ({
    agent_file: body.agent_file,
  })) as never);
  agentsMeta.runtime = null;
  agentsMeta.agentFiles = [];
});

afterEach(cleanup);

const select = () => document.querySelector<HTMLSelectElement>('select[aria-label="Chat agent"]');

test('lists only chat-capable agents and selects the daemon default when none is set', async () => {
  render(AgentChip, { sessionId: 's1' });
  await vi.waitFor(() => expect(select()).not.toBeNull());
  expect([...select()!.options].map((o) => o.value)).toEqual(['default', 'hollis']);
  expect(select()!.value).toBe('default');
});

test('shows the session agent and switching PATCHes its agent_file', async () => {
  settings = { agent_file: '/ws/hollis.md' };
  render(AgentChip, { sessionId: 's1' });
  await vi.waitFor(() => expect(select()?.value).toBe('hollis'));

  await page.getByLabelText('Chat agent').selectOptions('default');
  await vi.waitFor(() =>
    expect(api.patch).toHaveBeenCalledWith('/api/sessions/s1/settings', { agent_file: null }),
  );

  await page.getByLabelText('Chat agent').selectOptions('hollis');
  await vi.waitFor(() =>
    expect(api.patch).toHaveBeenCalledWith('/api/sessions/s1/settings', {
      agent_file: '/ws/hollis.md',
    }),
  );
});

test('is hidden when there is no agent to switch to', async () => {
  vi.mocked(api.get).mockImplementation(((path: string) => {
    if (path === '/api/runtime') return Promise.resolve({ agent_file: 'default' });
    if (path === '/api/agent-files') return Promise.resolve({ files: [FILES[0], FILES[2]] });
    return Promise.resolve(settings);
  }) as never);
  render(AgentChip, { sessionId: 's1' });
  await vi.waitFor(() => expect(agentsMeta.agentFiles.length).toBe(2));
  expect(select()).toBeNull();
});

test('a failed agent list toasts instead of rejecting unhandled', async () => {
  toasts.items = [];
  vi.mocked(api.get).mockImplementation(((path: string) => {
    if (path === '/api/agent-files') return Promise.reject(new Error('offline'));
    return Promise.resolve(path === '/api/runtime' ? { agent_file: 'default' } : settings);
  }) as never);
  render(AgentChip, { sessionId: 's1' });
  await vi.waitFor(() =>
    expect(toasts.items.map((t) => [t.title, t.body])).toEqual([
      ['Could not load agents', 'offline'],
    ]),
  );
  expect(select()).toBeNull();
});
