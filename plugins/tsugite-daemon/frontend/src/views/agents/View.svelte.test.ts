/// <reference types="@vitest/browser/context" />
import { page } from '@vitest/browser/context';
import { render, cleanup } from 'vitest-browser-svelte';
import { afterEach, beforeEach, expect, test, vi } from 'vitest';

vi.mock('$lib/api/client', () => ({
  api: { get: vi.fn(), put: vi.fn() },
  authHeaders: () => ({}),
}));

import { api } from '$lib/api/client';
import { agentsMeta } from '$lib/stores/agentsMeta.svelte';
import { spaces } from '$lib/stores/spaces.svelte';
import View from './View.svelte';

const ODYN_PATH = '/ws/agents/odyn.md';
const DEFAULT_PATH = '/builtin/default.md';

const ODYN_SRC = `---
name: odyn
description: Primary interactive operator agent.
extends: default
model: claude_code:opus-4-8
reasoning_effort: medium
max_turns: 40
tools:
  - read_file
  - "@terminal"
---
You are odyn, the primary interactive agent.
`;

const DEFAULT_SRC = `---
name: default
description: Default base agent.
extends: none
max_turns: 10
---
Base agent body.
`;

const FILES = [
  {
    path: ODYN_PATH,
    name: 'odyn',
    source: 'project',
    readonly: false,
    description: 'Primary interactive operator agent.',
  },
  {
    path: '/ws/agents/ops-runner.md',
    name: 'ops-runner',
    source: 'project',
    readonly: false,
    description: 'Background automation agent.',
  },
  {
    path: DEFAULT_PATH,
    name: 'default',
    source: 'builtin',
    readonly: true,
    description: 'Default base agent.',
  },
];

const ROSTER = [
  { name: 'odyn', agent_file: 'agents/odyn.md', workspace_dir: '/ws', running_tasks: 2 },
];

function routeGet(path: string): Promise<unknown> {
  if (path === '/api/agents') return Promise.resolve({ agents: ROSTER });
  if (path === '/api/agent-files') return Promise.resolve({ files: FILES });
  if (path.startsWith('/api/agent-files/content')) {
    const target = decodeURIComponent(path);
    if (target.includes('default'))
      return Promise.resolve({ path: DEFAULT_PATH, content: DEFAULT_SRC, readonly: true });
    return Promise.resolve({ path: ODYN_PATH, content: ODYN_SRC, readonly: false });
  }
  return Promise.reject(new Error('unexpected GET ' + path));
}

afterEach(cleanup);

beforeEach(() => {
  vi.mocked(api.get).mockReset();
  vi.mocked(api.put).mockReset();
  vi.mocked(api.get).mockImplementation(routeGet as never);
  vi.mocked(api.put).mockResolvedValue(undefined as never);
  agentsMeta.agents = [];
  agentsMeta.agentFiles = [];
  agentsMeta.loading = false;
  agentsMeta.error = null;
});

test('roster lists agent files registered-first, and auto-selects the top agent into the Form tab', async () => {
  render(View);
  await expect.element(page.getByTestId('agent-row-odyn')).toBeInTheDocument();
  // Registered odyn carries a running badge; builtin default carries a lock.
  await expect.element(page.getByTestId('agent-row-odyn')).toHaveTextContent('2');
  // Auto-selected odyn's Form tab shows its model + inheritance note.
  await expect.element(page.getByText('claude_code:opus-4-8')).toBeInTheDocument();
  await expect.element(page.getByText(/Inherits from/)).toBeInTheDocument();
});

test('switching to the Markdown tab shows the editable source; Save is disabled until dirty', async () => {
  const { container } = await render(View);
  await expect.element(page.getByText('claude_code:opus-4-8')).toBeInTheDocument();

  await page.getByRole('button', { name: 'markdown' }).click();

  const ta = container.querySelector<HTMLTextAreaElement>('.agent-src');
  await vi.waitFor(() => expect(ta).not.toBeNull());
  expect(ta!.value).toContain('name: odyn');
  expect(ta!.readOnly).toBe(false);

  const save = page.getByTestId('agent-save');
  await expect.element(save).toBeDisabled();

  // Editing makes the buffer dirty and enables Save.
  ta!.value = ODYN_SRC + '\n- extra rule\n';
  ta!.dispatchEvent(new Event('input', { bubbles: true }));
  await expect.element(save).toBeEnabled();
});

test('a builtin file is read-only: the editor blocks edits and Save stays disabled', async () => {
  const { container } = await render(View);
  await expect.element(page.getByTestId('agent-row-default')).toBeInTheDocument();

  await page.getByTestId('agent-row-default').click();
  await page.getByRole('button', { name: 'markdown' }).click();

  await expect.element(page.getByText(/This is a builtin agent/)).toBeInTheDocument();
  const ta = container.querySelector<HTMLTextAreaElement>('.agent-src');
  await vi.waitFor(() => expect(ta?.value).toContain('name: default'));
  expect(ta!.readOnly).toBe(true);
  await expect.element(page.getByTestId('agent-save')).toBeDisabled();
});

test('the editor offers exactly form and markdown modes (no preview tab)', async () => {
  render(View);
  await expect.element(page.getByText('claude_code:opus-4-8')).toBeInTheDocument();

  await expect.element(page.getByRole('button', { name: 'form' })).toBeInTheDocument();
  await expect.element(page.getByRole('button', { name: 'markdown' })).toBeInTheDocument();
  await expect.element(page.getByRole('button', { name: 'preview' })).not.toBeInTheDocument();
});

test('Run opens the launcher, whose Start is gated on a non-empty prompt and docks a chat surface', async () => {
  const openSpy = vi.spyOn(spaces, 'open');
  render(View);
  await expect.element(page.getByTestId('agent-run')).toBeInTheDocument();

  await page.getByTestId('agent-run').click();
  const start = page.getByRole('button', { name: /Start chat/ });
  await expect.element(start).toBeInTheDocument();
  await expect.element(start).toBeDisabled();

  await page.getByLabelText('Run prompt').fill('summarise open incidents');
  await expect.element(start).toBeEnabled();
  await start.click();

  expect(openSpy).toHaveBeenCalledTimes(1);
  const ref = openSpy.mock.calls[0]![0] as { kind: string; params?: Record<string, string> };
  expect(ref.kind).toBe('chat');
  expect(ref.params?.agent).toBe('odyn');
  expect(ref.params?.prompt).toBe('summarise open incidents');
  openSpy.mockRestore();
});

test('the roster filter narrows the list by name', async () => {
  render(View);
  await expect.element(page.getByTestId('agent-row-odyn')).toBeInTheDocument();

  await page.getByRole('searchbox', { name: 'Filter agents' }).fill('ops');
  await expect.element(page.getByTestId('agent-row-ops-runner')).toBeInTheDocument();
  expect(page.getByTestId('agent-row-odyn').elements()).toHaveLength(0);
});
