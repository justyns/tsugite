/// <reference types="@vitest/browser/context" />
import { page } from '@vitest/browser/context';
import { render, cleanup } from 'vitest-browser-svelte';
import { afterEach, beforeEach, expect, test, vi } from 'vitest';

vi.mock('$lib/api/client', () => ({
  api: { get: vi.fn(), post: vi.fn(), patch: vi.fn(), put: vi.fn() },
  authHeaders: () => ({}),
}));

import { api } from '$lib/api/client';
import { agents, type Agent } from '$lib/stores/agents.svelte';
import { agentsMeta } from '$lib/stores/agentsMeta.svelte';
import { sessions } from '$lib/stores/sessions.svelte';
import AgentRecords from './AgentRecords.svelte';

const MAIN: Agent = {
  id: 'main',
  name: 'Main',
  workspace: '/ws',
  template: 'default',
  model: null,
  reasoning_effort: null,
  color: 'blue',
  created_at: '',
  is_default: true,
};
const HOMELAB: Agent = {
  id: 'homelab',
  name: 'Homelab',
  workspace: '/data/agents/homelab',
  template: 'default',
  model: 'anthropic:claude-x',
  reasoning_effort: 'low',
  color: 'green',
  created_at: '',
  is_default: false,
};
const FILES = ['default', 'ops'].map((name) => ({
  path: `/ws/${name}.md`,
  name,
  source: 'project',
  readonly: false,
  description: '',
}));

let roster: Agent[];

afterEach(cleanup);

beforeEach(() => {
  roster = [MAIN, HOMELAB];
  location.hash = '';
  for (const fn of [api.get, api.post, api.patch]) vi.mocked(fn).mockReset();
  vi.mocked(api.get).mockImplementation((async (path: string) => {
    if (path === '/api/agents') return { agents: roster };
    if (path === '/api/agent-files') return { files: FILES };
    throw new Error('unexpected GET ' + path);
  }) as never);
  agents.list = [];
  agents.error = null;
  agentsMeta.agentFiles = [];
});

test('lists each agent with its id, model, color dot and a default badge', async () => {
  const { container } = await render(AgentRecords);

  const main = page.getByTestId('agent-record-main');
  await expect.element(main).toHaveTextContent('Main');
  await expect.element(main).toHaveTextContent('default');
  await expect
    .element(page.getByTestId('agent-record-homelab'))
    .toHaveTextContent('homelab · anthropic:claude-x');
  const dots = container.querySelectorAll('.ag-row .agent-dot');
  expect([...dots].map((d) => d.getAttribute('data-color'))).toEqual(['blue', 'green']);
});

test('create validates the id client-side and shows the server error', async () => {
  vi.mocked(api.post).mockRejectedValue(new Error('template not found: ops'));
  await render(AgentRecords);
  await page.getByTestId('agent-create').click();
  const create = page.getByTestId('agent-record-save');

  await page.getByLabelText('id', { exact: true }).fill('Bad Id');
  await expect.element(page.getByText(/Use lowercase letters/)).toBeInTheDocument();
  await expect.element(create).toBeDisabled();

  await page.getByLabelText('id', { exact: true }).fill('atlas');
  await page.getByLabelText('template').selectOptions('ops');
  await expect.element(create).toBeEnabled();
  await create.click();

  await expect
    .element(page.getByTestId('agent-record-error'))
    .toHaveTextContent('template not found: ops');
  expect(api.get).toHaveBeenCalledTimes(2);
});

test('create posts the form, refreshes the roster and selects the new agent', async () => {
  const atlas = {
    ...HOMELAB,
    id: 'atlas',
    name: 'Atlas',
    model: null,
    reasoning_effort: null,
    color: 'teal',
  };
  vi.mocked(api.post).mockImplementation((async () => {
    roster = [MAIN, HOMELAB, atlas];
    return atlas;
  }) as never);
  await render(AgentRecords);
  await page.getByTestId('agent-create').click();

  await page.getByLabelText('id', { exact: true }).fill('atlas');
  await page.getByLabelText('name').fill('Atlas');
  await page.getByLabelText('color').selectOptions('teal');
  await page.getByLabelText('identity').fill('I map the homelab.');
  await page.getByTestId('agent-record-save').click();

  await expect.element(page.getByTestId('agent-record-atlas')).toBeInTheDocument();
  expect(api.post).toHaveBeenCalledWith('/api/agents', {
    id: 'atlas',
    name: 'Atlas',
    model: null,
    reasoning_effort: null,
    color: 'teal',
    identity: 'I map the homelab.',
  });
  await expect.element(page.getByText('agents / atlas')).toBeInTheDocument();
});

test('edit sends only the form fields, keeps id and workspace read-only and refreshes the store', async () => {
  vi.mocked(api.patch).mockImplementation((async () => {
    const renamed = { ...HOMELAB, name: 'Lab', model: null };
    roster = [MAIN, renamed];
    return renamed;
  }) as never);
  await render(AgentRecords);
  await page.getByTestId('agent-record-homelab').click();

  await expect.element(page.getByLabelText('id', { exact: true })).toBeDisabled();
  await expect.element(page.getByText('/data/agents/homelab')).toBeInTheDocument();
  const save = page.getByTestId('agent-record-save');
  await expect.element(save).toBeDisabled();

  await page.getByLabelText('name').fill('Lab');
  await page.getByLabelText('model').fill('');
  await save.click();

  expect(api.patch).toHaveBeenCalledWith('/api/agents/homelab', {
    name: 'Lab',
    template: 'default',
    model: null,
    reasoning_effort: 'low',
    color: 'green',
  });
  await expect.element(page.getByTestId('agent-record-homelab')).toHaveTextContent('Lab');
  expect(agents.list.map((a) => a.name)).toEqual(['Main', 'Lab']);
});

test("the default agent's template is read-only and left out of the patch", async () => {
  vi.mocked(api.patch).mockResolvedValue(MAIN as never);
  await render(AgentRecords);
  await expect.element(page.getByTestId('agent-record-main')).toBeInTheDocument();

  await expect.element(page.getByLabelText('template')).toBeDisabled();
  await page.getByLabelText('name').fill('Primary');
  await page.getByTestId('agent-record-save').click();

  expect(vi.mocked(api.patch).mock.calls[0]![1]).not.toHaveProperty('template');
});

test('Open main chat opens the agent main session', async () => {
  vi.mocked(api.post).mockResolvedValue({ id: 'sess-9' } as never);
  vi.spyOn(sessions, 'load').mockResolvedValue(undefined as never);
  await render(AgentRecords);
  await page.getByTestId('agent-record-homelab').click();

  await page.getByTestId('agent-open-main').click();

  await vi.waitFor(() => expect(location.hash).toContain('sessionId=sess-9'));
  expect(api.post).toHaveBeenCalledWith('/api/agents/homelab/main');
});

test('an agent with the id "new" is edited with a PATCH, not created', async () => {
  const odd = { ...HOMELAB, id: 'new', name: 'New' };
  roster = [MAIN, odd];
  vi.mocked(api.patch).mockResolvedValue({ ...odd, name: 'Newer' } as never);
  await render(AgentRecords);
  await page.getByTestId('agent-record-new').click();

  await page.getByLabelText('name').fill('Newer');
  await page.getByTestId('agent-record-save').click();

  expect(api.post).not.toHaveBeenCalled();
  expect(vi.mocked(api.patch).mock.calls[0]![0]).toBe('/api/agents/new');
});

test('a save that resolves after another row is selected leaves that row untouched', async () => {
  let resolve!: (a: Agent) => void;
  vi.mocked(api.patch).mockImplementation(
    (() => new Promise<Agent>((r) => (resolve = r))) as never,
  );
  await render(AgentRecords);
  await page.getByTestId('agent-record-homelab').click();
  await page.getByLabelText('name').fill('Lab');
  await page.getByTestId('agent-record-save').click();

  await page.getByTestId('agent-record-main').click();
  roster = [MAIN, { ...HOMELAB, name: 'Lab' }];
  resolve({ ...HOMELAB, name: 'Lab' });
  await vi.waitFor(() => expect(agents.list.find((a) => a.id === 'homelab')?.name).toBe('Lab'));
  await new Promise((r) => setTimeout(r, 0));

  await expect.element(page.getByLabelText('name')).toHaveValue('Main');
  await expect.element(page.getByLabelText('id', { exact: true })).toHaveValue('main');
});
