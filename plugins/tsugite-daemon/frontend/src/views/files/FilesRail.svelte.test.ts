/// <reference types="@vitest/browser/context" />
import { page, userEvent } from '@vitest/browser/context';
import { render } from 'vitest-browser-svelte';
import { expect, test, vi, beforeEach } from 'vitest';
import { WORKSPACE } from './__fixtures__/workspace';

vi.mock('$lib/api/client', () => ({ authHeaders: () => ({}), api: WORKSPACE.api }));

beforeEach(async () => {
  await page.viewport(1440, 900);
  const { agentsMeta } = await import('$lib/stores/agentsMeta.svelte');
  agentsMeta.agents = [];
  const { filesWorkspace } = await import('./workspace.svelte');
  filesWorkspace.agent = '';
  filesWorkspace.ws = null;
  filesWorkspace.loading = false;
  filesWorkspace.error = null;
  filesWorkspace.indexState = 'none';
});

async function mountRail() {
  const { default: FilesRail } = await import('./FilesRail.svelte');
  render(FilesRail, {
    props: { focusedFilePath: null, onOpenFile: () => {}, onPinFile: () => {} },
  });
}

test('lists the workspace tree (dirs expanded, files as nodes)', async () => {
  await mountRail();
  await expect.element(page.getByTestId('file-node-index.md')).toBeInTheDocument();
  await expect.element(page.getByTestId('file-node-ops/alpha.md')).toBeInTheDocument();
});

test('a #tag search kicks the on-demand index scan and filters to notes carrying that tag', async () => {
  await mountRail();
  await expect.element(page.getByTestId('file-node-ops/alpha.md')).toBeInTheDocument();

  await page.getByRole('searchbox', { name: 'Search workspace' }).fill('#x');
  await expect.element(page.getByTestId('file-node-ops/alpha.md')).toBeInTheDocument();
  await expect.element(page.getByTestId('file-node-ops/beta.md')).toBeInTheDocument();
  await expect.element(page.getByTestId('file-node-index.md')).not.toBeInTheDocument();
});

test('a file click asks the shell to open it as a surface', async () => {
  const opened: Array<[string, string]> = [];
  const { default: FilesRail } = await import('./FilesRail.svelte');
  render(FilesRail, {
    props: {
      focusedFilePath: null,
      onOpenFile: (agent: string, path: string) => opened.push([agent, path]),
      onPinFile: () => {},
    },
  });
  await page.getByTestId('file-node-ops/alpha.md').click();
  expect(opened).toContainEqual(['smoke', 'ops/alpha.md']);
});

test('a file double-click asks the shell to pin it (preview-tab keep)', async () => {
  const pinned: Array<[string, string]> = [];
  const { default: FilesRail } = await import('./FilesRail.svelte');
  render(FilesRail, {
    props: {
      focusedFilePath: null,
      onOpenFile: () => {},
      onPinFile: (agent: string, path: string) => pinned.push([agent, path]),
    },
  });
  await userEvent.dblClick(page.getByTestId('file-node-ops/alpha.md'));
  expect(pinned).toContainEqual(['smoke', 'ops/alpha.md']);
});
