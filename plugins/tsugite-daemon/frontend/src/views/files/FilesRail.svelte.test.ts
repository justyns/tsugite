/// <reference types="@vitest/browser/context" />
import { page, userEvent } from '@vitest/browser/context';
import { render } from 'vitest-browser-svelte';
import { expect, test, vi, beforeEach } from 'vitest';
import { WORKSPACE } from './__fixtures__/workspace';
import { buildTree } from './wiki';

vi.mock('$lib/api/client', () => ({ authHeaders: () => ({}), api: WORKSPACE.api }));

beforeEach(async () => {
  await page.viewport(1440, 900);
  const { filesShowIgnored } = await import('$lib/stores/filesShowIgnored.svelte');
  filesShowIgnored.set(false);
  const { agentsMeta } = await import('$lib/stores/agentsMeta.svelte');
  agentsMeta.runtime = null;
  const { filesWorkspace } = await import('./workspace.svelte');
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
  const opened: string[] = [];
  const { default: FilesRail } = await import('./FilesRail.svelte');
  render(FilesRail, {
    props: {
      focusedFilePath: null,
      onOpenFile: (path: string) => opened.push(path),
      onPinFile: () => {},
    },
  });
  await page.getByTestId('file-node-ops/alpha.md').click();
  expect(opened).toContain('ops/alpha.md');
});

test('a file double-click asks the shell to pin it (preview-tab keep)', async () => {
  const pinned: string[] = [];
  const { default: FilesRail } = await import('./FilesRail.svelte');
  render(FilesRail, {
    props: {
      focusedFilePath: null,
      onOpenFile: () => {},
      onPinFile: (path: string) => pinned.push(path),
    },
  });
  await userEvent.dblClick(page.getByTestId('file-node-ops/alpha.md'));
  expect(pinned).toContain('ops/alpha.md');
});

test('a workspace reload keeps a collapsed directory collapsed', async () => {
  await mountRail();
  await expect.element(page.getByTestId('file-node-ops/alpha.md')).toBeInTheDocument();

  await page.getByRole('button', { name: 'ops', exact: true }).click();
  await expect.element(page.getByTestId('file-node-ops/alpha.md')).not.toBeInTheDocument();

  const { filesWorkspace } = await import('./workspace.svelte');
  const before = filesWorkspace.ws!;
  const dir = before.entries.find((e) => e.is_dir)!;
  const file = before.entries.find((e) => !e.is_dir)!;
  const entries = [
    ...before.entries,
    { ...dir, path: 'notes', name: 'notes' },
    { ...file, path: 'notes/todo.md', name: 'todo.md' },
  ];
  filesWorkspace.ws = { ...before, entries, tree: buildTree(entries) };

  await expect.element(page.getByTestId('file-node-notes/todo.md')).toBeInTheDocument();
  expect(page.getByTestId('file-node-ops/alpha.md').elements()).toHaveLength(0);
});

test('the Show ignored files switch reloads the tree with ignored entries, dimmed', async () => {
  await mountRail();
  await expect.element(page.getByTestId('file-node-index.md')).toBeInTheDocument();
  expect(page.getByTestId('file-node-debug.log').elements()).toHaveLength(0);

  await page.getByRole('switch', { name: 'Show ignored files' }).click();

  const node = page.getByTestId('file-node-debug.log');
  await expect.element(node).toBeInTheDocument();
  await expect.element(node).toHaveClass('is-ignored');
  await expect.element(page.getByTestId('file-node-index.md')).not.toHaveClass('is-ignored');
  const { filesShowIgnored } = await import('$lib/stores/filesShowIgnored.svelte');
  expect(filesShowIgnored.enabled).toBe(true);

  await page.getByRole('switch', { name: 'Show ignored files' }).click();
  await expect.element(page.getByTestId('file-node-debug.log')).not.toBeInTheDocument();
});

test('right-click Download on a file saves it through the files store', async () => {
  const { files } = await import('$lib/stores/files.svelte');
  const download = vi.spyOn(files, 'download').mockResolvedValue();
  await mountRail();
  await expect.element(page.getByTestId('file-node-ops/alpha.md')).toBeInTheDocument();

  await page.getByTestId('file-node-ops/alpha.md').click({ button: 'right' });
  await page.getByRole('menuitem', { name: 'Download' }).click();

  expect(download).toHaveBeenCalledWith('ops/alpha.md');
  download.mockRestore();
});

test('a directory row has no file menu', async () => {
  await mountRail();
  await expect.element(page.getByRole('button', { name: 'ops', exact: true })).toBeInTheDocument();
  await page.getByRole('button', { name: 'ops', exact: true }).click({ button: 'right' });
  expect(page.getByRole('menuitem', { name: 'Download' }).elements()).toHaveLength(0);
});
